"""Model archive transfer only: bounded-memory parallel ranges, single disk copy.

Adapted from fetch_raw_release_20260913; no local model/data computation.
Incomplete ranges are re-fetched. Final SHA256 is mandatory before promotion.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1] / 'submission_candidate_ssl_raw_20260914'
REMOTE = '/workspace/clear-ec-research/experiments/ssl_release_20260914/'
POD = 'clear-ec-target-preflight-0912'
NAME = 'model_full_quality77_ssl_raw_20260914.tar.gz'
SIZE = 7964665157
SHA = 'e37c83db830cc5fa5215f7fd63267beb11e6c09a8778aefe2b8c6180a37a9730'
CHUNK = 64 * 1024**2
BASE = ['kubectl', '-n', 'p-ajou-mlxp-project', 'exec', POD, '-c', 'researcher', '--']
# Same cluster transport setting as the previously verified range downloader.
os.environ['KUBECTL_REMOTE_COMMAND_WEBSOCKETS'] = 'false'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def retain(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, f'Existing metadata differs: {path}'
    else:
        with path.open('xb') as stream:
            stream.write(data)


def main():
    ROOT.mkdir(exist_ok=True)
    manifest_bytes = subprocess.check_output(BASE + ['cat', REMOTE + 'artifacts/model_archive.json'], timeout=60)
    m = json.loads(manifest_bytes)
    assert (m['name'], m['bytes'], m['sha256']) == (NAME, SIZE, SHA)
    assert m['archived_hashes_verified'] and m['calibration_unchanged'] and len(m['members']) == 8
    assert m['ssl_images'] == 9000 and m['supervised_images'] == 8998
    retain(ROOT / 'model_archive.json', manifest_bytes)
    for name in ['manifest.json', 'gpu_complete.json', 'package_ready.json', 'stage_manifest.json',
                 'pretrain/result.json', 'pretrain/coverage.json', 'raw_deploy/result.json',
                 'raw_deploy/config.json', 'artifacts/SHA256SUMS.txt']:
        data = subprocess.check_output(BASE + ['cat', REMOTE + name], timeout=60)
        retain(ROOT / (name.removeprefix('artifacts/')), data)
    # Runtime per-image records stay on PVC. Export an aggregate only.
    code = ('import json; p=json.load(open(' + repr(REMOTE + 'runtime_check/result.json') + ')); '
            'q={k:v for k,v in p.items() if k != "records"}; '
            'q["checked_images"]=len(p["records"]); '
            'q["maximum_relative_precision_drift"]=max(r["relative_precision_drift"] for r in p["records"]); '
            'assert q["passed"] and q["unchanged_first_five_exact"]; print(json.dumps(q,indent=2))')
    retain(ROOT / 'runtime_summary.json', subprocess.check_output(BASE + ['/workspace/venv/bin/python', '-c', code], timeout=60))
    final, partial = ROOT / NAME, ROOT / (NAME + '.partial')
    if final.exists():
        assert final.stat().st_size == SIZE and digest(final) == SHA
        print('Existing final SHA256 verified', flush=True)
        return
    state_path = ROOT / 'transfer_plan.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        assert (state['sha256'], state['bytes'], state['chunk']) == (SHA, SIZE, CHUNK)
        assert partial.exists() and partial.stat().st_size == SIZE
    else:
        assert not partial.exists(), 'Untracked partial; do not overwrite'
        assert shutil.disk_usage(ROOT).free >= SIZE + 1024**3, 'Need archive size plus 1GiB free'
        state = dict(sha256=SHA, bytes=SIZE, chunk=CHUNK, completed=[])
        with partial.open('xb') as stream:
            stream.truncate(SIZE)
        with state_path.open('x') as stream:
            json.dump(state, stream)
    started = time.monotonic()
    jobs = [(off, min(CHUNK, SIZE-off)) for off in range(0, SIZE, CHUNK) if off not in state['completed']]

    def fetch(job):
        off, count = job
        received, failures = 0, 0
        range_start = time.monotonic()
        with partial.open('r+b') as stream:
            stream.seek(off)
            while received < count:
                assert time.monotonic() - range_start < 900, 'Range time limit'
                amount = min(2 * 1024**2, count - received)
                # Short chunks and a drain interval avoid premature exec stream close.
                command = BASE + ['bash', '-lc',
                    f'dd if={REMOTE}artifacts/{NAME} bs=1048576 iflag=skip_bytes,count_bytes '
                    f'skip={off+received} count={amount} status=none; sleep 2']
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    data, error = process.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    data, error = process.communicate(timeout=10)
                assert len(data) <= amount
                if not data:
                    failures += 1
                    assert failures < 8, f'No transfer progress: {error[-200:]!r}'
                    continue
                stream.write(data)
                received += len(data)
            stream.flush()
            os.fsync(stream.fileno())
        return off

    with ThreadPoolExecutor(max_workers=16) as pool:
        for future in as_completed([pool.submit(fetch, job) for job in jobs]):
            state['completed'].append(future.result())
            temporary = ROOT / 'transfer_plan.json.tmp'
            temporary.write_text(json.dumps(state))
            temporary.replace(state_path)
            done = sum(min(CHUNK, SIZE-off) for off in state['completed'])
            print(f'Ranges {done}/{SIZE} bytes; elapsed {time.monotonic()-started:.0f}s', flush=True)
    assert partial.stat().st_size == SIZE and digest(partial) == SHA, 'Final SHA256 mismatch'
    assert not final.exists()
    partial.rename(final)
    retain(ROOT / 'download_verified.json', json.dumps(dict(name=NAME, bytes=SIZE, sha256=SHA,
           verified_at=datetime.now(timezone.utc).isoformat()), indent=2).encode())
    print(f'COMPLETE: SHA256 verified {final}', flush=True)


if __name__ == '__main__':
    main()
