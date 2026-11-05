"""Resumable archive download only; no local training, model loading or tests."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1] / 'submission_candidate_raw_20260913'
REMOTE = '/workspace/clear-ec-research/experiments/raw_ssl_next_20260913/artifacts/'
POD = 'clear-ec-ssl-raw-opt-0913-tfhx2'
CHUNK = 64 * 1024**2


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((ROOT / 'model_archive.json').read_text())
    name, size, expected = (manifest[k] for k in ('name', 'bytes', 'sha256'))
    assert name == 'model_full_quality77_raw_20260913.tar.gz'
    assert size == 7964624301
    assert expected == '5bcf45dc1f057ef431ce9504a31b6506e671e332bb9c0d698644bf662370cec3'
    final = ROOT / name
    if final.exists():
        assert final.stat().st_size == size and digest(final) == expected
        print('Existing final archive verified', flush=True)
        return
    partial = ROOT / (name + '.partial')
    parts = ROOT / 'transfer_ranges'
    parts.mkdir(exist_ok=True)
    plan_file = parts / 'plan.json'
    if plan_file.exists():
        plan = json.loads(plan_file.read_text())
        assert plan['sha256'] == expected and plan['bytes'] == size
    else:
        plan = dict(prefix=partial.stat().st_size if partial.exists() else 0,
                    sha256=expected, bytes=size, chunk=CHUNK)
        with plan_file.open('x') as stream:
            json.dump(plan, stream, indent=2)
    assert plan['prefix'] <= (partial.stat().st_size if partial.exists() else 0) <= size
    jobs = [(offset, min(CHUNK, size-offset)) for offset in range(plan['prefix'], size, CHUNK)]
    started = time.monotonic()

    def fetch(job):
        offset, count = job
        path = parts / f'{offset}.part'
        current = path.stat().st_size if path.exists() else 0
        assert current <= count
        for attempt in range(8):
            if current == count:
                return count
            command = ['kubectl', '-n', 'p-ajou-mlxp-project', 'exec', POD,
                       '-c', 'researcher', '--', 'dd', 'if=' + REMOTE + name,
                       'bs=1048576', 'iflag=skip_bytes,count_bytes',
                       f'skip={offset+current}', f'count={count-current}', 'status=none']
            with path.open('ab') as stream:
                p = subprocess.Popen(command, stdout=stream, stderr=subprocess.PIPE)
                try:
                    _, error = p.communicate(timeout=180)
                except subprocess.TimeoutExpired:
                    p.terminate()
                    _, error = p.communicate(timeout=15)
            current = path.stat().st_size
            assert current <= count
            if current == count:
                return count
        raise RuntimeError(f'Incomplete range {offset}: {current}/{count}; {error[-300:]!r}')

    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(fetch, job) for job in jobs]):
            done += future.result()
            print(f'Ranges {done}/{size-plan["prefix"]} bytes; elapsed {time.monotonic()-started:.0f}s', flush=True)
    current = partial.stat().st_size if partial.exists() else 0
    with partial.open('ab') as output:
        for offset, count in jobs:
            path = parts / f'{offset}.part'
            assert path.stat().st_size == count
            if offset + count <= current:
                continue
            with path.open('rb') as stream:
                stream.seek(max(0, current-offset))
                shutil.copyfileobj(stream, output)
            current = offset + count
    assert partial.stat().st_size == size
    actual = digest(partial)
    assert actual == expected, f'Archive checksum mismatch: {actual}'
    assert not final.exists()
    partial.rename(final)
    with (ROOT / 'download_verified.json').open('x') as stream:
        json.dump(dict(name=name, bytes=size, sha256=actual,
                       verified_at=time.strftime('%FT%T%z')), stream, indent=2)
    # Only transfer fragments created by this downloader; archive remains intact.
    for offset, _ in jobs:
        (parts / f'{offset}.part').unlink()
    print(f'COMPLETE: SHA256 verified {final}', flush=True)


if __name__ == '__main__':
    main()
