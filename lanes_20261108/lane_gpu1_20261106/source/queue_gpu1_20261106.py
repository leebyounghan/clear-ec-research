"""Job queue for GPU 1 (lane GPU 1, 2026-11-06). Runs lines of queue.tsv in order, one at a time: name<TAB>shell command. Add lines at any time; this script is never edited while it runs.
A finished job is appended to done.tsv with its return code. When nothing is left and the GPU is free, the keepalive runs (pid in keepalive.pid) and is stopped by pid before the next job."""
import os, signal, subprocess, time
from pathlib import Path

L = Path('/workspace/clear-ec-research/experiments/lane_gpu1_20261106')
ENV = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='2', PYTHONPATH='/workspace/clear-ec-research/experiments/phase2_foundation_20261002/runtime:experiments/lane_gpu1_20261106/source')
KEEP = ['/workspace/venv/bin/python', 'experiments/phase2_pooling_replication_20261026/gpu_keepalive.py', '--ignore-jobs', '--duty', '0.35', '--minutes', '1440']
keep = None


def done():
    p = L / 'done.tsv'
    return {r.split('\t')[0] for r in p.read_text().splitlines()} if p.exists() else set()


def todo():
    p = L / 'queue.tsv'
    d = done()
    for r in (p.read_text().splitlines() if p.exists() else []):
        if r.strip() and not r.startswith('#') and r.split('\t')[0] not in d:
            return r.split('\t', 1)
    return None


def gpu_free():
    out = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).split()
    return not [q for q in out if keep is None or q != str(keep.pid)]


while True:
    job = todo()
    if job is None:
        if keep is None and gpu_free():
            keep = subprocess.Popen(KEEP, cwd='/workspace/clear-ec-research', env=ENV, stdout=(L / 'keepalive.log').open('a'), stderr=subprocess.STDOUT)
            (L / 'keepalive.pid').write_text(str(keep.pid))
        time.sleep(30)
        continue
    while not gpu_free():
        time.sleep(20)
    if keep is not None:
        keep.send_signal(signal.SIGTERM); keep.wait(); keep = None
    t = time.time()
    with (L / 'logs' / f'{job[0]}.log').open('a') as log:
        rc = subprocess.call(['bash', '-c', job[1]], cwd='/workspace/clear-ec-research', env=ENV, stdout=log, stderr=subprocess.STDOUT)
    with (L / 'done.tsv').open('a') as f:
        f.write(f'{job[0]}\t{rc}\t{round(time.time() - t)}\t{time.strftime("%FT%TZ", time.gmtime())}\n')
