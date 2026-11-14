"""Second job stream on GPU 1 for short SSL arms (lane GPU 1, 2026-11-07). Lines of queue_short.tsv in order, one at a time, next to the main queue's long job: the GPU has the memory, total work is the same and short reads arrive sooner.
No keepalive here; the main queue owns that. Finished jobs go to done_short.tsv. Exits never: idle lines cost nothing."""
import os, subprocess, time
from pathlib import Path

L = Path('/workspace/clear-ec-research/experiments/lane_gpu1_20261106')
ENV = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='2', PYTHONPATH='/workspace/clear-ec-research/experiments/phase2_foundation_20261002/runtime:experiments/lane_gpu1_20261106/source')


def todo():
    d = (L / 'done_short.tsv'); done = {r.split('\t')[0] for r in d.read_text().splitlines()} if d.exists() else set()
    q = L / 'queue_short.tsv'
    for r in (q.read_text().splitlines() if q.exists() else []):
        if r.strip() and not r.startswith('#') and r.split('\t')[0] not in done:
            return r.split('\t', 1)
    return None


while True:
    job = todo()
    if job is None:
        time.sleep(30)
        continue
    t = time.time()
    with (L / 'logs' / f'{job[0]}.log').open('a') as log:
        rc = subprocess.call(['bash', '-c', job[1]], cwd='/workspace/clear-ec-research', env=ENV, stdout=log, stderr=subprocess.STDOUT)
    with (L / 'done_short.tsv').open('a') as f:
        f.write(f'{job[0]}\t{rc}\t{round(time.time() - t)}\t{time.strftime("%FT%TZ", time.gmtime())}\n')
