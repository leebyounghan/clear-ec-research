"""Sequential job queue for the sweep GPU (2026-11-05). Re-reads the job list before every job, so jobs can be appended while it runs. A job is finished when its marker exists; a failure is logged and skipped, never retried. Keeps the GPU loaded by real jobs; when the list is exhausted it waits and re-reads, and the caller decides what runs next.
usage: sweep_queue_20261105.py GPU JOBS.json LOG_DIR"""
import json, os, subprocess, sys, time
from pathlib import Path
gpu, jobs, logdir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]); logdir.mkdir(parents=True, exist_ok=True); tried = set(); env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
while True:
    todo = [j for j in json.loads(jobs.read_text()) if j['name'] not in tried and not (j.get('done') and Path(j['done']).exists())]
    if not todo:
        time.sleep(60); continue
    j = todo[0]; tried.add(j['name']); print(time.strftime('%FT%TZ', time.gmtime()), 'start', j['name'], flush=True)
    with open(logdir / f"{j['name']}.log", 'a') as f:
        rc = subprocess.call(j['cmd'], stdout=f, stderr=subprocess.STDOUT, env=env, cwd='/workspace/clear-ec-research')
    print(time.strftime('%FT%TZ', time.gmtime()), 'end', j['name'], 'rc', rc, flush=True)
