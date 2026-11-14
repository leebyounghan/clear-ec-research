"""GPU 3 job queue (lane GPU 3, 2026-11-06). Re-reads jobs.json before every job, so jobs can be appended while it runs. A job is done when its marker exists; a failure is logged and skipped, never retried.
When the list is exhausted it starts this lane's keepalive (pid in keepalive.pid) and stops it by pid before the next job. Run from /workspace/clear-ec-research."""
import json, os, signal, subprocess, time
from pathlib import Path

LANE = Path('experiments/lane_gpu3_20261106'); JOBS = LANE / 'jobs.json'; LOGS = LANE / 'logs'; LOGS.mkdir(exist_ok=True)
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='3', OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
KEEP = ['/workspace/venv/bin/python', 'experiments/phase2_pooling_replication_20261026/gpu_keepalive.py', '--ignore-jobs', '--duty', '0.35', '--minutes', '1440']
tried, keep = set(), None


def say(*a):
    print(time.strftime('%FT%TZ', time.gmtime()), *a, flush=True)


while True:
    todo = [j for j in json.loads(JOBS.read_text()) if j['name'] not in tried and not (j.get('done') and Path(j['done']).exists())]
    if not todo:
        if keep is None or keep.poll() is not None:
            keep = subprocess.Popen(KEEP, stdout=open(LANE / 'keepalive.log', 'a'), stderr=subprocess.STDOUT, env=ENV, start_new_session=True)
            (LANE / 'keepalive.pid').write_text(str(keep.pid)); say('idle: keepalive', keep.pid)
        time.sleep(60); continue
    if keep is not None and keep.poll() is None:
        keep.send_signal(signal.SIGTERM); keep.wait(timeout=60); say('keepalive stopped', keep.pid)
    keep = None
    j = todo[0]; tried.add(j['name']); say('start', j['name'])
    with open(LOGS / f"{j['name']}.log", 'a') as f:
        rc = subprocess.call(j['cmd'], stdout=f, stderr=subprocess.STDOUT, env=ENV)
    say('end', j['name'], 'rc', rc)
