"""Sequential single-GPU job queue with exclusive GPU lock and keepalive while waiting.

jobs.json: {"gpu": "3", "log_dir": "...", "jobs": [{"name": ..., "wait_for": [paths], "skip_if": [paths],
"cmd": [...], "env": {...}}]}. For each job: run gpu_keepalive.py on the GPU until every wait_for
path exists, stop it, skip if any skip_if path exists, else run cmd with CUDA_VISIBLE_DEVICES set.
Holds experiments/phase2_backbone_ssl_20261020/gpu<N>.lock for the whole queue. No retries; each
job's exit code is recorded; after the queue a keepalive is left running on the GPU.
"""
import datetime
import fcntl
import json
import os
import pathlib
import subprocess
import sys
import time

PY = '/workspace/venv/bin/python'
KEEPALIVE = 'experiments/phase2_pooling_replication_20261026/gpu_keepalive.py'
LOCKDIR = pathlib.Path('experiments/phase2_backbone_ssl_20261020')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def gpu_env(gpu, extra=None):
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4')
    env.update(extra or {})
    return env


def busy(gpu):
    return subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()


def start_keepalive(gpu, log_dir):
    log = (log_dir/'keepalive.log').open('a')
    return subprocess.Popen([PY, KEEPALIVE, '--ignore-jobs', '--duty', '0.35'], stdout=log, stderr=subprocess.STDOUT,
                            env=gpu_env(gpu), start_new_session=True)


def stop(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main(spec_path):
    spec = json.loads(pathlib.Path(spec_path).read_text())
    gpu, log_dir = spec['gpu'], pathlib.Path(spec['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = (LOCKDIR/f'gpu{gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = log_dir/'queue_state.json'
    state = dict(started_utc=now(), pid=os.getpid(), gpu=gpu, spec=spec_path, jobs={})
    state_path.write_text(json.dumps(state, indent=2))
    keep = None
    for job in spec['jobs']:
        name = job['name']
        waited = False
        while not all(pathlib.Path(p).exists() for p in job.get('wait_for', [])):
            if keep is None or keep.poll() is not None:
                keep = start_keepalive(gpu, log_dir)
            waited = True
            time.sleep(60)
        stop(keep)
        keep = None
        for _ in range(30):
            if not busy(gpu):
                break
            time.sleep(10)
        if any(pathlib.Path(p).exists() for p in job.get('skip_if', [])):
            state['jobs'][name] = dict(status='skipped', utc=now())
            state_path.write_text(json.dumps(state, indent=2))
            continue
        state['jobs'][name] = dict(status='running', started_utc=now(), waited=waited)
        state_path.write_text(json.dumps(state, indent=2))
        with (log_dir/f'{name}.log').open('a') as log:
            log.write(f'### {now()} {" ".join(job["cmd"])}\n')
            log.flush()
            rc = subprocess.call(job['cmd'], stdout=log, stderr=subprocess.STDOUT, env=gpu_env(gpu, job.get('env')))
        state['jobs'][name].update(status='complete' if rc == 0 else 'failed', exit_code=rc, finished_utc=now())
        state_path.write_text(json.dumps(state, indent=2))
    keep = start_keepalive(gpu, log_dir)
    state.update(finished_utc=now(), keepalive_pid=keep.pid)
    state_path.write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
