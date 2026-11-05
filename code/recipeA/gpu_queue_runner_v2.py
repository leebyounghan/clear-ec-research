"""Sequential GPU job queue with exclusive GPU locks, spec hot-reload and keepalive while waiting.

jobs.json: {"gpu": "3" or "2,3", "log_dir": ..., "jobs": [{"name", "wait_for": [paths], "skip_if": [paths],
"cmd": [...], "env": {...}}]}. The spec is re-read at every job boundary, so jobs appended to the
file are picked up without a restart; jobs already recorded complete/skipped in queue_state.json
are not re-run. Before each job: keepalive while wait_for paths are missing, then wait (no
keepalive) while any listed GPU still has a compute process, then skip if a skip_if path exists.
Holds experiments/phase2_backbone_ssl_20261020/gpu<N>.lock for every listed GPU. No retries.
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


def load_spec(path):
    for _ in range(10):
        try:
            return json.loads(pathlib.Path(path).read_text())
        except json.JSONDecodeError:
            time.sleep(3)
    raise


def next_job(spec_path, done):
    for job in load_spec(spec_path)['jobs']:
        if job['name'] not in done:
            return job
    return None


def main(spec_path):
    spec = load_spec(spec_path)
    gpu, log_dir = spec['gpu'], pathlib.Path(spec['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    locks = []
    for g in gpu.split(','):
        f = (LOCKDIR/f'gpu{g}.lock').open('a')
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(f)
    state_path = log_dir/'queue_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(jobs={})
    state.update(started_utc=now(), pid=os.getpid(), gpu=gpu, spec=spec_path, runner_version=2)
    state['jobs'] = {k: v for k, v in state['jobs'].items() if v.get('status') in ('complete', 'skipped')}

    def flush():
        state_path.write_text(json.dumps(state, indent=2))
    flush()
    keep = None
    idle_since = None
    while True:
        job = next_job(spec_path, set(state['jobs']))
        if job is None:
            if keep is None or keep.poll() is not None:
                keep = start_keepalive(gpu, log_dir)
            if idle_since is None:
                idle_since = now()
                state.update(idle_since_utc=idle_since, keepalive_pid=keep.pid)
                flush()
            time.sleep(60)
            continue
        idle_since = None
        name = job['name']
        waited = False
        while not all(pathlib.Path(p).exists() for p in job.get('wait_for', [])):
            if keep is None or keep.poll() is not None:
                keep = start_keepalive(gpu, log_dir)
            waited = True
            time.sleep(60)
        stop(keep)
        keep = None
        while busy(gpu):
            time.sleep(15)
        if any(pathlib.Path(p).exists() for p in job.get('skip_if', [])):
            state['jobs'][name] = dict(status='skipped', utc=now())
            flush()
            continue
        state['jobs'][name] = dict(status='running', started_utc=now(), waited=waited)
        flush()
        with (log_dir/f'{name}.log').open('a') as log:
            log.write(f'### {now()} {" ".join(job["cmd"])}\n')
            log.flush()
            rc = subprocess.call(job['cmd'], stdout=log, stderr=subprocess.STDOUT, env=gpu_env(gpu, job.get('env')))
        state['jobs'][name].update(status='complete' if rc == 0 else 'failed', exit_code=rc, finished_utc=now())
        flush()


if __name__ == '__main__':
    main(sys.argv[1])
