"""GREEN-G1 supervisor: CPU gates now, GPU lanes once the pooling replication sweep frees them.

Lane B (GPUs 2,3) starts after pooling f0_s47_mean exits; lane A (GPUs 0,1) after
pooling f0_s47_std exits. The pooling sweep's idle-lane keepalive is stopped
before a lane is taken. Each arm: exclusive locks -> smoke -> train -> audit. No
retries, no overwrite, no promotion. Finished lanes get gpu_keepalive.py.
"""
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import subprocess
import threading
import time

R = pathlib.Path('experiments/green_aux_g1_20261026')
STAGE = R/'staging_source'
SCRIPT = 'green_aux_regression_20261026.py'
PY = '/workspace/venv/bin/python'
LOCKDIR = pathlib.Path('experiments/phase2_backbone_ssl_20261020')
POOL = pathlib.Path('experiments/phase2_pooling_replication_20261026')
KEEPALIVE = POOL/'gpu_keepalive.py'
ARMS = {'control_w0': '0', 'aux_w0p1': '0.1', 'aux_w0p3': '0.3'}
LANES = {'B': ('2,3', POOL/'f0_s47_mean/exit.json', ['control_w0', 'aux_w0p3']),
         'A': ('0,1', POOL/'f0_s47_std/exit.json', ['aux_w0p1'])}
BASE = ['--split', 'experiments/target_fold4_20260912/preflight/fold4.npz', '--fold', '4', '--pooling', 'mean',
        '--seed', '46', '--ssl-epochs', '5', '--ssl-root', 'experiments/phase2_crossfold_20261015/fold4/ssl5',
        '--head-lr', '0.001', '--clip', '1', '--epochs', '20', '--schedule-epochs', '10', '--deadline-utc', 'none']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def save(p, d):
    with pathlib.Path(p).open('x') as f:
        json.dump(d, f, indent=2)


def read(p):
    return json.loads(pathlib.Path(p).read_text())


def args_for(arm):
    return ['--root', str(R/arm)] + BASE + ['--aux-weight', ARMS[arm]]


def call(cmd, log, env):
    log.write(f'\n### {now()} {" ".join(cmd[:5])}\n')
    log.flush()
    return subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)


def busy(gpus):
    return subprocess.check_output(['nvidia-smi', '-i', gpus, '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip()


def cpu_gate(arm):
    r = R/arm
    if (r/'aux_cpu_tests.json').exists():
        return 0
    r.mkdir(exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4')
    with (r/'cpu.log').open('x') as log:
        rc = call([PY, str(STAGE/SCRIPT), 'prepare'] + args_for(arm), log, env)
        if rc == 0:
            rc = call([PY, str(r/'source'/SCRIPT), 'cpu-tests'] + args_for(arm), log, env)
    return rc


def keepalive_pids(gpus):
    found = []
    for d in pathlib.Path('/proc').iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d/'cmdline').read_bytes().split(b'\0')
            if not any(c.endswith(b'gpu_keepalive.py') for c in cmd):
                continue
            env = dict(kv.split(b'=', 1) for kv in (d/'environ').read_bytes().split(b'\0') if b'=' in kv)
            if env.get(b'CUDA_VISIBLE_DEVICES', b'').decode() == gpus:
                found.append(int(d.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return found


def pool_running():
    return subprocess.run(['pgrep', '-f', 'phase2_pooling_replication_20261026/supervisor.py'],
                          capture_output=True).returncode == 0


def take_lane(gpus, exit_file):
    while not exit_file.exists():
        time.sleep(60)
    while True:
        pids = keepalive_pids(gpus)
        if pids:
            for p in pids:
                os.kill(p, 15)
            time.sleep(15)
            if not keepalive_pids(gpus):
                break
        elif not pool_running():
            break
        time.sleep(10)
    for _ in range(60):
        if not busy(gpus):
            return
        time.sleep(10)
    raise RuntimeError(f'GPUs {gpus} still busy')


def run_arm(arm, gpus):
    r = R/arm
    a = args_for(arm)
    if not (r/'aux_cpu_tests.json').exists():
        save(r/'exit.json', dict(exit_code=-1, step='cpu_gate', finished_utc=now()))
        return
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpus, OMP_NUM_THREADS='4')
    step, rc, locks = 'lock', None, []
    with (r/'full.log').open('x') as log:
        try:
            for p in [r/'execution.lock'] + [LOCKDIR/f'gpu{g}.lock' for g in gpus.split(',')]:
                f = p.open('a')
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locks.append(f)
            assert not busy(gpus), f'GPUs {gpus} busy'
            ms = sha(r/'manifest.json')
            save(r/'launch.json', dict(started_utc=now(), pid=os.getpid(), manifest_sha=ms,
                                       aux_manifest_sha=sha(r/'aux_manifest.json'),
                                       gpus=[int(g) for g in gpus.split(',')], deadline=None))
            tr = [PY, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', str(r/'source'/SCRIPT)]
            step = 'smoke'
            rc = call(tr + ['smoke'] + a, log, env)
            if rc == 0:
                s = read(r/'smoke.json')
                assert s['status'] == 'passed' and s['manifest_sha'] == ms
                save(r/'started.json', dict(started_utc=now(), manifest_sha=ms, smoke_sha=sha(r/'smoke.json')))
                step = 'train'
                rc = call(tr + ['train'] + a, log, env)
            if rc == 0:
                step = 'audit'
                rc = call([PY, str(r/'source'/SCRIPT), 'audit'] + a, log, env)
        except Exception as e:
            log.write(f'\n### supervisor exception at {step}: {e!r}\n')
            rc = rc or -1
        finally:
            for f in locks:
                f.close()
    save(r/'exit.json', dict(exit_code=rc, step=step if rc else 'complete', finished_utc=now()))


def keep_busy(name, gpus):
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpus)
    log = (R/f'keepalive_lane{name}.log').open('a')
    p = subprocess.Popen([PY, str(KEEPALIVE), '--ignore-jobs', '--duty', '0.35'], stdout=log,
                         stderr=subprocess.STDOUT, env=env, start_new_session=True)
    save(R/f'keepalive_lane{name}.json', dict(pid=p.pid, gpus=gpus, started_utc=now(),
                                             stop='kill this pid before launching new GPU work'))


def lane(name, gpus, exit_file, arms):
    try:
        take_lane(gpus, exit_file)
        for arm in arms:
            run_arm(arm, gpus)
    finally:
        keep_busy(name, gpus)


def compare():
    def get(p):
        return read(p) if pathlib.Path(p).exists() else None
    control = get(R/'control_w0/aux_audit.json')
    historical = get('experiments/phase2_pooling_20261020/arms/mean/audit.json')
    out = dict(created_utc=now(), control=control,
               control_minus_historical_fold4_seed46_mean=(
                   {k: control['score'][k]-historical['score'][k] for k in ('CD', 'CV', 'HEX', 'mean')}
                   if control and historical else None), arms={})
    for arm in ('aux_w0p1', 'aux_w0p3'):
        a = get(R/arm/'aux_audit.json')
        d = {k: a['score'][k]-control['score'][k] for k in ('CD', 'CV', 'HEX', 'mean')} if a and control else None
        out['arms'][arm] = dict(audit=a, delta_aux_minus_control=d,
                                improves_at_least_0p03=bool(d and d['mean'] <= -0.03),
                                component_regression_over_0p05=[k for k in ('CD', 'CV', 'HEX') if d and d[k] > 0.05],
                                validation_mean_f1=a and a['validation_mean_f1'],
                                frozen_probe_validation_mean_f1=a and a['frozen_probe_validation_mean_f1'])
    out['gate'] = 'see protocol.json; promotion and G2 are never automatic'
    save(R/'comparison.json', out)


if __name__ == '__main__':
    assert not (R/'started.json').exists()
    save(R/'started.json', dict(started_utc=now(), pid=os.getpid(), supervisor_sha=sha(__file__)))
    gates = {arm: cpu_gate(arm) for arm in ARMS}
    save(R/'cpu_gates.json', dict(finished_utc=now(), returncodes=gates))
    ts = [threading.Thread(target=lane, args=(k, g, e, a)) for k, (g, e, a) in LANES.items()]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    compare()
    save(R/'exit.json', dict(finished_utc=now(), arms={k: read(R/k/'exit.json') for k in ARMS if (R/k/'exit.json').exists()}))
