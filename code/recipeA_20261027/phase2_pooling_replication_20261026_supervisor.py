"""Pooling replication sweep: std vs mean, second seed and second fold.

Two lanes of 2 H200 each. Every arm reuses the byte-identical frozen sources of
phase2_backbone_fold0_regression_ssl5_20261022 (the fold0 mean seed46 control)
and follows the same gates: prepare -> cpu-tests -> exclusive GPU lock ->
smoke -> train -> audit. No retries, no overwrite, no promotion.
"""
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import subprocess
import threading

R = pathlib.Path('experiments/phase2_pooling_replication_20261026')
SRC = pathlib.Path('experiments/phase2_backbone_fold0_regression_ssl5_20261022/source')
SCRIPT = 'phase2_backbone_ssl_regression_20261020.py'
PY = '/workspace/venv/bin/python'
LOCKDIR = pathlib.Path('experiments/phase2_backbone_ssl_20261020')
FOLDS = {
    0: ('experiments/ssl_screen_20260911/preflight/fold0.npz', 'experiments/phase2_crossfold_20261015/fold0/ssl5'),
    4: ('experiments/target_fold4_20260912/preflight/fold4.npz', 'experiments/phase2_crossfold_20261015/fold4/ssl5'),
}
ARMS = {
    'f4_s47_mean': (4, 47, 'mean'), 'f4_s47_std': (4, 47, 'std'),
    'f0_s46_std': (0, 46, 'std'),
    'f0_s47_mean': (0, 47, 'mean'), 'f0_s47_std': (0, 47, 'std'),
}
LANES = {'A': ('0,1', ['f4_s47_mean', 'f0_s46_std', 'f0_s47_std']),
         'B': ('2,3', ['f4_s47_std', 'f0_s47_mean'])}
done = {lane: threading.Event() for lane in LANES}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def save(p, d):
    with pathlib.Path(p).open('x') as f:
        json.dump(d, f, indent=2)


def read(p):
    return json.loads(pathlib.Path(p).read_text())


def args_for(name):
    fold, seed, pool = ARMS[name]
    split, ssl = FOLDS[fold]
    return ['--root', str(R/name), '--split', split, '--fold', str(fold), '--pooling', pool,
            '--seed', str(seed), '--ssl-epochs', '5', '--ssl-root', ssl, '--head-lr', '0.001',
            '--clip', '1', '--epochs', '20', '--schedule-epochs', '10', '--deadline-utc', 'none']


def run_arm(name, gpus):
    r = R/name
    a = args_for(name)
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpus, OMP_NUM_THREADS='4')
    step, rc = 'prepare', None
    r.mkdir(parents=True, exist_ok=False)
    locks = []
    with (r/'full.log').open('x') as log:
        def call(cmd):
            log.write(f'\n### {now()} {" ".join(cmd[:4])}\n'); log.flush()
            return subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            rc = call([PY, str(SRC/SCRIPT), 'prepare'] + a)
            if rc == 0:
                step = 'cpu_tests'
                rc = call([PY, str(r/'source'/SCRIPT), 'cpu-tests'] + a)
            if rc == 0:
                assert read(r/'cpu_tests.json')['status'] == 'passed'
                step = 'lock'
                for p in [r/'execution.lock'] + [LOCKDIR/f'gpu{g}.lock' for g in gpus.split(',')]:
                    f = p.open('a'); fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(f)
                busy = subprocess.check_output(['nvidia-smi', '-i', gpus, '--query-compute-apps=pid',
                                                '--format=csv,noheader'], text=True).strip()
                assert not busy, f'GPUs {gpus} busy: {busy}'
                ms = sha(r/'manifest.json')
                save(r/'launch.json', dict(started_utc=now(), pid=os.getpid(), manifest_sha=ms,
                                           gpus=[int(g) for g in gpus.split(',')], deadline=None))
                tr = [PY, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', str(r/'source'/SCRIPT)]
                step = 'smoke'
                rc = call(tr + ['smoke'] + a)
                if rc == 0:
                    s = read(r/'smoke.json'); assert s['status'] == 'passed' and s['manifest_sha'] == ms
                    save(r/'started.json', dict(started_utc=now(), manifest_sha=ms, smoke_sha=sha(r/'smoke.json')))
                    step = 'train'
                    rc = call(tr + ['train'] + a)
                if rc == 0:
                    step = 'audit'
                    rc = call([PY, str(r/'source'/SCRIPT), 'audit'] + a)
        except Exception as e:
            log.write(f'\n### supervisor exception at {step}: {e!r}\n'); rc = rc or -1
        finally:
            for f in locks:
                f.close()
    save(r/'exit.json', dict(exit_code=rc, step=step if rc else 'complete', finished_utc=now()))


def lane(name, gpus, arms):
    for arm in arms:
        run_arm(arm, gpus)
    done[name].set()
    others = [e for k, e in done.items() if k != name]
    if all(e.is_set() for e in others):
        return
    # Fill this lane's GPUs until the other lane finishes (reclaim below 10% util).
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpus)
    with (R/f'keepalive_lane{name}.log').open('a') as log:
        p = subprocess.Popen([PY, str(R/'gpu_keepalive.py'), '--ignore-jobs', '--duty', '0.35'],
                             stdout=log, stderr=subprocess.STDOUT, env=env)
        for e in others:
            e.wait()
        p.terminate(); p.wait()


def compare():
    def score(p):
        return read(p)['score'] if pathlib.Path(p).exists() else None
    E = 'experiments/'
    pairs = {
        'fold4_seed46_historical': (E+'phase2_pooling_20261020/arms/mean/audit.json',
                                    E+'phase2_pooling_20261020/arms/std/audit.json'),
        'fold0_seed46': (E+'phase2_backbone_fold0_regression_ssl5_20261022/audit.json', str(R/'f0_s46_std/audit.json')),
        'fold4_seed47': (str(R/'f4_s47_mean/audit.json'), str(R/'f4_s47_std/audit.json')),
        'fold0_seed47': (str(R/'f0_s47_mean/audit.json'), str(R/'f0_s47_std/audit.json')),
    }
    out = {}
    for k, (m, s) in pairs.items():
        sm, ss = score(m), score(s)
        d = {t: ss[t]-sm[t] for t in ('CD', 'CV', 'HEX', 'mean')} if sm and ss else None
        out[k] = dict(mean_pool=sm, std_pool=ss, delta_std_minus_mean=d,
                      component_regression_over_0p05=[t for t in ('CD', 'CV', 'HEX') if d and d[t] > 0.05])
    ds = [v['delta_std_minus_mean']['mean'] for v in out.values() if v['delta_std_minus_mean']]
    save(R/'comparison.json', dict(created_utc=now(), pairs=out, complete_pairs=len(ds),
                                   average_delta_mean=sum(ds)/len(ds) if ds else None,
                                   pairs_improving_at_least_0p03=sum(d <= -0.03 for d in ds),
                                   gate='see protocol.json; promotion is never automatic'))


if __name__ == '__main__':
    assert not (R/'started.json').exists()
    save(R/'started.json', dict(started_utc=now(), pid=os.getpid(), supervisor_sha=sha(__file__)))
    ts = [threading.Thread(target=lane, args=(k, g, a)) for k, (g, a) in LANES.items()]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    compare()
    save(R/'exit.json', dict(finished_utc=now(), arms={k: read(R/k/'exit.json') for k in ARMS if (R/k/'exit.json').exists()}))
