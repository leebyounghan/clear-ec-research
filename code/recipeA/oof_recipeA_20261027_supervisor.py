"""Recipe A 5-fold OOF: fresh fold-train-only SSL5 for folds 1-3, then std seed46 regression.

SSL fold1 on GPU0, fold3 on GPU1, fold2 on GPU2 (single-GPU runner), concurrently.
Regression lane GPUs 0,1 starts once both of its SSL jobs end: fold1, fold3, fold2
(fold2 through the official zero-target wrapper). GPU2 gets gpu_keepalive.py after
fold2 SSL until reassigned; GPU3 is reserved for green work. Folds 0 and 4 reuse
the completed std seed46 runs. No retries, no overwrite, no promotion.
"""
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import subprocess
import threading

import numpy as np

R = pathlib.Path('experiments/oof_recipeA_20261027')
PY = '/workspace/venv/bin/python'
LOCKDIR = pathlib.Path('experiments/phase2_backbone_ssl_20261020')
KEEPALIVE = pathlib.Path('experiments/phase2_pooling_replication_20261026/gpu_keepalive.py')
SSL_RUNNER = pathlib.Path('experiments/phase2_crossfold_20261015/fold0/ssl5/source/phase2_crossfold_ssl_20261015.py')
SSL_WRAP = R/'staging_ssl_zero/phase2_crossfold_ssl_zero_policy_20261027.py'
REG_RUNNER = pathlib.Path('experiments/phase2_backbone_fold0_regression_ssl5_20261022/source/phase2_backbone_ssl_regression_20261020.py')
REG_WRAP = R/'staging_regression_zero/recipeA_regression_zero_policy_20261027.py'
SPLITS = {0: 'experiments/ssl_screen_20260911/preflight/fold0.npz', 1: 'experiments/ssl_screen_20260911/preflight/fold1.npz',
          2: 'experiments/overnight_target_20260912/preflight/fold2.npz', 3: 'experiments/overnight_target_20260912/preflight/fold3.npz',
          4: 'experiments/target_fold4_20260912/preflight/fold4.npz'}
PREFLIGHT = {1: 'experiments/phase2_crossfold_20261015/data_preflight_legacy.json',
             2: 'experiments/phase2_crossfold_20261015/data_preflight.json',
             3: 'experiments/phase2_crossfold_20261015/data_preflight.json'}
SSL_GPU = {1: '0', 3: '1', 2: '2'}
DEADLINE = '2026-11-10T00:00:00Z'
EXISTING = {0: 'experiments/phase2_pooling_replication_20261026/f0_s46_std', 4: 'experiments/phase2_pooling_20261020/arms/std'}
ssl_ok = {}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def save(p, d):
    with pathlib.Path(p).open('x') as f:
        json.dump(d, f, indent=2)


def read(p):
    return json.loads(pathlib.Path(p).read_text())


def call(cmd, log, env):
    log.write(f'\n### {now()} {" ".join(cmd[:5])}\n')
    log.flush()
    return subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)


def busy(gpus):
    return subprocess.check_output(['nvidia-smi', '-i', gpus, '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip()


def lock(paths):
    held = []
    for p in paths:
        f = p.open('a')
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        held.append(f)
    return held


def cpu_env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4')


def gpu_env(gpus):
    return dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpus, OMP_NUM_THREADS='4')


def ssl_job(fold):
    root = R/f'fold{fold}'/'ssl5'
    gpu = SSL_GPU[fold]
    runner = SSL_WRAP if fold == 2 else SSL_RUNNER
    frozen = root/'source'/runner.name
    common = ['--root', str(root), '--split', SPLITS[fold], '--fold', str(fold), '--data-preflight', PREFLIGHT[fold],
              '--seed', '1003', '--epochs', '5', '--deadline-utc', DEADLINE]
    step, rc, held = 'prepare', None, []
    with (R/f'fold{fold}_ssl.log').open('x') as log:
        try:
            rc = call([PY, '-u', str(runner), 'prepare'] + common, log, cpu_env())
            if rc == 0:
                step = 'tests'
                rc = call([PY, '-u', str(frozen), 'tests'] + common, log, cpu_env())
            if rc == 0:
                step = 'lock'
                held = lock([LOCKDIR/f'gpu{gpu}.lock'])
                assert not busy(gpu), f'GPU {gpu} busy'
                save(R/f'fold{fold}_ssl_launch.json', dict(started_utc=now(), gpu=int(gpu), manifest_sha=sha(root/'manifest.json')))
                step = 'smoke'
                rc = call([PY, '-u', str(frozen), 'smoke'] + common, log, gpu_env(gpu))
                if rc == 0:
                    step = 'train'
                    rc = call([PY, '-u', str(frozen), 'train'] + common, log, gpu_env(gpu))
                if rc == 0 and not (root/'run'/'complete.json').exists():
                    step, rc = 'incomplete', -2
        except Exception as e:
            log.write(f'\n### supervisor exception at {step}: {e!r}\n')
            rc = rc or -1
        finally:
            for f in held:
                f.close()
    save(R/f'fold{fold}_ssl_exit.json', dict(exit_code=rc, step=step if rc else 'complete', finished_utc=now()))
    ssl_ok[fold] = rc == 0


def reg_job(fold, gpus='0,1'):
    root = R/f'fold{fold}'/'regression_std_s46'
    runner = REG_WRAP if fold == 2 else REG_RUNNER
    frozen = root/'source'/runner.name
    a = ['--root', str(root), '--split', SPLITS[fold], '--fold', str(fold), '--pooling', 'std', '--seed', '46',
         '--ssl-epochs', '5', '--ssl-root', str(R/f'fold{fold}'/'ssl5'), '--head-lr', '0.001', '--clip', '1',
         '--epochs', '20', '--schedule-epochs', '10', '--deadline-utc', 'none']
    step, rc, held = 'prepare', None, []
    with (R/f'fold{fold}_regression.log').open('x') as log:
        try:
            rc = call([PY, str(runner), 'prepare'] + a, log, cpu_env())
            if rc == 0:
                step = 'cpu_tests'
                rc = call([PY, str(frozen), 'cpu-tests'] + a, log, cpu_env())
            if rc == 0:
                assert read(root/'cpu_tests.json')['status'] == 'passed'
                if fold == 2:
                    assert read(root/'zero_policy_tests.json')['status'] == 'passed'
                step = 'lock'
                held = lock([root/'execution.lock'] + [LOCKDIR/f'gpu{g}.lock' for g in gpus.split(',')])
                assert not busy(gpus), f'GPUs {gpus} busy'
                ms = sha(root/'manifest.json')
                save(root/'launch.json', dict(started_utc=now(), pid=os.getpid(), manifest_sha=ms,
                                              gpus=[int(g) for g in gpus.split(',')], deadline=None))
                tr = [PY, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', str(frozen)]
                step = 'smoke'
                rc = call(tr + ['smoke'] + a, log, gpu_env(gpus))
                if rc == 0:
                    s = read(root/'smoke.json')
                    assert s['status'] == 'passed' and s['manifest_sha'] == ms
                    save(root/'started.json', dict(started_utc=now(), manifest_sha=ms, smoke_sha=sha(root/'smoke.json')))
                    step = 'train'
                    rc = call(tr + ['train'] + a, log, gpu_env(gpus))
                if rc == 0:
                    step = 'audit'
                    rc = call([PY, str(frozen), 'audit'] + a, log, cpu_env())
        except Exception as e:
            log.write(f'\n### supervisor exception at {step}: {e!r}\n')
            rc = rc or -1
        finally:
            for f in held:
                f.close()
    save(R/f'fold{fold}_regression_exit.json', dict(exit_code=rc, step=step if rc else 'complete', finished_utc=now()))


def keep_busy(name, gpus):
    log = (R/f'keepalive_{name}.log').open('a')
    p = subprocess.Popen([PY, str(KEEPALIVE), '--ignore-jobs', '--duty', '0.35'], stdout=log, stderr=subprocess.STDOUT,
                         env=dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpus),
                         start_new_session=True)
    save(R/f'keepalive_{name}.json', dict(pid=p.pid, gpus=gpus, started_utc=now(),
                                         stop='kill this pid before launching new GPU work'))


def official(pred, y):
    scored = y > 0
    rel = np.where(scored, np.abs(pred-y)/np.where(scored, y, 1.), np.nan)
    v = np.nanmean(rel, 0)*100
    return dict(CD=float(v[0]), CV=float(v[1]), HEX=float(v[2]), mean=float(v.mean()), n=int(len(y)),
                zero_targets_unscored=dict(zip(('CD', 'CV', 'HEX'), (~scored).sum(0).tolist())))


def aggregate():
    with np.load('datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float64)
    pred = np.full((len(ids), 3), np.nan)
    fold_of = np.full(len(ids), -1)
    sources, missing = {}, []
    for f in range(5):
        root = pathlib.Path(EXISTING.get(f, R/f'fold{f}'/'regression_std_s46'))
        p = root/'runs'/f'fold{f}'/'cpdino_vitb'/'predictions.npz'
        if not (root/'audit.json').exists() or not p.exists():
            missing.append(f)
            continue
        with np.load(p) as z:
            ix, pr = z['indices'], z['pred'].astype(np.float64)
            assert (ids[ix] == z['ids'].astype(str)).all()
        assert (fold_of[ix] == -1).all(), 'Overlapping OOF rows'
        pred[ix], fold_of[ix] = pr, f
        sources[f] = dict(root=str(root), predictions_sha=sha(p), audit_score=read(root/'audit.json')['score'],
                          official_rule_score=official(pr, y[ix]))
    covered = fold_of >= 0
    result = dict(created_utc=now(), recipe='A (std pooling, SSL5 seed1003, regression seed46 terminal20)',
                  folds=sources, missing_folds=missing, covered_images=int(covered.sum()), total_images=len(ids),
                  oof=official(pred[covered], y[covered]) if covered.any() else None,
                  metric='per-target MAPE, zero ground truth unscored (public CLEAR-EC evaluate.py), mean of three targets; '
                         'per-case errors not rounded',
                  limits=['Development folds historically reused; not an official score',
                          'Patient independence and near-duplicate leakage unresolved',
                          'Folds 0/4 reuse completed runs; folds 1-3 trained here with the identical recipe'])
    np.savez_compressed(R/'oof_predictions.npz', ids=ids, target=y, pred=pred, fold=fold_of)
    result['predictions_sha'] = sha(R/'oof_predictions.npz')
    save(R/'oof_result.json', result)


def lane(threads):
    threads[1].join()
    threads[3].join()
    for fold in (1, 3, 2):
        if fold == 2:
            threads[2].join()
        if ssl_ok.get(fold):
            reg_job(fold)
        else:
            save(R/f'fold{fold}_regression_exit.json', dict(exit_code=-1, step='ssl_failed', finished_utc=now()))
    keep_busy('lane01', '0,1')


if __name__ == '__main__':
    assert not (R/'started.json').exists()
    save(R/'started.json', dict(started_utc=now(), pid=os.getpid(), supervisor_sha=sha(__file__)))
    threads = {f: threading.Thread(target=ssl_job, args=(f,)) for f in (1, 3, 2)}
    for t in threads.values():
        t.start()
    gpu2 = threading.Thread(target=lambda: (threads[2].join(), keep_busy('gpu2', '2')))
    gpu2.start()
    main_lane = threading.Thread(target=lane, args=(threads,))
    main_lane.start()
    main_lane.join()
    gpu2.join()
    aggregate()
    save(R/'exit.json', dict(finished_utc=now(), ssl={f: read(R/f'fold{f}_ssl_exit.json') for f in (1, 2, 3)},
                             regression={f: read(R/f'fold{f}_regression_exit.json') for f in (1, 2, 3)}))
