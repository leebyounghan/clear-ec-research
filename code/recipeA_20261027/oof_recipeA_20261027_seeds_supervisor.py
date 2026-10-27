"""Recipe A seed sensitivity: seeds 47 and 48 on all five folds after the seed46 OOF lane.

Waits for the OOF supervisor to finish, stops its lane-0,1 keepalive, then runs the
frozen recipe A regression for (fold, seed) pairs not yet trained. Folds 0/4 seed47
reuse the pooling replication runs; fold2 uses the official zero-target wrapper.
Writes seeds_result.json: per-fold/per-seed scores, seed spread, per-seed OOF and the
equal-weight seed-average OOF (decoded predictions, no TTA or calibration).
"""
import importlib.util
import os
import pathlib
import time

import numpy as np

spec = importlib.util.spec_from_file_location('oof', 'experiments/oof_recipeA_20261027/supervisor.py')
oof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oof)
R = oof.R
QUEUE = [(1, 47), (3, 47), (2, 47), (0, 48), (4, 48), (1, 48), (3, 48), (2, 48)]
SSL_ROOT = {0: 'experiments/phase2_crossfold_20261015/fold0/ssl5', 4: 'experiments/phase2_crossfold_20261015/fold4/ssl5'}
DONE = {(0, 46): 'experiments/phase2_pooling_replication_20261026/f0_s46_std',
        (4, 46): 'experiments/phase2_pooling_20261020/arms/std',
        (0, 47): 'experiments/phase2_pooling_replication_20261026/f0_s47_std',
        (4, 47): 'experiments/phase2_pooling_replication_20261026/f4_s47_std'}


def root_of(fold, seed):
    if (fold, seed) in DONE:
        return pathlib.Path(DONE[(fold, seed)])
    return R/f'fold{fold}'/f'regression_std_s{seed}'


def reg_job(fold, seed, gpus='0,1'):
    root = root_of(fold, seed)
    runner = oof.REG_WRAP if fold == 2 else oof.REG_RUNNER
    frozen = root/'source'/runner.name
    a = ['--root', str(root), '--split', oof.SPLITS[fold], '--fold', str(fold), '--pooling', 'std', '--seed', str(seed),
         '--ssl-epochs', '5', '--ssl-root', SSL_ROOT.get(fold, str(R/f'fold{fold}'/'ssl5')), '--head-lr', '0.001',
         '--clip', '1', '--epochs', '20', '--schedule-epochs', '10', '--deadline-utc', 'none']
    step, rc, held = 'prepare', None, []
    with (R/f'fold{fold}_s{seed}_regression.log').open('x') as log:
        try:
            rc = oof.call([oof.PY, str(runner), 'prepare'] + a, log, oof.cpu_env())
            if rc == 0:
                step = 'cpu_tests'
                rc = oof.call([oof.PY, str(frozen), 'cpu-tests'] + a, log, oof.cpu_env())
            if rc == 0:
                assert oof.read(root/'cpu_tests.json')['status'] == 'passed'
                if fold == 2:
                    assert oof.read(root/'zero_policy_tests.json')['status'] == 'passed'
                step = 'lock'
                held = oof.lock([root/'execution.lock'] + [oof.LOCKDIR/f'gpu{g}.lock' for g in gpus.split(',')])
                assert not oof.busy(gpus), f'GPUs {gpus} busy'
                ms = oof.sha(root/'manifest.json')
                oof.save(root/'launch.json', dict(started_utc=oof.now(), pid=os.getpid(), manifest_sha=ms,
                                                  gpus=[int(g) for g in gpus.split(',')], deadline=None))
                tr = [oof.PY, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', str(frozen)]
                step = 'smoke'
                rc = oof.call(tr + ['smoke'] + a, log, oof.gpu_env(gpus))
                if rc == 0:
                    s = oof.read(root/'smoke.json')
                    assert s['status'] == 'passed' and s['manifest_sha'] == ms
                    oof.save(root/'started.json', dict(started_utc=oof.now(), manifest_sha=ms, smoke_sha=oof.sha(root/'smoke.json')))
                    step = 'train'
                    rc = oof.call(tr + ['train'] + a, log, oof.gpu_env(gpus))
                if rc == 0:
                    step = 'audit'
                    rc = oof.call([oof.PY, str(frozen), 'audit'] + a, log, oof.cpu_env())
        except Exception as e:
            log.write(f'\n### supervisor exception at {step}: {e!r}\n')
            rc = rc or -1
        finally:
            for f in held:
                f.close()
    oof.save(R/f'fold{fold}_s{seed}_regression_exit.json', dict(exit_code=rc, step=step if rc else 'complete', finished_utc=oof.now()))


def take_lane():
    while not (R/'exit.json').exists():
        time.sleep(60)
    marker = R/'keepalive_lane01.json'
    if marker.exists():
        try:
            os.kill(oof.read(marker)['pid'], 15)
        except ProcessLookupError:
            pass
    for _ in range(60):
        if not oof.busy('0,1'):
            return
        time.sleep(10)
    raise RuntimeError('lane 0,1 still busy')


def summarize():
    with np.load('datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float64)
    seeds = (46, 47, 48)
    pred = {s: np.full((len(ids), 3), np.nan) for s in seeds}
    table, missing = {}, []
    for fold in range(5):
        for seed in seeds:
            root = root_of(fold, seed)
            p = root/'runs'/f'fold{fold}'/'cpdino_vitb'/'predictions.npz'
            if not (root/'audit.json').exists() or not p.exists():
                missing.append([fold, seed])
                continue
            with np.load(p) as z:
                ix, pr = z['indices'], z['pred'].astype(np.float64)
                assert (ids[ix] == z['ids'].astype(str)).all()
            pred[seed][ix] = pr
            table[f'fold{fold}_seed{seed}'] = oof.official(pr, y[ix])
    out = dict(created_utc=oof.now(), per_fold_seed=table, missing=missing, per_seed_oof={}, spread={})
    for seed in seeds:
        ok = np.isfinite(pred[seed]).all(1)
        if ok.all():
            out['per_seed_oof'][seed] = oof.official(pred[seed], y)
    for fold in range(5):
        vals = [table[f'fold{fold}_seed{s}']['mean'] for s in seeds if f'fold{fold}_seed{s}' in table]
        if len(vals) > 1:
            out['spread'][f'fold{fold}'] = dict(values=vals, range=max(vals)-min(vals), std=float(np.std(vals, ddof=1)))
    complete = all(np.isfinite(pred[s]).all() for s in seeds)
    if complete:
        avg = np.mean([pred[s] for s in seeds], axis=0)
        out['seed_average_oof'] = oof.official(avg, y)
        np.savez_compressed(R/'seeds_oof_predictions.npz', ids=ids, target=y, **{f'seed{s}': pred[s] for s in seeds}, seed_average=avg)
    out['note'] = 'Regression seed varies; SSL seed fixed at 1003. Decoded-prediction average; no TTA or calibration.'
    oof.save(R/'seeds_result.json', out)


if __name__ == '__main__':
    assert not (R/'seeds_started.json').exists()
    oof.save(R/'seeds_started.json', dict(started_utc=oof.now(), pid=os.getpid(), supervisor_sha=oof.sha(__file__), queue=QUEUE))
    try:
        take_lane()
        for fold, seed in QUEUE:
            reg_job(fold, seed)
    finally:
        oof.keep_busy('lane01_after_seeds', '0,1')
    summarize()
    oof.save(R/'seeds_exit.json', dict(finished_utc=oof.now(), runs={f'{f}_{s}': oof.read(R/f'fold{f}_s{s}_regression_exit.json')
                                                                     for f, s in QUEUE if (R/f'fold{f}_s{s}_regression_exit.json').exists()}))
