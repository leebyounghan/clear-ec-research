"""Calibration of the frozen screen against trained arms, lane GPU 2 (2026-11-06). Descriptive; gate in protocol.json amendments.
Frozen own read of mean alone and of two combinations, then last-five-epoch deltas of the already trained pooling arms.
Usage: calib_frozen_e2e_20261106.py"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
L = Path('/workspace/clear-ec-research/experiments/lane_gpu2_20261106')
spec = importlib.util.spec_from_file_location('FR', L / 'source/frozen_read_20261106.py'); FR = importlib.util.module_from_spec(spec); spec.loader.exec_module(FR)
sys.path.insert(0, str(FR.X / 'meta_ceiling_20261103/source'))
E = FR.E; REFDIR = FR.LANE / 'reads/own'; FR.READS = FR.LANE / 'reads/own/calib'; FR.READS.mkdir(parents=True, exist_ok=True); LOG = FR.READS / 'calib.jsonl'; folds = [0, 4]; seeds = [0, 1, 2]
R0 = {}
for f in folds:
    z = np.load(REFDIR / f'std_f{f}.npz'); R0[f] = dict(rows=z['rows'], e=z['e'], lg=z['lg'], per_seed=z['per_seed'].tolist())


def per_fold(name, feats):
    out = {}
    for f in folds:
        FR.OUT = FR.LANE / f'frozen_own_f{f}'; out.update(FR.read_arm(name, feats, [f], seeds))
    return out


def write(o):
    LOG.open('a').write(json.dumps(o) + '\n'); print(json.dumps(o), flush=True)


pos = {int(i): n for n, i in enumerate(FR.bb['idx'])}


def last5(root, f):
    run = FR.X / root / 'runs' / f'fold{f}' / 'cpdino_vitb'; fs = sorted(run.glob('predictions_epoch_*.npz')); last = int(fs[-1].stem[-2:]); use = [p for p in fs if int(p.stem[-2:]) > last - 5]; out = []
    for p in use:
        with np.load(p, allow_pickle=True) as z: out.append((z['indices'].astype(int), z['pred'].astype(np.float64), z['target'].astype(np.float64)))
    ix, _, y = out[0]; return ix, y, np.mean([E.ape(p, y) for _, p, _ in out], 0), [int(p.stem[-2:]) for p in use]


def e2e(name, arm, ref, f):
    ix, y, e1, ep1 = last5(arm, f); ix0, y0, e0, ep0 = last5(ref, f); assert np.array_equal(ix, ix0) and np.array_equal(y, y0); g = FR.gp[np.array([pos[int(i)] for i in ix])]; pb = E.paired_bootstrap(e1, e0, g); s1, s0 = E.official(e1), E.official(e0)
    write(dict(kind='e2e', arm=name, fold=f, delta=round(pb['delta'], 4), ci95=[round(v, 4) for v in pb['ci95']], per={k: round(s1[k] - s0[k], 4) for k in FR.K}, arm_score=round(s1['mean'], 4), ref_score=round(s0['mean'], 4), epochs=ep1))


A = 'phase2_pooling_20261020/arms'; RP = 'phase2_pooling_replication_20261026'
for arm in ('mean', 'mean_std', 'max', 'min', 'mean_std_min_max'):
    e2e(arm + '_s46', f'{A}/{arm}', f'{A}/std', 4)
e2e('mean_s47', f'{RP}/f0_s47_mean', f'{RP}/f0_s47_std', 0)
e2e('mean_s47', f'{RP}/f4_s47_mean', f'{RP}/f4_s47_std', 4)
for name, feats in (('mean', ['mean']), ('ac_cov', ['std', 'autocorr', 'cov_sqrt']), ('ac_spec', ['std', 'autocorr', 'spectrum'])):
    a = per_fold(name, feats); o = FR.compare(name, a, R0, folds); o['kind'] = 'frozen_own'; LOG.open('a').write(json.dumps(o) + '\n')
print('calib done', flush=True)
