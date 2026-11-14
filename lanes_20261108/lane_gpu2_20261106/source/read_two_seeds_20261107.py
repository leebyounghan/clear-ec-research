"""Two seeds of one family averaged, added to the best pipeline (lane GPU 2, 2026-11-07). Same added formula as read_arm and read_family: H x exp(w (log new - log recipe A TTA)), stretch from best_pipeline.
new = log mean over seeds of the log mean of epochs 16-20. Folds 0 and 4. Also prints each seed alone in the same form, so a result can be checked against seed 46's stored read.
Usage: read_two_seeds_20261107.py <name> <seed A root pattern with {f}> <seed B root pattern, or - for seed A alone> <family dir>"""
import json, sys
from pathlib import Path
import numpy as np
X = Path('/workspace/clear-ec-research/experiments')
sys.path.insert(0, str(X / 'eval_harness_20261031/source')); sys.path.insert(0, str(X / 'meta_ceiling_20261103/source'))
import eval_oof_20261031 as E
import best_pipeline_20261106 as BP
K = ['CD', 'CV', 'HEX']
b = E.load_base(); gp = E.donors(b['ids']); pos = {int(i): n for n, i in enumerate(b['idx'])}; assert np.array_equal(BP.idx, b['idx'])


def last5(root, f, fam):
    run = X / root / 'runs' / f'fold{f}' / fam; out = []
    for e in range(16, 21):
        with np.load(run / f'predictions_epoch_{e:02d}.npz', allow_pickle=True) as z: out.append((z['indices'].astype(int), z['pred'].astype(np.float64)))
    ix = out[0][0]; assert all(np.array_equal(ix, o[0]) for o in out)
    return np.array([pos[int(i)] for i in ix]), np.mean([np.log(p) for _, p in out], 0)


def added(rows, lg):
    A = BP._b['A']; o = {}
    for w in (0.125, 0.25, 0.5):
        c = np.zeros_like(BP.H); c[rows] = w * (lg - np.log(A[rows])); P = BP.pipe(BP.add(BP.BEST, {ff: c for ff in range(5)})); eb, er = E.ape(P[rows], BP.y[rows]), E.ape(BP.REF_BEST[rows], BP.y[rows]); pb = E.paired_bootstrap(eb, er, gp[rows]); s1, s0 = E.official(eb), E.official(er)
        o[str(w)] = dict(delta=round(pb['delta'], 4), ci95=[round(v, 4) for v in pb['ci95']], per={k: round(s1[k] - s0[k], 4) for k in K}, best_on_rows=round(s0['mean'], 4))
    return o


name, pa, pb_, fam = sys.argv[1:5]; folds = (0, 4); res = dict(name=name, folds=list(folds), seeds={})
Ra, La, Lb = [], [], []
for f in folds:
    r, a = last5(pa.format(f=f), f, fam); Ra.append(r); La.append(a)
    if pb_ != '-':
        r2, q = last5(pb_.format(f=f), f, fam); assert np.array_equal(r, r2); Lb.append(q)
rows = np.concatenate(Ra); la = np.concatenate(La); res['seeds']['a'] = added(rows, la)
if pb_ != '-':
    lb = np.concatenate(Lb); res['seeds']['b'] = added(rows, lb); res['two_seeds'] = added(rows, (la + lb) / 2)
    res['r_between_seeds'] = {k: round(float(np.corrcoef(la[:, j], lb[:, j])[0, 1]), 4) for j, k in enumerate(K)}
print(json.dumps(res, indent=1))
if pb_ != '-':
    (X / 'lane_gpu2_20261106/reads').mkdir(exist_ok=True); (X / f'lane_gpu2_20261106/reads/{name}.json').write_text(json.dumps(res, indent=1))
