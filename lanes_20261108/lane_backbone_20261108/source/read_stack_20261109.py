"""Stacked added read (2026-11-09): best pipeline + w * (log member - log recipe A TTA) for one or more members at once, folds 0 and 4,
members = log mean of epochs 8-12 (raw track). Same formula as read_bb_20261108.py --added. Usage: read_stack_20261109.py w name=pattern ..."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source'); import eval_oof_20261031 as E
X = Path('/workspace/clear-ec-research/experiments'); sys.path.insert(0, str(X / 'meta_ceiling_20261103/source')); import best_pipeline_20261106 as BP
K = ['CD', 'CV', 'HEX']; pos = {int(i): n for n, i in enumerate(BP.idx)}
def member(pat):
    L = np.full_like(BP.H, np.nan)
    for f in (0, 4):
        run = sorted((X / pat.format(f=f) / 'runs' / f'fold{f}').glob('*'))[0]; lg = []
        for e in range(8, 13):
            with np.load(run / f'predictions_epoch_{e:02d}.npz', allow_pickle=True) as z:
                ix, p = z['indices'].astype(int), z['pred'].astype(np.float64)
            lg.append(np.log(p))
        lg = np.mean(lg, 0); keep = np.array([int(i) in pos for i in ix]); L[[pos[int(i)] for i in ix[keep]]] = lg[keep]
    return L
w = float(sys.argv[1]); mem = dict(a.split('=', 1) for a in sys.argv[2:]); logs = {k: member(v) for k, v in mem.items()}
rows = np.where(np.isfinite(next(iter(logs.values()))).all(1))[0]; A = BP._b['A']; er = E.ape(BP.REF_BEST[rows], BP.y[rows]); s0 = E.official(er)
def err(names):
    c = np.zeros_like(BP.H)
    for k in names: c[rows] += w * (logs[k][rows] - np.log(A[rows]))
    return E.official(E.ape(BP.pipe(BP.add(BP.BEST, {ff: c for ff in range(5)}))[rows], BP.y[rows]))
import itertools
for r in range(1, len(logs) + 1):
    for combo in itertools.combinations(logs, r):
        s = err(combo); print(json.dumps(dict(members=combo, w=w, delta=round(s['mean'] - s0['mean'], 4), per={k: round(s[k] - s0[k], 4) for k in K})))
