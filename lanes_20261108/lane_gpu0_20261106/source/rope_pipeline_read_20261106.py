"""RoPE scale at inference, read on the pipeline (lane GPU 0, 2026-11-06). Recipe A in the hybrid is the mean of seeds 46 and 47 with flip TTA and fold-out scalars. The change is applied as a log difference: mean over the two seeds of log(member at scale r) - log(member at scale 1), no flips, multiplied onto A. Then hybrid, hex6 weight on HEX, graph and simulation corrections as in the best pipeline, stretch refit per fold.
Usage: rope_pipeline_read_20261106.py <variants, comma>. Variants missing a member are skipped. Saves corrections in best_pipeline format."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
E = BP.E
G = Path('experiments/lane_gpu0_20261106/rope_grid'); OUT = Path('experiments/lane_gpu0_20261106/rope_cor'); OUT.mkdir(exist_ok=True)
A, P1, B = BP._b['A'], BP._b['P1'], BP.B
pos = {int(i): n for n, i in enumerate(BP.idx)}
w_lin = np.abs(0.5 * A + 0.5 * P1 - B).max(); w_geo = np.abs(np.exp(0.5 * np.log(A) + 0.5 * np.log(P1)) - B).max()
print('hybrid form: max difference linear', round(float(w_lin), 6), 'geometric', round(float(w_geo), 6))
linear = w_lin < w_geo
SEEDS = (46, 47)


def delta(v):
    d = np.full_like(A, np.nan)
    for f in range(5):
        parts = []
        for s in SEEDS:
            a, b = G / f'f{f}_s{s}_{v}.npz', G / f'f{f}_s{s}_1.0.npz'
            if not a.exists():
                return None
            za, zb = np.load(a), np.load(b); assert np.array_equal(za['indices'], zb['indices'])
            parts.append(np.log(za['pred'].astype(np.float64)) - np.log(zb['pred'].astype(np.float64)))
        rows = np.array([pos[int(i)] for i in za['indices']]); d[rows] = np.mean(parts, 0)
    assert not np.isnan(d).any(), 'every row belongs to one fold'
    return d


def correction(d):
    An = A * np.exp(d)
    Bn = 0.5 * An + 0.5 * P1 if linear else np.exp(0.5 * np.log(An) + 0.5 * np.log(P1))
    c = np.log(Bn) - np.log(B)
    c[:, 2] *= np.where(BP._have, 0.72, 1.0)  # the hex6 mix keeps 72 % of hybrid B on HEX where it applies
    return c


variants = sys.argv[1].split(','); COR = {}
for v in variants:
    d = delta(v)
    if d is None:
        print(v, 'not complete'); continue
    print(f'scale {v}: median change of recipe A, percent', np.round(np.median(d, 0) * 100, 2).tolist())
    c = correction(d); COR[v] = c; cf = {f: c for f in range(5)}
    BP.save(OUT / f'scale_{v}.npz', cf)
    BP.report(f'hybrid B + stretch, scale {v}, against scale 1', BP.pipe(cf), BP.REF_HEX6)
    BP.report(f'best pipeline, scale {v}, against best', BP.pipe(BP.add(BP.BEST, cf)), BP.REF_BEST)
if len(COR) > 1:
    # Scale chosen on the other folds.
    scores = {v: E.ape(BP.pipe(BP.add(BP.BEST, {f: COR[v] for f in range(5)})), BP.y) for v in COR}
    scores['1.0'] = E.ape(BP.REF_BEST, BP.y)
    pick = {}; Cn = np.zeros_like(A)
    for f in range(5):
        best = min(scores, key=lambda v: float(np.nanmean(scores[v][BP.fold != f], 0).mean()))
        pick[f] = best
        if best != '1.0':
            Cn[BP.fold == f] = COR[best][BP.fold == f]
    print('scale chosen on the other folds', pick)
    # The stretch is refit on all rows of the other folds using the same correction per fold's choice.
    cf = {f: (COR[pick[f]] if pick[f] != '1.0' else np.zeros_like(A)) for f in range(5)}
    BP.report('best pipeline, scale chosen on the other folds', BP.pipe(BP.add(BP.BEST, cf)), BP.REF_BEST)
    BP.save(OUT / 'scale_nested.npz', cf)
