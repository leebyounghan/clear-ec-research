"""Other encoders as ensemble members (2026-11-05): added at 20 % to B on the rows they cover, with their own scalar from the other folds of B's calibration not available, so a per-metric median-ratio scalar on the same fold (slight optimism, stated). Gate: added <= -0.02 on folds 0 and 3, then fill folds 1, 2, 4."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
E = C.E; K = list(C.K); b = E.load_base(); y, B, fold = b['y'], b['B'], b['fold']; g = E.donors(b['ids']); pos = {int(i): n for n, i in enumerate(b['idx'])}
R = C.X / 'meta_ceiling_20261103/sweep/backbones'; F3 = C.X / 'phase2_foundation_train_20261002/runs/fold3'
items = [(a, 3, F3 / a / 'predictions.npz', 'pred') for a in ('dinov3_vitb', 'convnext_base')]
for p in sorted(R.glob('fold*/*/predictions_tta4.npz')):
    items.append((p.parent.name.rsplit('_s', 1)[0], int(p.parent.parent.name[4:]), p, 'pred_single')); items.append((p.parent.name.rsplit('_s', 1)[0] + ':tta', int(p.parent.parent.name[4:]), p, 'pred_tta4'))
for a, f, p, k in items:
    z = np.load(p, allow_pickle=True); r = np.array([pos[int(i)] for i in z['indices']]); q = z[k].astype(float); s = y[r] > 0; c = np.array([np.median(y[r][s[:, j], j] / q[s[:, j], j]) for j in range(3)]); q2 = q * c; o = {}
    for w in (0.1, 0.2, 0.33):
        h = (1 - w) * B[r] + w * q2; pb = E.paired_bootstrap(E.ape(h, y[r]), E.ape(B[r], y[r]), np.unique(g[r], return_inverse=True)[1]); o[w] = (round(pb['delta'], 4), [round(v, 4) for v in pb['ci95']])
    e = [round(float(np.corrcoef(np.log(q[s[:, j], j] / y[r][s[:, j], j]), np.log(B[r][s[:, j], j] / y[r][s[:, j], j]))[0, 1]), 3) for j in range(3)]; print(a, 'fold', f, 'alone', round(E.official(E.ape(q2, y[r]))['mean'], 3), 'B', round(E.official(E.ape(B[r], y[r]))['mean'], 3), 'err r with B', e, 'added', json.dumps(o), flush=True)
