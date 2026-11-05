"""Per-row fold-4 arrays of the composition submission 2 will ship on (for the hex6 transfer check), 2026-10-31.
A3 = mean of four-flip TTA of recipe A s46/s47/s48 (SSL 1003) x the fold-4 LOFO scalars of the two-seed hybrid
(s48 is missing on other folds, so A3-specific LOFO scalars cannot be fitted); hist6c = Phase-1 six
((5 * five + SSL sixth) / 6) x the fixed Phase-1 scalars; shipped = 0.5 * A3 + 0.5 * hist6c.
"""
import json, sys
import numpy as np
sys.path.insert(0, 'experiments/recipeA_analysis_20261028/source')
import hybrid_oof_20261028 as H

E = H.E
z = np.load('experiments/recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True)
idx, fold, y, Bcur = z['indices'], z['fold'], z['target'], z['B_hybrid_w0p5']
pos = {int(i): k for k, i in enumerate(idx)}
rows = np.where(fold == 4)[0]
rpos = {int(idx[r]): n for n, r in enumerate(rows)}
A = []
for seed in (46, 47, 48):
    with np.load(E/H.RECIPE[seed][4]/'runs'/'fold4'/'cpdino_vitb'/'predictions_tta4.npz') as t:
        a = np.full((len(rows), 3), np.nan)
        for i, v in zip(t['indices'].astype(int), t['pred_tta4']):
            a[rpos[int(i)]] = v
        A.append(a)
sc = np.array(json.load(open('experiments/recipeA_analysis_20261028/hybrid_oof.json'))['recipeA_lofo_scalars']['4'])
A3 = np.mean(A, 0) * sc
preds = []
for name, pat in H.HIST.items():
    with np.load(E/pat.format(f=4), allow_pickle=True) as t:
        p = np.full((len(rows), 3), np.nan)
        for i, v in zip(t['indices'].astype(int), t['pred'].astype(np.float64)):
            p[rpos[int(i)]] = v
        preds.append(p)
five = np.mean(preds, 0)
with np.load(E/H.SIXTH.format(f=4)) as t:
    sixth = np.full((len(rows), 3), np.nan)
    for i, v in zip(t['indices'].astype(int), t['pred'].astype(np.float64)):
        sixth[rpos[int(i)]] = v
hist6c = (5 * five + sixth) / 6 * H.SCALE
shipped = 0.5 * A3 + 0.5 * hist6c
assert np.isfinite(A3).all() and np.isfinite(hist6c).all()
out = 'experiments/recipeA_analysis_20261028/fold4_shipped_base_20261031.npz'
np.savez_compressed(out, indices=idx[rows], ids=z['ids'][rows], target=y[rows], A3_3seed=A3, historical_six_calibrated=hist6c,
                    shipped_hybrid=shipped, B_hybrid_w0p5=Bcur[rows], A3_scalars=sc, phase1_scalars=np.asarray(H.SCALE))
print(out, len(rows), 'shipped', H.official(shipped, y[rows]), 'B', H.official(Bcur[rows], y[rows]))
