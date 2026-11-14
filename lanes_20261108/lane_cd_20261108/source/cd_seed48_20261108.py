"""Does a third recipe A seed (s48, tta4) help CD inside the family blend? Folds where s48 exists (0, 4), cross-fit between them."""
import os, numpy as np
os.environ['A2SNAP'] = '0'
import cd_read_20261108 as R
from cd_common import *
pos = R.pos; sc = R.sc
s48 = np.full(len(yc), np.nan)
for f in (0, 4):
    z = np.load(X / f'oof_recipeA_20261027/fold{f}/regression_std_s48/runs/fold{f}/cpdino_vitb/predictions_tta4.npz'); p = np.full(len(yc), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred_tta4'][:, 0]; s48[fold == f] = p[fold == f]
A2 = R.A2; A3 = (R.seeds[46][:, 0] + R.seeds[47][:, 0] + s48 / sc * sc) / 3 * sc if False else ((R.seeds[46][:, 0] + R.seeds[47][:, 0] + s48) / 3) * sc
F = np.exp(0.5 * np.log(R.member('dinov3_s46_tta')) + 0.5 * np.log(R.member('convnext_s46_tta')))
rows = np.isin(fold, (0, 4))
print('A2 alone', round(cd(A2, rows), 4), 'A3 alone', round(cd(A3, rows), 4), flush=True)
def blend(A, Fm):
    Lc = [np.log(R.P1), np.log(A), np.log(Fm)]; Q = {wt: np.exp(sum(w * l for w, l in zip(wt, Lc))) for wt in R.G}; out = np.full(len(yc), np.nan)
    for f, o in ((0, 4), (4, 0)):
        tr = fold == o; best = None
        for wt, q in Q.items():
            prm = fit_stretch1(q, tr & good); e = cd(apply1(q, prm), tr)
            if best is None or e < best[0]: best = (e, wt, prm)
        out[fold == f] = apply1(Q[best[1]][fold == f], best[2])
    return out
pa = blend(A2, F); pb = blend(A3, F)
rep('A2 + families (folds 0,4 cross-fit)', pa, REF, rows=rows); rep('A3 (s48 added) + families', pb, REF, rows=rows); rep('A3 vs A2', pb, pa, rows=rows)
