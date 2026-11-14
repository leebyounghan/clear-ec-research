"""Window-crop recipe A (backbone lane) in the CD stack over the folds where it exists, nested over those folds (weights and mask weight chosen on
the other available folds). Stack: P1 / A2snap / DINOv3 TTA / ViT-L seed mean (+ window member) + mask."""
import os, itertools, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *
W = R.member('win2x_s46_tt'); AV = [f for f in range(5) if np.isfinite(W[fold == f]).all()]; rows = np.isin(fold, AV)
print('window member folds', AV, 'alone CD', round(cd(W, rows), 3), flush=True)
D = R.member('dinov3_s46_tta'); V = R.member('vitl_s4647_tt')
def stackN(Ms, name):
    Lc = [np.log(R.P1), np.log(R.A2)] + [np.log(np.where(np.isfinite(m), m, 1.)) for m in Ms]; grids = []
    for a in np.round(np.arange(0.25, 0.751, 0.05), 3):
        for mw in itertools.product(np.round(np.arange(0, 0.401, 0.05), 3), repeat=len(Ms)):
            r = 1 - a - sum(mw)
            if r >= 0.099: grids.append((round(r, 3), a) + tuple(mw))
    out = np.full(len(yc), np.nan); info = {}
    for f in AV:
        tr = rows & (fold != f); best = None
        for wt in grids:
            q = np.exp(sum(w * l for w, l in zip(wt, Lc))); e = cd(apply1(q, fit_stretch1(q, tr & good)), tr)
            if best is None or e < best[0]: best = (e, wt, q)
        q = best[2]; trm = tr & K.have & (yc > 0) & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
        for wmk in (0, 0.05, 0.1, 0.15, 0.2):
            qq = np.where(K.have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
            if b2 is None or e < b2[0]: b2 = (e, wmk, qq, p2)
        out[fold == f] = apply1(b2[2][fold == f], b2[3]); info[f] = (best[1], b2[1])
    rep(name, out, REF, rows=rows, note=str(info)); return out
a = stackN([D, V], f'stack (DINOv3 + ViT-L mean) + mask, folds {AV}')
b = stackN([D, V, W], 'stack + window recipe A + mask')
rep('window added vs stack', b, a, rows=rows)
