"""ViT-L s47 on the folds where it exists, nested over those folds (weights and mask weight chosen on the other available folds). Compares the current stack's members
(DINOv3 TTA + ViT-L s46) with ViT-L s46 replaced by the mean of s46 and s47, and with s47 as a separate member."""
import os, itertools, numpy as np
os.environ['A2SNAP'] = '1'
import cd_stack_20261109 as K
import cd_read_20261108 as R
from cd_common import *
V47_ = R.member("vitl_s47_tt"); AV = [f for f in range(5) if np.isfinite(V47_[fold == f]).all()]; rows = np.isin(fold, AV); print("folds", AV, flush=True)
def stack2(Ms, name):
    Lc = [np.log(R.P1), np.log(R.A2)] + [np.log(m) for m in Ms]; grids = []
    for a in np.round(np.arange(0.25, 0.751, 0.05), 3):
        for mw in itertools.product(np.round(np.arange(0, 0.401, 0.05), 3), repeat=len(Ms)):
            r = 1 - a - sum(mw)
            if r >= 0.099: grids.append((round(r, 3), a) + tuple(mw))
    out = np.full(len(yc), np.nan); info = {}
    for f in AV:
        tr = rows & (fold != f); best = None
        for wt in grids:
            q = np.exp(sum(w * l for w, l in zip(wt, Lc))); prm = fit_stretch1(q, tr & good); e = cd(apply1(q, prm), tr)
            if best is None or e < best[0]: best = (e, wt, q)
        q = best[2]; trm = tr & K.have & (yc > 0) & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
        for wmk in (0, 0.05, 0.1, 0.15, 0.2):
            qq = np.where(K.have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
            if b2 is None or e < b2[0]: b2 = (e, wmk, qq, p2)
        out[fold == f] = apply1(b2[2][fold == f], b2[3]); info[f] = (best[1], b2[1])
    rep(name, out, REF, rows=rows, note=str(info)); return out
D = R.member('dinov3_s46_tta'); V46 = R.member('vitl_s46_tt'); V47 = R.member('vitl_s47_tt')
print('ViT-L alone on available folds: s46', round(cd(V46, rows), 3), 's47', round(cd(V47, rows), 3), 'mean', round(cd(np.sqrt(V46 * V47), rows), 3), flush=True)
a = stack2([D, V46], 'DINOv3 + ViT-L s46 + mask (available folds)')
b = stack2([D, np.sqrt(V46 * V47)], 'DINOv3 + ViT-L mean(s46,s47) + mask')
c = stack2([D, V46, V47], 'DINOv3 + ViT-L s46 + ViT-L s47 separate + mask')
rep('mean(s46,s47) vs s46', b, a, rows=rows)
