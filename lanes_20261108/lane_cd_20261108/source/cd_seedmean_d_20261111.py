"""Seed audit (coordinator 11-11): the CD stack's DINOv3-B member read as the two-seed mean (geometric mean of s46 and s47 four-flip TTA over
epochs 16-20) in place of s46 alone. Five folds, nested (weights and mask weight chosen on the other four folds; same search that reproduces
the freeze candidate 6.1726). No seed is picked: the seed mean is the member from here on."""
import os, itertools, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *
rows = np.isin(fold, range(5))
D46 = R.member('dinov3_s46_tta'); D47 = R.member('dinov3_s47_tta'); V = R.member('vitl_s4647_tt')
assert np.isfinite(D46).all() and np.isfinite(D47).all()
print('alone CD: DINOv3 s46', round(cd(D46, rows), 3), '| s47', round(cd(D47, rows), 3), '| seed mean', round(cd(np.sqrt(D46 * D47), rows), 3), flush=True)


def stackN(Ms, name):
    Lc = [np.log(R.P1), np.log(R.A2)] + [np.log(m) for m in Ms]; grids = []
    for a in np.round(np.arange(0.25, 0.751, 0.05), 3):
        for mw in itertools.product(np.round(np.arange(0, 0.401, 0.05), 3), repeat=len(Ms)):
            r = 1 - a - sum(mw)
            if r >= 0.099: grids.append((round(r, 3), a) + tuple(mw))
    out = np.full(len(yc), np.nan); info = {}
    for f in range(5):
        tr = fold != f; best = None
        for wt in grids:
            q = np.exp(sum(w * l for w, l in zip(wt, Lc))); e = cd(apply1(q, fit_stretch1(q, tr & good)), tr)
            if best is None or e < best[0]: best = (e, wt, q)
        q = best[2]; trm = tr & K.have & (yc > 0) & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
        for wmk in (0, 0.05, 0.1, 0.15, 0.2):
            qq = np.where(K.have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
            if b2 is None or e < b2[0]: b2 = (e, wmk, qq, p2)
        out[fold == f] = apply1(b2[2][fold == f], b2[3]); info[f] = (best[1], b2[1])
    rep(name, out, REF, rows=rows, note='weights (P1, A, D, V, mask) ' + str(info)); return out


a = stackN([D46, V], 'freeze stack, DINOv3 single seed s46 (five folds)')
b = stackN([np.sqrt(D46 * D47), V], 'stack with DINOv3 seed mean (s46, s47)')
rep('seed mean vs single seed', b, a, rows=rows)
np.savez_compressed(X / 'lane_cd_20261108/reads/seedmean_d_oof_20261111.npz', idx=idx, single=a, seedmean=b, D46=D46, D47=D47)
