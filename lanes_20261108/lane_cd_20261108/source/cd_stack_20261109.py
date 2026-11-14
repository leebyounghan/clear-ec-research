"""Full CD stack read (CD lane, 2026-11-09), five folds nested. usage: cd_stack_20261109.py m1,m2,... [m1,m2,... ...]
Per fold f, on folds != f: separate log-blend weights for P1, A2 snapshot and each member (grid 0.05), stretch; then the seg_loose mask CD
(calibrated affine in log, frames >= 50 interior cells) with one weight from {0..0.25}, stretch again. Reference: best pipeline CD 6.264."""
import os, sys, glob, json, itertools, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
from cd_common import *
pos = R.pos
F = np.full((len(yc), 17), np.nan)
for pth in sorted(glob.glob(str(X / 'lane_hex_20261108/seg_loose/c*.npz'))):
    z = np.load(pth); names = list(z['names'])
    for k, fr in enumerate(z['frames']):
        r = pos.get(int(fr), -1)
        if r >= 0: F[r] = z['F'][k]
nm = {n: i for i, n in enumerate(names)}; have = F[:, nm['n_int']] >= 50; ly = np.log(np.where(yc > 0, yc, np.nan))
lm = np.log(np.where(have, F[:, nm['cd_mm2']], 1.))
def stack(keys, step=0.05):
    Ms = [R.member(k) for k in keys]; assert all(np.isfinite(m).all() for m in Ms), 'need five folds'
    Lc = [np.log(R.P1), np.log(R.A2)] + [np.log(m) for m in Ms]
    wm_ = np.round(np.arange(0, 0.401, step), 3); grids = []
    for a in np.round(np.arange(0.25, 0.751, step), 3):
        for mw in itertools.product(wm_, repeat=len(Ms)):
            r = 1 - a - sum(mw)
            if r >= 0.099: grids.append((round(r, 3), a) + tuple(mw))
    out0 = np.full(len(yc), np.nan); out1 = np.full(len(yc), np.nan); info = {}
    for f in range(5):
        tr = fold != f; best = None
        for wt in grids:
            q = np.exp(sum(w * l for w, l in zip(wt, Lc))); prm = fit_stretch1(q, tr & good); e = cd(apply1(q, prm), tr)
            if best is None or e < best[0]: best = (e, wt, prm, q)
        _, wt, prm, q = best; out0[fold == f] = apply1(q[fold == f], prm)
        trm = tr & have & (yc > 0) & good; co = np.polyfit(lm[trm], ly[trm], 1); Mc = np.polyval(co, lm); b2 = None
        for wmk in (0, 0.05, 0.1, 0.15, 0.2, 0.25):
            qq = np.where(have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
            if b2 is None or e < b2[0]: b2 = (e, wmk, qq, p2)
        out1[fold == f] = apply1(b2[2][fold == f], b2[3]); info[f] = (wt, b2[1])
    print('weights per fold (P1, A2, members..., mask):', json.dumps({str(k): v for k, v in info.items()}), flush=True)
    rep(f'members {"+".join(keys)}', out0, REF, rows=yc > 0)
    rep(f'members {"+".join(keys)} + mask CD', out1, REF, rows=yc > 0)
    return out1
if __name__ == '__main__':
    for s in sys.argv[1:]: stack(s.split(','))
