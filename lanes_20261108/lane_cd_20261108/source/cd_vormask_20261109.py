"""One CPU try (coordinator, 11-09): Voronoi-of-mask-centroids CD (rootcause_20261109/out/emulator_feats.npz, read-only) as the mask component,
in place of / together with seg_loose cd_mm2. Members fixed to the current stack (DINOv3 s46 TTA + ViT-L s46); nested five folds as cd_stack."""
import os, itertools, numpy as np
os.environ['A2SNAP'] = '1'
import cd_stack_20261109 as K
import cd_read_20261108 as R
from cd_common import *
z = np.load(X / 'rootcause_20261109/out/emulator_feats.npz')
ly = K.ly; ok = yc > 0
for k in ('vor_whole', 'vor_lw', 'vor_mb', 'ok_whole'):
    v = z[k][:, 0]; m = np.isfinite(v) & (v > 0) & ok
    print(k, 'finite', int(np.isfinite(v).sum()), 'corr log y', round(float(np.corrcoef(np.log(v[m]), ly[m])[0, 1]), 3), flush=True)
D = R.member('dinov3_s46_tta'); V = R.member('vitl_s46_tt')
Lc = [np.log(R.P1), np.log(R.A2), np.log(D), np.log(V)]
def run(name, comps):
    """comps: list of (log density array, have mask); one weight each from a coarse grid, nested; members' weights fixed per fold by search."""
    grids = []
    for a in np.round(np.arange(0.3, 0.651, 0.05), 3):
        for d in (0.05, 0.1, 0.15, 0.2):
            for v in (0.05, 0.1, 0.15, 0.2):
                r = 1 - a - d - v
                if r >= 0.099: grids.append((round(r, 3), a, d, v))
    out = np.full(len(yc), np.nan); info = {}
    for f in range(5):
        tr = fold != f; best = None
        for wt in grids:
            q = np.exp(sum(w * l for w, l in zip(wt, Lc))); e = cd(apply1(q, fit_stretch1(q, tr & good)), tr)
            if best is None or e < best[0]: best = (e, wt, q)
        q = best[2]; Mcs = []
        for ld, hv in comps:
            trm = tr & hv & ok & good; co = np.polyfit(ld[trm], ly[trm], 1); Mcs.append((np.polyval(co, np.where(hv, ld, 0)), hv))
        b2 = None
        for ws in itertools.product((0, 0.05, 0.1, 0.15), repeat=len(Mcs)):
            lq = np.log(q).copy()
            for wm, (Mc, hv) in zip(ws, Mcs): lq = np.where(hv, lq + wm * (Mc - lq), lq)
            qq = np.exp(lq); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
            if b2 is None or e < b2[0]: b2 = (e, ws, qq, p2)
        out[fold == f] = apply1(b2[2][fold == f], b2[3]); info[f] = (best[1], b2[1])
    rep(name, out, REF, rows=ok, note=str(info)); return out
segm = (np.where(K.have, K.lm, 0), K.have)
vw = z['vor_whole'][:, 0]; hv = np.isfinite(vw) & (vw > 0) & (z['vor_whole'][:, 3] >= 50); vor = (np.log(np.where(hv, vw, 1)), hv)
a = run('stack + seg cd_mm2 (reference)', [segm])
b = run('stack + Voronoi mask CD (vor_whole, >=50 cells) instead', [vor])
c = run('stack + seg cd_mm2 + Voronoi mask CD', [segm, vor])
rep('Voronoi + seg vs seg only', c, a, rows=ok)
