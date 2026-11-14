"""HEX lane's Cellpose-SAM watershed features (seg_loose, read-only) on CD (CD lane, 2026-11-08). Base: SEP five-fold stack
(P1/A2snap/DINOv3 TTA/ConvNeXt TTA, nested weights + stretch, CD 6.204). Frames with >= 50 interior cells only; others unchanged.
(a) ridge (one alpha) on the log residual of the base, features: log mask CD, log mask CD - log base, area CV, coverage, log n_int, h5/h6/h7,
    log base; fit on folds != f, applied to fold f.
(b) mask CD calibrated to log CD (affine, folds != f) as a fifth blend component with one weight (grid), then stretch, nested."""
import os, json, glob, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
from cd_common import *
pos = R.pos
F = np.full((len(yc), 17), np.nan)
for pth in sorted(glob.glob(str(X / 'lane_hex_20261108/seg_loose/c*.npz'))):
    z = np.load(pth); names = list(z['names']); r = [pos.get(int(i), -1) for i in z['frames']]
    for k, rr in enumerate(r):
        if rr >= 0: F[rr] = z['F'][k]
nm = {n: i for i, n in enumerate(names)}
have = np.isfinite(F[:, nm['cd_mm2']]) & (F[:, nm['n_int']] >= 50)
print('frames with features', int(np.isfinite(F[:, 0]).sum()), 'with >=50 interior cells', int(have.sum()), flush=True)
D = R.member('dinov3_s46_tta'); Cx = R.member('convnext_s46_tta'); Ms = [R.P1, R.A2, D, Cx]; Lc = [np.log(m) for m in Ms]
base_rep = R.read_sep(['dinov3_s46_tta', 'convnext_s46_tta']); note = json.loads(base_rep['note'])
base = np.full(len(yc), np.nan)
for f in range(5):
    q = np.exp(sum(w * l for w, l in zip(note[str(f)], Lc))); base[fold == f] = apply1(q[fold == f], fit_stretch1(q, (fold != f) & good))
print('base CD', round(cd(base), 4), flush=True)
ly = np.log(np.where(yc > 0, yc, np.nan)); lm = np.log(np.where(have, F[:, nm['cd_mm2']], np.nan))
ok = have & (yc > 0)
print('mask CD alone (raw, >=50 cells) CD', round(cd(np.where(have, F[:, nm['cd_mm2']], base), have), 3), 'base on same rows', round(cd(base, have), 3),
      'corr(log mask, log y)', round(float(np.corrcoef(lm[ok], ly[ok])[0, 1]), 3), 'corr(log base, log y)', round(float(np.corrcoef(np.log(base[ok]), ly[ok])[0, 1]), 3), flush=True)
def design(rows):
    lb = np.log(base)
    cols = [lm, lm - lb, F[:, nm['area_cv']], F[:, nm['coverage']], np.log(F[:, nm['n_int']]), F[:, nm['h5']], F[:, nm['h6']], F[:, nm['h7']], lb]
    return np.stack([c[rows] for c in cols], 1)
# (a) ridge on residual
corr = np.zeros(len(yc))
for f in range(5):
    tr = (fold != f) & ok & good; te = (fold == f) & have
    Xtr = design(tr); mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9; Xs = (Xtr - mu) / sd
    t = ly[tr] - np.log(base[tr]); A = Xs.T @ Xs + 10.0 * np.eye(Xs.shape[1]); w = np.linalg.solve(A, Xs.T @ (t - t.mean())); b0 = t.mean()
    corr[te] = b0 + ((design(te) - mu) / sd) @ w
pa = base * np.exp(corr)
rep('(a) ridge on CD residual with seg features (>=50 cells)', pa, base, rows=yc > 0)
rep('(a) vs best pipeline', pa, REF, rows=yc > 0)
# (b) calibrated mask CD as fifth component
outb = np.full(len(yc), np.nan); ws = {}
for f in range(5):
    tr = (fold != f) & ok & good; co = np.polyfit(lm[tr], ly[tr], 1); Mc = np.where(have, np.exp(np.polyval(co, np.where(have, lm, 0))), np.nan)
    best = None
    for wm in (0, 0.05, 0.1, 0.15, 0.2, 0.25):
        q = np.exp(sum(w * l for w, l in zip(note[str(f)], Lc))); q = np.where(have, np.exp((1 - wm) * np.log(q) + wm * np.log(np.where(have, Mc, 1.))), q)
        prm = fit_stretch1(q, (fold != f) & good); e = cd(apply1(q, prm), fold != f)
        if best is None or e < best[0]: best = (e, wm, q, prm)
    ws[f] = best[1]; outb[fold == f] = apply1(best[2][fold == f], best[3])
rep('(b) calibrated mask CD as blend component (>=50 cells)', outb, base, rows=yc > 0, note=str(ws))
rep('(b) vs best pipeline', outb, REF, rows=yc > 0)
