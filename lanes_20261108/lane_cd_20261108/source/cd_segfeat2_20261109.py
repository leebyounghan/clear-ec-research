"""Mask-density variants as the fifth CD blend component (CD lane, 2026-11-09). Same protocol as cd_segfeat (b): calibrated affine in log on
folds != f (frames with >= 50 interior cells), one weight from a coarse grid on folds != f, stretch, applied to fold f. Base = SEP stack 6.204."""
import os, json, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
from cd_common import *
pos = R.pos
z = np.load(LANE / 'feats/seg_loose_area_stats.npz'); names = [str(n) for n in z['names']]; Sx = np.full((len(yc), len(names)), np.nan)
for i, fr in enumerate(z['frames']):
    if int(fr) in pos: Sx[pos[int(fr)]] = z['S'][i]
ix = {n: i for i, n in enumerate(names)}
D = R.member('dinov3_s46_tta'); Cx = R.member('convnext_s46_tta'); Lc = [np.log(m) for m in (R.P1, R.A2, D, Cx)]
note = json.loads(R.read_sep(['dinov3_s46_tta', 'convnext_s46_tta'])['note'])
Q = {f: np.exp(sum(w * l for w, l in zip(note[str(f)], Lc))) for f in range(5)}
base = np.full(len(yc), np.nan)
for f in range(5): base[fold == f] = apply1(Q[f][fold == f], fit_stretch1(Q[f], (fold != f) & good))
ly = np.log(np.where(yc > 0, yc, np.nan)); have0 = Sx[:, ix['n_int']] >= 50
print('base', round(cd(base), 4), 'frames >=50 interior', int(have0.sum()), flush=True)
def run(name, dens, have):
    ld = np.log(np.where(have & np.isfinite(dens) & (dens > 0), dens, np.nan)); have = have & np.isfinite(ld); okr = have & (yc > 0)
    print(name, 'corr log y', round(float(np.corrcoef(ld[okr], ly[okr])[0, 1]), 3), flush=True)
    out = np.full(len(yc), np.nan); ws = {}
    for f in range(5):
        tr = (fold != f) & okr & good; co = np.polyfit(ld[tr], ly[tr], 1); Mc = np.where(have, np.polyval(co, np.where(have, ld, 0)), 0)
        best = None
        for wm in (0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
            q = np.where(have, np.exp((1 - wm) * np.log(Q[f]) + wm * Mc), Q[f]); prm = fit_stretch1(q, (fold != f) & good); e = cd(apply1(q, prm), fold != f)
            if best is None or e < best[0]: best = (e, wm, q, prm)
        ws[f] = best[1]; out[fold == f] = apply1(best[2][fold == f], best[3])
    rep(name, out, base, rows=yc > 0, note=str(ws)); return out
res = {}
for n in ('cd_mean', 'cd_median', 'cd_trim', 'cd_band', 'cd_win_mean', 'cd_win_median'):
    res[n] = run(n, Sx[:, ix[n]], have0)
gm = np.exp(np.nanmean(np.log(Sx[:, [ix['cd_median'], ix['cd_trim'], ix['cd_band']]]), 1)); run('geo(median, trim, band)', gm, have0)
run('cd_band, >=30 cells', Sx[:, ix['cd_band']], Sx[:, ix['n_int']] >= 30)
run('cd_band, >=100 cells', Sx[:, ix['cd_band']], Sx[:, ix['n_int']] >= 100)
