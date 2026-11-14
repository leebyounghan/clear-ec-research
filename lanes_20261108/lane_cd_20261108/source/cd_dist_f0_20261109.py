"""Fold-0 first look at the distribution-head DINOv3 (epochs 16-20 single view) vs the recipe-loss DINOv3 s46 (same SSL, seed), raw blends with
the freeze-candidate weights (not chosen on fold 0), no stretch."""
import os, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
from cd_common import *
pos = R.pos; r0 = fold == 0; ok = r0 & (yc > 0); ly = np.log(np.where(yc > 0, yc, np.nan))
def ld(root, f):
    L = []
    for e in range(16, 21):
        z = np.load(X / root / f'runs/fold{f}/dinov3_b16/predictions_epoch_{e:02d}.npz'); p = np.full(len(yc), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; L.append(np.log(p))
    return np.exp(np.mean(L, 0))
Dd = ld('lane_cd_20261108/arms/dinov3_dist_f0_s46', 0); Dr = ld('lane_gpu3_20261106/runs/dinov3_b16_f0_s46', 0); V = R.member('vitl_s4647_tt')
for nm, p in (('dist', Dd), ('recipe', Dr)):
    print(nm, 'member CD fold0', round(cd(p, r0), 3), 'errcorr with recipe', round(float(np.corrcoef(np.log(p[ok]) - ly[ok], np.log(Dr[ok]) - ly[ok])[0, 1]), 3), flush=True)
def blend(D, extra=None, we=0.0):
    l = 0.25 * np.log(R.P1) + 0.45 * np.log(R.A2) + 0.10 * np.log(D) + 0.20 * np.log(V)
    if extra is not None: l = (1 - we) * l + we * np.log(extra)
    return np.exp(l)
print('raw blend fold0: recipe D', round(cd(blend(Dr), r0), 4), '| dist D', round(cd(blend(Dd), r0), 4), flush=True)
for we in (0.05, 0.1, 0.15):
    print(f'raw blend fold0: recipe D + dist extra w {we}', round(cd(blend(Dr, Dd, we), r0), 4), flush=True)
