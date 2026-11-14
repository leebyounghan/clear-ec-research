"""Fold-0 look at the backbone lane's ViT-L/16 (raw track, epochs 8-12) as a CD member, raw blends (no stretch), fixed weights not chosen on fold 0."""
import os, numpy as np
os.environ['A2SNAP'] = '0'
import cd_read_20261108 as R
from cd_common import *
pos = R.pos; r0 = fold == 0; ok = r0 & (yc > 0); ly = np.log(np.where(yc > 0, yc, np.nan))
L = []
for e in range(8, 13):
    z = np.load(X / f'lane_backbone_20261108/runs/vitl16_avg_f0_s46/raw/runs/fold0/dinov3_l16t/predictions_epoch_{e:02d}.npz'); p = np.full(len(yc), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; L.append(np.log(p))
VL = np.exp(np.mean(L, 0)); Dt = R.member('dinov3_s46_tta'); Ct = R.member('convnext_s46_tta'); Bc = R.b['B'][:, 0]
for nm, p in (('ViT-L e8-12', VL), ('DINOv3 B TTA', Dt), ('ConvNeXt TTA', Ct)):
    print(nm, 'CD raw', round(cd(p, r0), 3), 'errcorr B', round(np.corrcoef(np.log(p[ok]) - ly[ok], np.log(Bc[ok]) - ly[ok])[0, 1], 3), 'errcorr DINOv3', round(np.corrcoef(np.log(p[ok]) - ly[ok], np.log(Dt[ok]) - ly[ok])[0, 1], 3), flush=True)
def g(*ps): return np.exp(np.mean([np.log(p) for p in ps], 0))
for w in (0.2, 0.25, 0.3, 0.35):
    for nm, F in (('D', Dt), ('D+C', g(Dt, Ct)), ('D+C+L', g(Dt, Ct, VL)), ('D+L', g(Dt, VL)), ('L', VL)):
        q = np.exp((1 - w) * np.log(Bc) + w * np.log(F)); print(w, nm, round(cd(q, r0), 4), flush=True)
print('B raw', round(cd(Bc, r0), 4))
