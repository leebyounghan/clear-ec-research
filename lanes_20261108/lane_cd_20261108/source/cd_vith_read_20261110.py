"""DINOv3 ViT-H+/16 read (CD lane, 2026-11-10). (1) member alone on CD/CV/HEX against ViT-L s46 on the same folds (raw e8-11 + final TTA);
(2) added to the CD stack (P1/A2snap/DINOv3 TTA/ViT-L seed mean + mask): nested over the folds where ViT-H+ exists (cross-fit if two folds)."""
import os, itertools, json, numpy as np
os.environ['A2SNAP'] = '1'
import cd_vitl47_f04_20261109 as V4
import cd_read_20261108 as R
from cd_common import *
pos = R.pos; y = R.b['y']
def three(root_fmt, fam, folds):
    out = np.full((len(yc), 3), np.nan)
    for f in folds:
        r = X / root_fmt.format(f=f) / f'raw/runs/fold{f}/{fam}'
        L = []
        for e in range(8, 12):
            z = np.load(r / f'predictions_epoch_{e:02d}.npz'); p = np.full((len(yc), 3), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred']; L.append(np.log(p))
        z = np.load(r / 'predictions_final_tta.npz'); p = np.full((len(yc), 3), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred']
        m = np.exp(0.5 * np.mean(L, 0) + 0.5 * np.log(p)); out[fold == f] = m[fold == f]
    return out
H = R.member('vith_s46_tt'); AV = [f for f in range(5) if np.isfinite(H[fold == f]).all()]; rows = np.isin(fold, AV)
print('ViT-H+ folds available', AV, flush=True)
H3 = three('lane_cd_20261108/arms/vith16_f{f}_s46', 'dinov3_hp16t', AV)
L3 = three('lane_backbone_20261108/runs/vitl16_avg_f{f}_s46' if False else 'lane_cd_20261108/arms/vitl16_f{f}_s46', 'dinov3_l16t', [f for f in AV if f in (1, 2, 3)])
for f in AV:
    if f in (0, 4): L3[fold == f] = three('lane_backbone_20261108/runs/vitl16_avg_f{f}_s46', 'dinov3_l16t', [f])[fold == f]
def mape3(P):
    o = {}
    for j, k in enumerate(('CD', 'CV', 'HEX')):
        ok = rows & (y[:, j] > 0); o[k] = round(float(np.mean(np.abs(P[ok, j] - y[ok, j]) / y[ok, j]) * 100), 3)
    o['mean'] = round(float(np.mean(list(o.values()))), 3); return o
print('member alone (raw, no stretch) ViT-H+', json.dumps(mape3(H3)), '| ViT-L s46', json.dumps(mape3(L3)), flush=True)
np.savez_compressed(LANE / 'reads/vith16_s46_member_preds.npz', idx=idx, fold=fold, pred=H3, folds=np.array(AV))
D = R.member('dinov3_s46_tta'); V = R.member('vitl_s4647_tt')
if len(AV) == 2 and AV == [0, 4]:
    a = V4.stack2([D, V], 'stack (DINOv3 + ViT-L mean) + mask, folds 0,4'); b = V4.stack2([D, V, H], 'stack + ViT-H+ + mask'); c = V4.stack2([D, np.exp((np.log(V) + np.log(H)) / 2)], 'stack with ViT-L mean and ViT-H+ averaged + mask')
    rep('ViT-H+ added vs stack', b, a, rows=SCREEN); rep('ViT-L/ViT-H+ averaged vs stack', c, a, rows=SCREEN)
else:
    import cd_vitl47_avail_20261109  # noqa (nested version for >2 folds is in cd_stack once five folds exist)
