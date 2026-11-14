"""Value of a third recipe A seed (s48, exists on folds 0 and 4) in the current CD stack: A3 snapshot vs A2 snapshot, two-fold cross-fit, with DINOv3 TTA,
ViT-L s46 and the mask component (cd_vitl47_f04 protocol)."""
import os, numpy as np
os.environ['A2SNAP'] = '1'
import cd_vitl47_f04_20261109 as V
import cd_read_20261108 as R
from cd_common import *
pos = R.pos; sc = R.sc
def rec(f, e_list, tta):
    d = X / f'oof_recipeA_20261027/fold{f}/regression_std_s48/runs/fold{f}/cpdino_vitb'; L = []
    if tta:
        z = np.load(d / 'predictions_tta4.npz'); p = np.full(len(yc), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred_tta4'][:, 0]; return np.log(p)
    for e in e_list:
        z = np.load(d / f'predictions_epoch_{e:02d}.npz'); p = np.full(len(yc), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; L.append(np.log(p))
    return np.mean(L, 0)
s48 = np.full(len(yc), np.nan)
for f in (0, 4):
    m = np.exp(0.5 * rec(f, None, True) + 0.5 * rec(f, range(8, 20), False)); s48[fold == f] = m[fold == f]
A2 = R.A2.copy(); A3 = np.exp((2 * np.log(A2) + np.log(s48 * sc)) / 3)
D = R.member('dinov3_s46_tta'); Vl = R.member('vitl_s46_tt')
a = V.stack2([D, Vl], 'A2snap stack (folds 0,4)')
R.A2 = A3
b = V.stack2([D, Vl], 'A3snap stack (s48 added)')
rep('A3 vs A2', b, a, rows=SCREEN)
