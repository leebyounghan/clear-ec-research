"""Backbone lane's C-Mixup DINOv3-B/16 (folds 0, 4; read-only) in the CD stack, two-fold cross-fit with mask (same search as
cd_vitl47_f04_20261109.stack2). Member forms per track: 'lm' = exp mean log epochs 8-12; 'tt' = exp(0.5 mean log e8-11 + 0.5 log final TTA)
(the ViT-L read). Tested as an added member, in place of DINOv3-B TTA, and averaged into the DINOv3-B slot. Weights are printed per fold."""
import os, itertools, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *
rows = np.isin(fold, [0, 4])


def cmix(track, form):
    out = np.full(len(yc), np.nan)
    for f in (0, 4):
        r = X / f'lane_backbone_20261108/runs/cmix_f{f}_dinov3b16_s46/{track}/runs/fold{f}/dinov3_b16'
        L = []
        for e in range(8, 13):
            z = np.load(r / f'predictions_epoch_{e:02d}.npz'); p = np.full(len(yc), np.nan); p[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; L.append(np.log(p))
        m = np.mean(L, 0)
        if form == 'tt':
            z = np.load(r / 'predictions_final_tta.npz'); p = np.full(len(yc), np.nan); p[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]
            m = 0.5 * np.mean(L[:4], 0) + 0.5 * np.log(p)
        out[fold == f] = np.exp(m[fold == f])
    return out


def stack2(Ms, name):
    Lc = [np.log(R.P1), np.log(R.A2)] + [np.log(m) for m in Ms]; grids = []
    for a in np.round(np.arange(0.25, 0.751, 0.05), 3):
        for mw in itertools.product(np.round(np.arange(0, 0.401, 0.05), 3), repeat=len(Ms)):
            r = 1 - a - sum(mw)
            if r >= 0.099: grids.append((round(r, 3), a) + tuple(mw))
    out = np.full(len(yc), np.nan); info = {}
    for f, o in ((0, 4), (4, 0)):
        tr = fold == o; best = None
        for wt in grids:
            q = np.exp(sum(w * l for w, l in zip(wt, Lc))); prm = fit_stretch1(q, tr & good); e = cd(apply1(q, prm), tr)
            if best is None or e < best[0]: best = (e, wt, q)
        q = best[2]; trm = tr & K.have & (yc > 0) & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
        for wmk in (0, 0.05, 0.1, 0.15, 0.2):
            qq = np.where(K.have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
            if b2 is None or e < b2[0]: b2 = (e, wmk, qq, p2)
        out[fold == f] = apply1(b2[2][fold == f], b2[3]); info[f] = (best[1], b2[1])
    rep(name, out, REF, rows=rows, note='weights (P1, A, members..., mask) ' + str(info)); return out


D = R.member('dinov3_s46_tta'); V = R.member('vitl_s4647_tt')
print('DINOv3-B TTA alone CD folds 0,4', round(cd(D, rows), 3), flush=True)
C = {}
for t in ('raw', 'ema998'):
    for fm in ('lm', 'tt'):
        C[(t, fm)] = cmix(t, fm); print(f'C-Mixup {t} {fm} alone CD folds 0,4', round(cd(C[(t, fm)], rows), 3), flush=True)
a = stack2([D, V], 'stack (DINOv3 TTA + ViT-L mean) + mask, folds 0,4')
for t in ('raw', 'ema998'):
    c = C[(t, 'tt')]
    b = stack2([D, V, c], f'stack + C-Mixup {t} tt (added member)'); rep(f'C-Mixup {t} added vs stack', b, a, rows=rows)
    d = stack2([c, V], f'C-Mixup {t} tt in place of DINOv3-B'); rep(f'C-Mixup {t} replacing DINOv3-B vs stack', d, a, rows=rows)
    g = stack2([np.sqrt(D * c), V], f'DINOv3-B slot = geometric mean(DINOv3-B TTA, C-Mixup {t} tt)'); rep(f'C-Mixup {t} averaged into DINOv3-B slot vs stack', g, a, rows=rows)
