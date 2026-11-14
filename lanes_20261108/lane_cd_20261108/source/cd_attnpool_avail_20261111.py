"""Backbone lane's attention-pool DINOv3-B/16 (recipe A with a learned 4-query attention pool; read-only) in the CD stack, nested over the
folds where it exists (weights and mask weight chosen on the other available folds), as cd_win2x_avail_20261110. Member form per track:
exp(0.5 mean log e8-11 + 0.5 log final TTA). Tested as an added member, in place of DINOv3-B TTA, and averaged into the DINOv3-B slot."""
import os, sys, itertools, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *
TRACKS = sys.argv[1:] or ['raw', 'ema998']


def attn(track):
    out = np.full(len(yc), np.nan)
    for f in range(5):
        r = X / f'lane_backbone_20261108/runs/attnpool_f{f}_dinov3b16_s46/{track}/runs/fold{f}/dinov3_b16'
        if not (r / 'predictions_final_tta.npz').exists(): continue
        L = []
        for e in range(8, 13):
            z = np.load(r / f'predictions_epoch_{e:02d}.npz'); p = np.full(len(yc), np.nan); p[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; L.append(np.log(p))
        z = np.load(r / 'predictions_final_tta.npz'); p = np.full(len(yc), np.nan); p[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]
        m = 0.5 * np.mean(L[:4], 0) + 0.5 * np.log(p)
        out[fold == f] = np.exp(m[fold == f])
    return out


A = {t: attn(t) for t in TRACKS}
AV = [f for f in range(5) if all(np.isfinite(A[t][fold == f]).all() for t in TRACKS)]; rows = np.isin(fold, AV)
D = R.member('dinov3_s46_tta'); V = R.member('vitl_s4647_tt')
if os.environ.get('DMEAN'):  # 11-11 seed audit: the DINOv3 member is the s46/s47 seed mean (freeze v3)
    D = np.exp(0.5 * np.log(D) + 0.5 * np.log(R.member('dinov3_s47_tta')))
print('folds', AV, '| DINOv3-B TTA alone', round(cd(D, rows), 3), '|', ' '.join(f'attn {t} alone {round(cd(A[t], rows), 3)}' for t in TRACKS), flush=True)


def stackN(Ms, name):
    Lc = [np.log(R.P1), np.log(R.A2)] + [np.log(np.where(np.isfinite(m), m, 1.)) for m in Ms]; grids = []
    for a in np.round(np.arange(0.25, 0.751, 0.05), 3):
        for mw in itertools.product(np.round(np.arange(0, 0.401, 0.05), 3), repeat=len(Ms)):
            r = 1 - a - sum(mw)
            if r >= 0.099: grids.append((round(r, 3), a) + tuple(mw))
    out = np.full(len(yc), np.nan); info = {}
    for f in AV:
        tr = rows & (fold != f); best = None
        for wt in grids:
            q = np.exp(sum(w * l for w, l in zip(wt, Lc))); e = cd(apply1(q, fit_stretch1(q, tr & good)), tr)
            if best is None or e < best[0]: best = (e, wt, q)
        q = best[2]; trm = tr & K.have & (yc > 0) & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
        for wmk in (0, 0.05, 0.1, 0.15, 0.2):
            qq = np.where(K.have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
            if b2 is None or e < b2[0]: b2 = (e, wmk, qq, p2)
        out[fold == f] = apply1(b2[2][fold == f], b2[3]); info[f] = (best[1], b2[1])
    rep(name, out, REF, rows=rows, note='weights (P1, A, members..., mask) ' + str(info)); return out


a = stackN([D, V], f'stack (DINOv3 TTA + ViT-L mean) + mask, folds {AV}')
for t in TRACKS:
    c = A[t]
    b = stackN([D, V, c], f'stack + attention-pool {t} (added member)'); rep(f'attention-pool {t} added vs stack', b, a, rows=rows)
    d = stackN([c, V], f'attention-pool {t} in place of DINOv3-B'); rep(f'attention-pool {t} replacing DINOv3-B vs stack', d, a, rows=rows)
    g = stackN([np.sqrt(D * c), V], f'DINOv3-B slot = geometric mean(DINOv3-B TTA, attention-pool {t})'); rep(f'attention-pool {t} averaged into DINOv3-B slot vs stack', g, a, rows=rows)
