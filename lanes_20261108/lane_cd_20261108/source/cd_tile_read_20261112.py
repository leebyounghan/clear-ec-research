"""(copy for D-CDSIMPLE-001 (b), 2026-11-12: the backbone lane's native-tile CD-only DINOv3-B s46, runs/tile_cd_f{f}_s46/raw, CD column only, member = exp mean log epochs 8-12; completion = predictions_epoch_12.npz.)
Backbone lane's synthetic-pretrained then fine-tuned DINOv3-B (synthft3; raw track) in the CD stack over the folds where it exists, strictly
nested over those folds: for each available fold f, P1 and recipe A scalars, weights, mask calibration and stretch from the other available folds.
Member forms: 'lm' = exp mean log epochs 8-12 (the backbone lane's read); 'tt' = exp(0.5 mean log e8-11 + 0.5 log final TTA). Tested added, in place
of the DINOv3-B seed mean, and averaged (geometric) into that slot; baseline = the v3 stack on the same folds. usage: cd_synthft_avail_20261111.py [lm|tt] [track]"""
import os, sys, itertools, json, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *
FORM = sys.argv[1] if len(sys.argv) > 1 else 'lm'; TRACK = sys.argv[2] if len(sys.argv) > 2 else 'raw'


def vec(p):
    z = np.load(p); v = np.full(len(yc), np.nan); v[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; return v


S = np.full(len(yc), np.nan)
for f in range(5):
    r = X / f'lane_backbone_20261108/runs/tile_cd_f{f}_s46/raw/runs/fold{f}/dinov3_b16'
    if not (r / 'predictions_epoch_12.npz').exists(): continue
    L = [np.log(vec(r / f'predictions_epoch_{e:02d}.npz')) for e in range(8, 13)]
    m = np.mean(L, 0) if FORM == 'lm' else 0.5 * np.mean(L[:4], 0) + 0.5 * np.log(vec(r / 'predictions_final_tta.npz'))
    S[fold == f] = np.exp(m[fold == f])
AV = [f for f in range(5) if np.isfinite(S[fold == f]).all()]; rows = np.isin(fold, AV)
SCALE_CD = 0.9970446110788349
P1raw = R.P1 / SCALE_CD; Asnap_raw = R.A2 / R.sc
D = np.exp(0.5 * np.log(R.member('dinov3_s46_tta')) + 0.5 * np.log(R.member('dinov3_s47_tta'))); V = R.member('vitl_s4647_tt')
print(f'tile CD-only, form {FORM}, folds {AV} | alone CD: tile {cd(S, rows):.3f}, DINOv3 seed mean {cd(D, rows):.3f}, DINOv3 s46 {cd(R.member("dinov3_s46_tta"), rows):.3f}', flush=True)


def wmed_scalar(p, tr):
    k = tr & (yc > 0); return float(C.wmed(yc[k] / p[k], p[k] / yc[k]))


def stack(members, name):
    out = np.full(len(yc), np.nan); info = {}
    for f in AV:
        tr = rows & (fold != f)
        Lc = [np.log(P1raw * wmed_scalar(P1raw, tr)), np.log(Asnap_raw * wmed_scalar(Asnap_raw, tr))] + [np.log(np.where(np.isfinite(m), m, 1.)) for m in members]
        grids = []
        for a in np.round(np.arange(0.25, 0.751, 0.05), 3):
            for mw in itertools.product(np.round(np.arange(0, 0.401, 0.05), 3), repeat=len(members)):
                r = 1 - a - sum(mw)
                if r >= 0.099: grids.append((round(r, 3), a) + tuple(mw))
        best = None
        for wt in grids:
            q = np.exp(sum(w * l for w, l in zip(wt, Lc))); e = cd(apply1(q, fit_stretch1(q, tr & good)), tr)
            if best is None or e < best[0]: best = (e, wt, q)
        q = best[2]; trm = tr & K.have & (yc > 0) & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
        for wmk in (0, 0.05, 0.1, 0.15, 0.2):
            qq = np.where(K.have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
            if b2 is None or e < b2[0]: b2 = (e, wmk, qq, p2)
        out[fold == f] = apply1(b2[2][fold == f], b2[3]); info[f] = (best[1], b2[1])
    rep(name, out, REF, rows=rows, note='weights (P1, A, members..., mask) ' + str(info)); return out


a = stack([D, V], f'v3 stack (strict) on folds {AV}')
b = stack([D, V, S], f'+ tile CD-only {FORM} (added)'); rep('tile added vs v3', b, a, rows=rows)
c = stack([S, V], f'tile {FORM} in place of the DINOv3 seed mean'); rep('tile replacing DINOv3 vs v3', c, a, rows=rows)
d = stack([np.sqrt(D * S), V], f'DINOv3 slot = geometric mean(DINOv3 seed mean, tile {FORM})'); rep('tile averaged into DINOv3 slot vs v3', d, a, rows=rows)
