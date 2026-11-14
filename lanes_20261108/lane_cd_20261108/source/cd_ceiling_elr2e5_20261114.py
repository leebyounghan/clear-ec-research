"""Ceiling read (coordinator 00:25 EST 11-14), REPORT ONLY, IN-SAMPLE on fold 0: tier-p CD weights, mask slot and stretch refitted on fold 0
itself (a) with the encoder LR 2e-5 arm in D's s46 half, (b) with the arm as a fifth member with its own weight (grid 0-0.4), (c) the same
with control P; arm and P as exp(mean log single view epochs 8-12). Standalone fold-0 CD of every member. Writes reads/cd_ceiling_elr2e5_20261114.json."""
import os, json, itertools, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *
f0 = fold == 0; ok = yc > 0
def vec(p):
    z = np.load(p); v = np.full(len(yc), np.nan); v[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; return v
def sv(root): d = LANE / 'arms' / root / 'raw/runs/fold0/dinov3_b16'; return np.exp(np.mean([np.log(vec(d / f'predictions_epoch_{e:02d}.npz')) for e in range(8, 13)], 0))
ARM = sv('sweep_elr2e5_f0_s46'); PC = sv('screen_plain_f0_s46')
SCALE_CD = 0.9970446110788349; P1raw = R.P1 / SCALE_CD; Afin_raw = 0.5 * (R.seeds[46][:, 0] + R.seeds[47][:, 0])
D46 = R.member('dinov3_s46_tta'); D47 = R.member('dinov3_s47_tta'); Dm = np.exp(0.5 * np.log(D46) + 0.5 * np.log(D47))
def vit(sd):
    o = np.full(len(yc), np.nan)
    for f in range(5):
        r = (X / f'lane_backbone_20261108/runs/vitl16_avg_f{f}_s46' if (f in (0, 4) and sd == 's46') else LANE / f'arms/vitl16_f{f}_{sd}') / f'raw/runs/fold{f}/dinov3_l16t'
        o[fold == f] = vec(r / 'predictions_final_tta.npz')[fold == f]
    return o
Vf = np.exp(0.5 * np.log(vit('s46')) + 0.5 * np.log(vit('s47')))
def wsc(p, tr): k = tr & ok; return float(C.wmed(yc[k] / p[k], p[k] / yc[k]))
tr = f0 & ok
def fit(Dk, extra=None):
    sP = wsc(P1raw, tr); sA = wsc(Afin_raw, tr); Lc = [np.log(P1raw * sP), np.log(Afin_raw * sA), np.log(Dk), np.log(Vf)] + ([np.log(extra)] if extra is not None else []); grids = []
    for a in np.round(np.arange(0.25, 0.751, 0.05), 3):
        for mw in itertools.product(np.round(np.arange(0, 0.401, 0.05), 3), repeat=len(Lc) - 2):
            r_ = 1 - a - sum(mw)
            if r_ >= 0.099: grids.append((round(r_, 3), a) + tuple(mw))
    best = None
    for wt in grids:
        q = np.exp(sum(w * l for w, l in zip(wt, Lc))); e = cd(apply1(q, fit_stretch1(q, tr & good)), tr)
        if best is None or e < best[0]: best = (e, wt, q)
    q = best[2]; trm = tr & K.have & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
    for wmk in (0, 0.05, 0.1, 0.15, 0.2):
        qq = np.where(K.have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
        if b2 is None or e < b2[0]: b2 = (e, wmk, qq, p2)
    return dict(cd_in_sample=round(cd(apply1(b2[2], b2[3]), f0), 4), weights_P1_A_D_V_extra=best[1], mask_w=b2[1])
gm = lambda a, b: np.exp(0.5 * np.log(a) + 0.5 * np.log(b))
out = {'label': 'IN-SAMPLE on fold 0 (weights, mask and stretch refitted on the scored rows): an upper bound, not a result',
       'incumbent_refit': fit(Dm), 'a_arm_in_s46_half': fit(gm(ARM, D47)), 'a_P_in_s46_half': fit(gm(PC, D47)),
       'b_arm_as_fifth_member': fit(Dm, ARM), 'b_P_as_fifth_member': fit(Dm, PC)}
out['standalone_fold0_cd'] = {'P1_x_scale': round(cd(R.P1, f0), 4), 'recipeA_s46s47_e20tta_x_lofo': round(cd(Afin_raw * R.sc, f0), 4), 'recipeA_snapshot_A2': round(cd(R.A2, f0), 4),
                              'DINOv3B_seedmean_tta': round(cd(Dm, f0), 4), 'DINOv3B_s46_tta': round(cd(D46, f0), 4), 'ViT-L_tt': round(cd(R.member('vitl_s4647_tt'), f0), 4), 'ViT-L_final_tta': round(cd(Vf, f0), 4),
                              'arm_elr2e5_sv_e8_12': round(cd(ARM, f0), 4), 'P_sv_e8_12': round(cd(PC, f0), 4)}
print(json.dumps(out, indent=1)); json.dump(out, open(LANE / 'reads/cd_ceiling_elr2e5_20261114.json', 'w'), indent=1, default=str)
