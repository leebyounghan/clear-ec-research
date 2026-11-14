"""Strict five-fold read of a follow-up DINOv3-B member (coordinator rule 00:15 EST 11-14). Member per fold = four-flip TTA over epochs 16-20:
fold 0 from the screen arm (tta/<F0NAME>_tta4_e16_20.npz), folds 1-4 from the follow-up runs (tta/fu_<NAME>_f<f>_tta4_e16_20.npz).
Against tier-p CD (t1 form, 6.1737) with the same strict procedure (P1 / A scalars, weight grid, mask slot, stretch all fit on the other four
folds): (a) new member in place of the plain s46 half of D; (b) in place of the s47 half (deploy-neutral: s47 is CD-only, s46b also serves
the CV term); (c) added as a third equal D member. Folds 1-4 reported separately (fold 0 chose the setting). Member CD / CV / HEX on five folds
against plain s46. Column: CD without the 11 rows of APE > 100 % under the frozen tier-p CD OOF. env NAME, F0NAME. Writes reads/cd_fu_read_<NAME>_20261114.json."""
import os, json, itertools, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *
NAME = os.environ['NAME']; F0 = os.environ['F0NAME']; ok = yc > 0; y3 = BP.y.astype(float)
def vec3(p):
    z = np.load(p); v = np.full((len(yc), 3), np.nan); v[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, :3]; return v
NEW = np.full((len(yc), 3), np.nan)
for f in range(5):
    v = vec3(LANE / 'tta' / (f'{F0}_tta4_e16_20.npz' if f == 0 else f'fu_{NAME}_f{f}_tta4_e16_20.npz')); NEW[fold == f] = v[fold == f]
assert np.isfinite(NEW[:, 0]).all()
SCALE_CD = 0.9970446110788349; P1raw = R.P1 / SCALE_CD; Afin_raw = 0.5 * (R.seeds[46][:, 0] + R.seeds[47][:, 0])
D46 = R.member('dinov3_s46_tta'); D47 = R.member('dinov3_s47_tta'); N0 = NEW[:, 0]
def vit(sd):
    o = np.full(len(yc), np.nan)
    for f in range(5):
        r = (X / f'lane_backbone_20261108/runs/vitl16_avg_f{f}_s46' if (f in (0, 4) and sd == 's46') else LANE / f'arms/vitl16_f{f}_{sd}') / f'raw/runs/fold{f}/dinov3_l16t'
        z = np.load(r / 'predictions_final_tta.npz'); p = np.full(len(yc), np.nan); p[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; o[fold == f] = p[fold == f]
    return o
Vf = np.exp(0.5 * np.log(vit('s46')) + 0.5 * np.log(vit('s47')))
gm = lambda *a: np.exp(np.mean([np.log(x) for x in a], 0))
VARS = {'t1_incumbent': gm(D46, D47), 'a_replace_s46': gm(N0, D47), 'b_replace_s47_deploy_neutral': gm(D46, N0), 'c_added_third': gm(D46, D47, N0)}
def wsc(p, tr): k = tr & ok; return float(C.wmed(yc[k] / p[k], p[k] / yc[k]))
def fit(tr, Dk):
    sP = wsc(P1raw, tr); sA = wsc(Afin_raw, tr); Lc = [np.log(P1raw * sP), np.log(Afin_raw * sA), np.log(Dk), np.log(Vf)]; grids = []
    for a in np.round(np.arange(0.25, 0.751, 0.05), 3):
        for mw in itertools.product(np.round(np.arange(0, 0.401, 0.05), 3), repeat=2):
            r_ = 1 - a - sum(mw)
            if r_ >= 0.099: grids.append((round(r_, 3), a) + tuple(mw))
    best = None
    for wt in grids:
        q = np.exp(sum(w * l for w, l in zip(wt, Lc))); e = cd(apply1(q, fit_stretch1(q, tr & good)), tr)
        if best is None or e < best[0]: best = (e, wt, q)
    q = best[2]; trm = tr & K.have & ok & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
    for wmk in (0, 0.05, 0.1, 0.15, 0.2):
        qq = np.where(K.have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
        if b2 is None or e < b2[0]: b2 = (e, wmk, qq, p2)
    return apply1(b2[2], b2[3]), best[1]
_t1 = np.load(LANE / 'reads/cd_tier_t1_20261112.npz')['oof_cd']; EX = ok & (np.abs(_t1 / np.where(ok, yc, 1) - 1) > 1.0); f14 = fold != 0
res = {'name': NAME}; OOF = {}
for v, Dk in VARS.items():
    o = np.full(len(yc), np.nan); w = {}
    for f in range(5):
        p, wt = fit(fold != f, Dk); o[fold == f] = p[fold == f]; w[f] = wt
    OOF[v] = o
base = OOF['t1_incumbent']
for v, o in OOF.items():
    d, ci = boot(o, base, ok)
    res[v] = dict(strict=round(cd(o), 4), delta=round(cd(o) - cd(base), 4), ci95=[round(x, 4) for x in ci], per_fold_delta={f: round(cd(o, fold == f) - cd(base, fold == f), 4) for f in range(5)},
                  folds14_delta=round(cd(o, f14) - cd(base, f14), 4), strict_ex100=round(cd(o, ~EX), 4), delta_ex100=round(cd(o, ~EX) - cd(base, ~EX), 4))
def m3(p, rows): return [round(float(np.mean(np.abs(p[rows & (y3[:, j] > 0), j] / y3[rows & (y3[:, j] > 0), j] - 1)) * 100), 4) for j in range(3)]
P46 = np.full((len(yc), 3), np.nan)
for f in range(5):
    pth = LANE / 'tta' / f'dinov3_s46_f{f}_tta4_e16_20.npz'; v = vec3(pth); P46[fold == f] = v[fold == f]
res['member_cd_cv_hex_new'] = m3(NEW, np.ones(len(yc), bool)); res['member_cd_cv_hex_plain_s46'] = m3(P46, np.ones(len(yc), bool))
res['member_cd_cv_hex_new_folds14'] = m3(NEW, f14); res['member_cd_cv_hex_plain_s46_folds14'] = m3(P46, f14)
print(json.dumps(res, indent=1)); json.dump(res, open(LANE / f'reads/cd_fu_read_{NAME}_20261114.json', 'w'), indent=1, default=str)
