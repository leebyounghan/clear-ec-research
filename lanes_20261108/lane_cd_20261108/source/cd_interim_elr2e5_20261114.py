"""Interim read (coordinator 00:15 EST 11-14): sweep arm encoder LR 2e-5 vs control P (both bb_runner, DINOv3-B/16 fold 0 s46, same SSL init),
each as exp(mean log single view epochs 8-12). Member CD/CV/HEX on fold 0; tier-p CD stack on fold 0 with every constant fitted on fold 4 with
the incumbent members, D's s46 half replaced by the arm vs by P, and the arm / P added as a third equal D member; log-residual correlations
with recipe A (e20 TTA), ViT-L (final TTA) and P1 on fold 0 for the arm and for P. Writes reads/cd_interim_elr2e5_20261114.json."""
import os, json, itertools, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *
f0 = fold == 0; ok = yc > 0; y3 = BP.y.astype(float)
def vec3(p):
    z = np.load(p); v = np.full((len(yc), 3), np.nan); v[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, :3]; return v
def sv(root, e0=8, e1=12):
    d = LANE / 'arms' / root / 'raw/runs/fold0/dinov3_b16'; return np.exp(np.mean([np.log(vec3(d / f'predictions_epoch_{e:02d}.npz')) for e in range(e0, e1 + 1)], 0))
ARM = sv('sweep_elr2e5_f0_s46'); P = sv('screen_plain_f0_s46')
def m3(p, rows): return [round(float(np.mean(np.abs(p[rows & (y3[:, j] > 0), j] / y3[rows & (y3[:, j] > 0), j] - 1)) * 100), 4) for j in range(3)]
out = {'member_e8_12_arm': m3(ARM, f0), 'member_e8_12_P': m3(P, f0)}
SCALE_CD = 0.9970446110788349; P1raw = R.P1 / SCALE_CD; Afin_raw = 0.5 * (R.seeds[46][:, 0] + R.seeds[47][:, 0])
D46 = R.member('dinov3_s46_tta'); D47 = R.member('dinov3_s47_tta'); Dm = np.exp(0.5 * np.log(D46) + 0.5 * np.log(D47))
def vit(sd):
    o = np.full(len(yc), np.nan)
    for f in range(5):
        r = (X / f'lane_backbone_20261108/runs/vitl16_avg_f{f}_s46' if (f in (0, 4) and sd == 's46') else LANE / f'arms/vitl16_f{f}_{sd}') / f'raw/runs/fold{f}/dinov3_l16t'
        z = np.load(r / 'predictions_final_tta.npz'); p = np.full(len(yc), np.nan); p[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; o[fold == f] = p[fold == f]
    return o
Vf = np.exp(0.5 * np.log(vit('s46')) + 0.5 * np.log(vit('s47')))
def wsc(p, tr): k = tr & ok; return float(C.wmed(yc[k] / p[k], p[k] / yc[k]))
tr = (fold == 4) & ok; sP = wsc(P1raw, tr); sA = wsc(Afin_raw, tr)
Lc = [np.log(P1raw * sP), np.log(Afin_raw * sA), np.log(Dm), np.log(Vf)]; grids = []
for a in np.round(np.arange(0.25, 0.751, 0.05), 3):
    for mw in itertools.product(np.round(np.arange(0, 0.401, 0.05), 3), repeat=2):
        r_ = 1 - a - sum(mw)
        if r_ >= 0.099: grids.append((round(r_, 3), a) + tuple(mw))
best = None
for wt in grids:
    q = np.exp(sum(w * l for w, l in zip(wt, Lc))); e = cd(apply1(q, fit_stretch1(q, tr & good)), tr)
    if best is None or e < best[0]: best = (e, wt, q)
wt = best[1]; q = best[2]; trm = tr & K.have & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
for wmk in (0, 0.05, 0.1, 0.15, 0.2):
    qq = np.where(K.have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
    if b2 is None or e < b2[0]: b2 = (e, wmk, p2)
def stack(D):
    qa = np.exp(wt[0] * np.log(P1raw * sP) + wt[1] * np.log(Afin_raw * sA) + wt[2] * np.log(D) + wt[3] * np.log(Vf))
    qa = np.where(K.have, np.exp((1 - b2[1]) * np.log(qa) + b2[1] * Mc), qa); return apply1(qa, b2[2])
def rep(M):
    Dr = Dm.copy(); Dr[f0] = np.exp(0.5 * np.log(M[f0, 0]) + 0.5 * np.log(D47[f0]))
    Da = Dm.copy(); Da[f0] = np.exp((np.log(M[f0, 0]) + np.log(D46[f0]) + np.log(D47[f0])) / 3)
    return round(cd(stack(Dr), f0), 4), round(cd(stack(Da), f0), 4)
ra, aa = rep(ARM); rp, ap = rep(P)
out['stack_fold0'] = dict(params_fit_on_fold4=dict(weights=wt, mask_w=b2[1]), incumbent=round(cd(stack(Dm), f0), 4), replace_arm=ra, replace_P=rp, replace_delta=round(ra - rp, 4),
                          added_arm=aa, added_P=ap, added_delta=round(aa - ap, 4))
lr = lambda p: np.log(p / np.where(ok, yc, 1)); m = f0 & ok
cols = {'arm': lr(ARM[:, 0]), 'P': lr(P[:, 0]), 'recipeA': lr(Afin_raw * sA), 'ViT-L': lr(Vf), 'P1': lr(P1raw * sP), 'D_incumbent': lr(Dm)}
out['residual_corr_fold0'] = {f'{a} vs {b}': round(float(np.corrcoef(cols[a][m], cols[b][m])[0, 1]), 3) for a in ('arm', 'P') for b in ('recipeA', 'ViT-L', 'P1', 'D_incumbent')}
out['mean_log_residual_fold0'] = {k: round(float(np.mean(v[m])), 4) for k, v in cols.items()}
print(json.dumps(out, indent=1)); json.dump(out, open(LANE / 'reads/cd_interim_elr2e5_20261114.json', 'w'), indent=1, default=str)
