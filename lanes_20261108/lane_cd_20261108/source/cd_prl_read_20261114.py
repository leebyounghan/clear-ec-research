"""PRL probe extra read (coordinator 11-14 night). Fold 0, members in the true form (e16-20 four-flip TTA): PRL K=1 arm
(tta/sweep_prl1_f0_tta4_e16_20.npz) vs control P (tta/screen_plain_f0_tta4_e16_20.npz). (1) Residual correlations on fold-0 CD rows,
residual = log(pred / label), of the arm and of P with recipe A (0.5 (s46 + s47) e20 TTA), ViT-L (geometric mean of s46 / s47 final TTA),
P1 and the incumbent D (geometric mean of s46b / s47 e16-20 TTA); lower = more diverse. (2) Rows above 50 % APE for each, and CD on the rows
where neither is above 50 %, with the donor-clustered paired interval. Writes reads/prl_read_20261114.json."""
import os, json, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
from cd_common import *
y3 = BP.y.astype(float); f0 = (fold == 0) & (yc > 0)
def tta(name):
    z = np.load(LANE / 'tta' / name); v = np.full((len(yc), 3), np.nan); v[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, :3]; return v
ARM = tta('sweep_prl1_f0_tta4_e16_20.npz'); P = tta('screen_plain_f0_tta4_e16_20.npz')
def vit(sd, f=0):
    r = (X / f'lane_backbone_20261108/runs/vitl16_avg_f{f}_s46' if sd == 's46' else LANE / f'arms/vitl16_f{f}_{sd}') / f'raw/runs/fold{f}/dinov3_l16t'
    z = np.load(r / 'predictions_final_tta.npz'); p = np.full(len(yc), np.nan); p[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; return p
gm = lambda a, b: np.exp(0.5 * np.log(a) + 0.5 * np.log(b))
others = {'recipeA': 0.5 * (R.seeds[46][:, 0] + R.seeds[47][:, 0]), 'vitl': gm(vit('s46'), vit('s47')), 'P1': R.P1,
          'D_incumbent': gm(R.member('dinov3_s46_tta'), R.member('dinov3_s47_tta'))}
res = lambda p: np.log(p[f0] / yc[f0])
out = {'rows': int(f0.sum()), 'resid_corr': {}}
for k, o in others.items():
    ok_ = np.isfinite(res(o))
    out['resid_corr'][k] = {'arm': round(float(np.corrcoef(res(ARM[:, 0])[ok_], res(o)[ok_])[0, 1]), 4), 'P': round(float(np.corrcoef(res(P[:, 0])[ok_], res(o)[ok_])[0, 1]), 4)}
out['resid_corr']['arm_vs_P'] = round(float(np.corrcoef(res(ARM[:, 0]), res(P[:, 0]))[0, 1]), 4)
ape = lambda p: np.abs(p[:, 0] / yc - 1) * 100
aA, aP = ape(ARM), ape(P)
out['rows_above_50'] = {'arm': int(((aA > 50) & f0).sum()), 'P': int(((aP > 50) & f0).sum()), 'either': int((((aA > 50) | (aP > 50)) & f0).sum()), 'both': int(((aA > 50) & (aP > 50) & f0).sum())}
out['rows_above_100'] = {'arm': int(((aA > 100) & f0).sum()), 'P': int(((aP > 100) & f0).sum())}
def paired(a, b, rows):
    d = a[rows] - b[rows]; u, inv = np.unique(donors[rows], return_inverse=True); s = np.bincount(inv, d); n = np.bincount(inv); rng = np.random.default_rng(0)
    bs = [s[k].sum() / n[k].sum() for k in (rng.integers(0, len(u), len(u)) for _ in range(2000))]
    return dict(arm=round(float(a[rows].mean()), 4), P=round(float(b[rows].mean()), 4), diff=round(float(d.mean()), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], rows=int(rows.sum()))
out['cd_all'] = paired(aA, aP, f0)
out['cd_without_rows_above_50_either'] = paired(aA, aP, f0 & (aA <= 50) & (aP <= 50))
out['cd_without_rows_above_50_P'] = paired(aA, aP, f0 & (aP <= 50))
json.dump(out, open(LANE / 'reads/prl_read_20261114.json', 'w'), indent=1); print(json.dumps(out, indent=1))
