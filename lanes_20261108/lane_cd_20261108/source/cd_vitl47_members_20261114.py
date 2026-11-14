"""Member table of ViT-L s47 encoder LR 2e-5 per fold as folds finish (coordinator 10:36 EST 11-14): final.pt four-flip TTA, raw track, vs the
plain s47 run of the fold (the tier-p V s47 half), in the same form; for comparison the s46 2e-5 arm vs plain s46 on the same fold. CD / CV /
HEX per fold, CD difference with a donor-clustered paired bootstrap (2,000). Writes reads/vitl47_members_20261114.json."""
import json, numpy as np
import cd_read_20261108 as R
from cd_common import *
y3 = BP.y.astype(float)
def v3(p):
    z = np.load(p); v = np.full((len(yc), 3), np.nan); v[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, :3]; return v
def m3(p, rows): return [round(float(np.mean(np.abs(p[rows & (y3[:, j] > 0), j] / y3[rows & (y3[:, j] > 0), j] - 1)) * 100), 4) for j in range(3)]
def paired(a, b, rows):
    r = rows & (y3[:, 0] > 0); d = (np.abs(a[r, 0] / y3[r, 0] - 1) - np.abs(b[r, 0] / y3[r, 0] - 1)) * 100; u, inv = np.unique(donors[r], return_inverse=True)
    s = np.bincount(inv, d); n = np.bincount(inv); rng = np.random.default_rng(0); bs = [s[k].sum() / n[k].sum() for k in (rng.integers(0, len(u), len(u)) for _ in range(2000))]
    return dict(diff=round(float(d.mean()), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)])
out = {}
for f in range(5):
    a47 = LANE / f'arms/vitl47fu_elr2e5_f{f}_s47/raw/runs/fold{f}/dinov3_l16t/predictions_final_tta.npz'
    if not a47.exists(): continue
    p47 = LANE / f'arms/vitl16_f{f}_s47/raw/runs/fold{f}/dinov3_l16t/predictions_final_tta.npz'
    p46 = (X / f'lane_backbone_20261108/runs/vitl16_avg_f{f}_s46' if f in (0, 4) else LANE / f'arms/vitl16_f{f}_s46') / f'raw/runs/fold{f}/dinov3_l16t/predictions_final_tta.npz'
    a46 = LANE / (f'arms/sweep_vitl_elr2e5_f{f}_s46' if f in (0, 4) else f'arms/vitlfu_elr2e5_f{f}_s46') / f'raw/runs/fold{f}/dinov3_l16t/predictions_final_tta.npz'
    rows = fold == f; A47, P47, P46, A46 = v3(a47), v3(p47), v3(p46), v3(a46)
    out[f'fold{f}'] = dict(arm_s47=m3(A47, rows), plain_s47=m3(P47, rows), arm_s47_vs_plain_s47_cd=paired(A47, P47, rows),
                           arm_s46=m3(A46, rows), plain_s46=m3(P46, rows), arm_s46_vs_plain_s46_cd=paired(A46, P46, rows))
json.dump(out, open(LANE / 'reads/vitl47_members_20261114.json', 'w'), indent=1); print(json.dumps(out, indent=1))
