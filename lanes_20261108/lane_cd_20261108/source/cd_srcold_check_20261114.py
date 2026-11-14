"""Check of the coordinator's srcold read (02:47 EST): old-format specialist vs control P, fold 0, final-epoch four-flip TTA files."""
import json, re, numpy as np
from cd_common import *
import cd_read_20261108 as R
y3 = BP.y.astype(float); f0 = fold == 0
def vec3(p):
    z = np.load(p); v = np.full((len(yc), 3), np.nan); v[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, :3]; return v
S_ = vec3(LANE / 'arms/sweep_srcold_f0_s46/raw/runs/fold301/dinov3_b16/predictions_final_tta.npz'); P = vec3(LANE / 'arms/screen_plain_f0_s46/raw/runs/fold0/dinov3_b16/predictions_final_tta.npz')
new = np.array([bool(re.match(r'^\d{4}-\d{4}', str(i).strip())) for i in ids]); old = f0 & ~new
def m(p, j, rows): r = rows & (y3[:, j] > 0); return float(np.mean(np.abs(p[r, j] / y3[r, j] - 1)) * 100)
out = {'old_rows': int(old.sum())}
rng = np.random.default_rng(0); u, inv = np.unique(donors[old], return_inverse=True)
for j, k in enumerate(('CD', 'CV', 'HEX')):
    r = old & (y3[:, j] > 0); d = (np.abs(S_[r, j] / y3[r, j] - 1) - np.abs(P[r, j] / y3[r, j] - 1)) * 100; g = donors[r]; uu, ii = np.unique(g, return_inverse=True)
    s = np.bincount(ii, d); n = np.bincount(ii); bs = [s[q].sum() / n[q].sum() for q in (rng.integers(0, len(uu), len(uu)) for _ in range(2000))]
    out[k] = dict(spec=round(m(S_, j, old), 3), P=round(m(P, j, old), 3), diff=round(float(d.mean()), 3), ci95=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)])
out['blend_cd_old'] = {w: round(m(np.exp(w * np.log(S_) + (1 - w) * np.log(P)), 0, old), 3) for w in (0.0, 0.25, 0.5)}
lr = lambda p: np.log(p[:, 0] / np.where(yc > 0, yc, 1)); r = old & (yc > 0); out['resid_corr_old'] = round(float(np.corrcoef(lr(S_)[r], lr(P)[r])[0, 1]), 3)
out['new_rows_cd'] = dict(spec=round(m(S_, 0, f0 & new), 3), P=round(m(P, 0, f0 & new), 3))
print(json.dumps(out))
