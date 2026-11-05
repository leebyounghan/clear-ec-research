"""Rows whose label disagrees most with the out-of-fold hybrid B (2026-11-04). Score = max over metrics of |log(y / B)| divided
by that metric's MAD-scaled residual spread, so the three metrics count equally. The top 3 % of all rows are written as the drop
list used by recipeA_label_clean_20261104.py. A train row's B prediction comes from models that saw the validation fold's
labels, a second-order leak accepted here. Also prints who gets dropped, since this removes real pathology as well as bad labels."""
import json, os, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
OUT = '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/label_clean'
FRAC = 0.03
b = E.load_base(); y, B = b['y'], b['B']
r = np.log(np.where(y > 0, y, np.nan)) - np.log(B)
s = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r, 0)), 0)
z = np.nanmax(np.abs(r) / s, 1)
cut = np.quantile(z, 1 - FRAC); drop = z >= cut
os.makedirs(OUT, exist_ok=True)
np.savez(f'{OUT}/drop_rows.npz', indices=b['idx'][drop], score=z[drop], cut=cut, frac=FRAC, scale=s)
st = E.image_stats(b); e = E.ape(B, y)
rep = dict(n=int(drop.sum()), cut=round(float(cut), 3), scale=dict(zip(E.K, map(float, s))),
           per_fold={int(f): int((drop & (b['fold'] == f)).sum()) for f in range(5)},
           label_median_dropped=dict(zip(E.K, np.nanmedian(y[drop], 0).tolist())), label_median_all=dict(zip(E.K, np.nanmedian(y, 0).tolist())),
           cd_lt1500_dropped=int((y[drop, 0] < 1500).sum()), cd_lt1500_all=int(((y[:, 0] > 0) & (y[:, 0] < 1500)).sum()),
           framed_dropped=round(float((st['black_frac'][drop] > 0.01).mean()), 3), framed_all=round(float((st['black_frac'] > 0.01).mean()), 3),
           b_score_all=E.official(e)['mean'], b_score_without_dropped=E.official(e[~drop])['mean'],
           share_flagged_by=dict(zip(E.K, ((np.abs(r) / s)[drop].argmax(1)[None] == np.arange(3)[:, None]).mean(1).tolist())))
json.dump(rep, open(f'{OUT}/drop_summary.json', 'w'), indent=1); print(json.dumps(rep))
