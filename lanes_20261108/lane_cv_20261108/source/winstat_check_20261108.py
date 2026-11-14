"""CV lane: r of per-frame window statistics with log CV label and with the best pipeline's log CV residual, within each source too."""
import sys, re
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
y = BP.y; idx = BP.idx.astype(int); ids = BP.ids; P = BP.REF_BEST; ly = np.log(np.where(y[:, 1] > 0, y[:, 1], np.nan)); rs = ly - np.log(P[:, 1])
src = np.array([0 if re.match(r'^\d{4}-\d{2}$', re.sub(r'\s+', ' ', i.strip())) else 1 for i in ids])
for nm in sys.argv[1:]:
    z = np.load(f'/workspace/clear-ec-research/experiments/lane_cv_20261108/graphs/{nm}.npz'); S = z['win_stats'][idx]; names = list(z['win_stat_names']); o = []
    for j, k in enumerate(names):
        x = np.log(np.maximum(S[:, j], 1e-4)); m = np.isfinite(x) & np.isfinite(ly) & BP.good
        rA = np.corrcoef(x[m & (src == 0)], ly[m & (src == 0)])[0, 1]; rB = np.corrcoef(x[m & (src == 1)], ly[m & (src == 1)])[0, 1]
        o.append(f'{k}: r {np.corrcoef(x[m], ly[m])[0, 1]:.3f} (A {rA:.3f}, B {rB:.3f}) resid {np.corrcoef(x[m], rs[m])[0, 1]:.3f}')
    print(nm, 'n', int(m.sum()), ' | '.join(o), flush=True)
