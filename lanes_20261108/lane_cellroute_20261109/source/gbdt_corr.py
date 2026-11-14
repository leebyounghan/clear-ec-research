"""Cell-route lane (2026-11-09): a different model on the same inputs. Histogram gradient boosting (sklearn, one fixed setting: L1 loss on the log
residual, lr 0.05, 300 iterations, 15 leaves, min 100 rows per leaf, l2 1.0, no early stopping) on every cell-route column (cellroute_feats, trim_feats,
click_feats; NaN kept native) plus the log lane-best prediction, predicting the log residual of the CV / HEX lane best. Nested: fit on folds != f, applied
to fold f; MAPE-optimal scale refit on the training rows (as corr.py). Run on clear-ec-cpu-1109 with OMP_NUM_THREADS=16."""
import sys, json
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import ceiling_meta_cpu_20261103 as C
ROOT = '/workspace/clear-ec-research/experiments/'; LANE = ROOT + 'lane_cellroute_20261109/'
E = BP.E; y = BP.y; fold = BP.fold; idx = BP.idx.astype(int); good = BP.good
Fs = [np.load(LANE + f'feats/{f}.npz')['F'][idx] for f in ('cellroute_feats', 'trim_feats', 'click_feats_P2_P5_m1_K8')]; X0 = np.column_stack(Fs)
BASE = BP.REF_BEST.copy(); BASE[:, 1] = np.load(ROOT + 'lane_cv_20261108/out/stack5_best_3seeds.npz')['P'][:, 1]; BASE[:, 2] = np.load(ROOT + 'lane_hex_20261108/preds/chain_cvw3.npz')['final_hex']
out = {}
for j, nm in ((2, 'HEX'), (1, 'CV')):
    X = np.c_[X0, np.log(BASE[:, j])]; ok = y[:, j] > 0; res = np.log(np.where(ok, y[:, j], 1.0)) - np.log(BASE[:, j]); P = BASE.copy()
    for f in range(5):
        tr = (fold != f) & good & ok
        m = HistGradientBoostingRegressor(loss='absolute_error', learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, early_stopping=False, random_state=0)
        m.fit(X[tr], res[tr]); q = BASE[:, j] * np.exp(m.predict(X))
        c = C.wmed(y[tr, j] / q[tr], q[tr] / y[tr, j]); c0 = C.wmed(y[tr, j] / BASE[tr, j], BASE[tr, j] / y[tr, j]); sel = fold == f; P[sel, j] = q[sel] * c / c0
    a, al = E.ape(P, y)[:, j], E.ape(BASE, y)[:, j]; f04 = (fold == 0) | (fold == 4)
    o = dict(value=round(float(np.nanmean(a)), 4), lane_best=round(float(np.nanmean(al)), 4), d_lane_best=round(float(np.nanmean(a) - np.nanmean(al)), 4),
             folds_vs_lane=[round(float(np.nanmean(a[fold == f]) - np.nanmean(al[fold == f])), 4) for f in range(5)], f04_vs_lane=round(float(np.nanmean(a[f04]) - np.nanmean(al[f04])), 4), n_features=int(X.shape[1]))
    out[nm] = o; print(nm, json.dumps(o), flush=True); np.savez_compressed(LANE + f'out/gbdt_{nm.lower()}.npz', P=P)
json.dump(out, open(LANE + 'out/gbdt_corr.json', 'w'), indent=1)
