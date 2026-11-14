"""Nested, leak-free teacher for target denoising (backbone lane, 2026-11-09). CPU only.

Rule (coordinator, hard): for outer fold f, the teacher value of a training row in fold g comes from a model that saw neither
fold f nor fold g. No image-model OOF satisfies that (every fold-g model saw fold f), so the teacher uses label-free features only:
detector-window cell statistics (legible_window.npz, 49), detector hex6 / detection count / dark fraction (green_features.npz),
image statistics (image_stats.npz). Model: sklearn HistGradientBoostingRegressor per target on log labels, fit on the three folds
outside {f, g}, predicting fold g. Rows are those of the split files the runner trains on; their fold g comes from the split files.
Writes data/teacher_nested_f{f}_20261109.npz {indices, pred} and data/rowweight_nested_f{f}_20261109.npz {indices, Z} (weights:
per-row discrepancy d = mean_k |log y - log t| / median_k; w = 1 up to the 90th percentile, then (q90 / d)^2, floor 0.2).
Also reports the teacher's own quality: nested MAPE on the training rows and plain 5-fold OOF MAPE."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

X = Path('/workspace/clear-ec-research/experiments'); R = Path('/workspace/clear-ec-research'); OUT = X / 'lane_backbone_20261108/data'
SPL = {0: 'experiments/ssl_screen_20260911/preflight/fold0.npz', 1: 'experiments/ssl_screen_20260911/preflight/fold1.npz',
       2: 'experiments/overnight_target_20260912/preflight/fold2.npz', 3: 'experiments/overnight_target_20260912/preflight/fold3.npz',
       4: 'experiments/target_fold4_20260912/preflight/fold4.npz'}
with np.load(R / 'datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True) as z:
    Y = z['Y'].astype(np.float64)
N = len(Y)
fold_of = np.full(N, -1)
for f, p in SPL.items():
    with np.load(R / p) as z:
        fold_of[z['validation'].astype(int)] = f
# features
lw = np.load(X / 'meta_ceiling_20261103/sweep/legible_window.npz', allow_pickle=True)['F'].astype(np.float64)
g = np.load(X / 'submission2_search_20261103/green_features.npz', allow_pickle=True)
gi = g['indices'].astype(int); G = np.full((N, 3), np.nan)
G[gi, 0] = np.where(g['f_active10_valid'], g['f_active10'], np.nan); G[gi, 1] = np.where(g['n_det_valid'], g['n_det'], np.nan); G[gi, 2] = g['black_frac']
s = np.load(X / 'eval_harness_20261031/image_stats.npz', allow_pickle=True); si = s['idx'].astype(int); ST = np.full((N, 4), np.nan)
for j, k in enumerate(('mean', 'contrast', 'black_frac', 'focus')):
    ST[si, j] = s[k]
F = np.concatenate([lw, G, ST], 1)
F = F[:, [j for j in range(F.shape[1]) if len(np.unique(F[np.isfinite(F[:, j]), j])) > 1]]  # drop constant / all-missing columns
ok = np.isfinite(Y).all(1) & (Y > 0).all(1) & (fold_of >= 0)
LY = np.log(np.where(Y > 0, Y, np.nan))
print(json.dumps(dict(features=F.shape[1], rows_with_fold=int((fold_of >= 0).sum()), usable=int(ok.sum()))), flush=True)

def fit_predict(train_rows, pred_rows):
    out = np.zeros((len(pred_rows), 3))
    for k in range(3):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=1109)
        m.fit(F[train_rows], LY[train_rows, k]); out[:, k] = m.predict(F[pred_rows])
    return np.exp(out)

mape = lambda p, y: (np.abs(p / y - 1) * 100).mean(0)
# plain 5-fold OOF (reference for teacher quality only, never used for training)
oof = np.full((N, 3), np.nan)
for f in range(5):
    oof[ok & (fold_of == f)] = fit_predict(np.where(ok & (fold_of != f))[0], np.where(ok & (fold_of == f))[0])
m = mape(oof[ok], Y[ok]); print(json.dumps(dict(teacher_plain_oof_mape=dict(zip(['CD', 'CV', 'HEX'], m.round(3).tolist())), mean=round(float(m.mean()), 3))), flush=True)
for f in [int(v) for v in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['0', '4'])]:
    with np.load(R / SPL[f]) as z:
        tr = z['train'].astype(int)
    assert (fold_of[tr] != f).all() and (fold_of[tr] >= 0).all(), 'every training row must sit in another fold'
    T = np.full((N, 3), np.nan)
    for gf in sorted(set(range(5)) - {f}):
        rows = tr[fold_of[tr] == gf]; fit_rows = np.where(ok & (fold_of != f) & (fold_of != gf))[0]
        assert not np.isin(fit_rows, np.where(fold_of == f)[0]).any() and not np.isin(fit_rows, rows).any()
        T[rows] = fit_predict(fit_rows, rows)
    assert np.isfinite(T[tr]).all() and (T[tr] > 0).all()
    d_k = np.abs(np.log(Y[tr]) - np.log(T[tr])); d = (d_k / np.median(d_k, 0)).mean(1); q = np.quantile(d, 0.9)
    w = np.where(d <= q, 1.0, np.maximum(0.2, (q / d) ** 2))
    np.savez_compressed(OUT / f'teacher_nested_f{f}_20261109.npz', indices=tr, pred=T[tr].astype(np.float32))
    np.savez_compressed(OUT / f'rowweight_nested_f{f}_20261109.npz', indices=tr, Z=w.astype(np.float32))
    mt = mape(T[tr], Y[tr])
    print(json.dumps(dict(outer_fold=f, train_rows=len(tr), nested_teacher_mape_on_train=dict(zip(['CD', 'CV', 'HEX'], mt.round(3).tolist())),
                          weight_mean=round(float(w.mean()), 4), share_below_1=round(float((w < 1).mean()), 4), share_at_floor=round(float((w <= 0.2001).mean()), 4))), flush=True)
