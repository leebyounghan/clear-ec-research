"""CV lane (2026-11-09): Konan weighted-bisector CV (k=2) from detector peaks as a CV correction on top of the one-recipe lane best (out/stack5_best_3seeds.npz).
Nested ridge on the log CV residual (fit on folds != f), rows with >= MINC interior cells; features: log cv_k2, log cv_k0, hex_k2, log n_cells, log stack CV. Scale refit
MAPE-optimally on the training rows. usage: konan_corr_20261109.py FEATFILE [ALPHA] [MINC] [STACK]"""
import sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import ceiling_meta_cpu_20261103 as C
E = BP.E; y = BP.y; fold = BP.fold; idx = BP.idx.astype(int); good = BP.good
D = '/workspace/clear-ec-research/experiments/lane_cv_20261108/out/'
F = np.load(D + sys.argv[1])['F'][idx]; ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0; MINC = int(sys.argv[3]) if len(sys.argv) > 3 else 50
P0 = np.load(D + (sys.argv[4] if len(sys.argv) > 4 else 'stack5_best_3seeds.npz'))['P']
ok = y[:, 1] > 0; ly = np.log(np.where(ok, y[:, 1], 1.0)); res = ly - np.log(P0[:, 1]); m = np.isfinite(F).all(1) & (np.nan_to_num(F[:, 4]) >= MINC); mm = m & ok
lg = lambda a: np.log(np.maximum(a, 1e-4))
for j, nm in ((0, 'cv_k0'), (1, 'cv_k2')):
    print('%s: r with log CV label %.3f | with stack residual %.3f (rows %d)' % (nm, np.corrcoef(lg(F[mm, j]), ly[mm])[0, 1], np.corrcoef(lg(F[mm, j]), res[mm])[0, 1], int(mm.sum())), flush=True)
X = np.c_[lg(F[:, 1]), lg(F[:, 0]), F[:, 3], np.log(np.nan_to_num(F[:, 4], nan=1) + 1), np.log(P0[:, 1])]; X = np.where(m[:, None], X, 0.0); P = P0.copy()
for f in range(5):
    tr = (fold != f) & good & mm; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Z = (X - mu) / sd; A = Z[tr]
    w = np.linalg.solve(A.T @ A + ALPHA * np.eye(A.shape[1]), A.T @ (res[tr] - res[tr].mean())); q = P0[:, 1] * np.exp(Z @ w)
    c = C.wmed(y[tr, 1] / q[tr], q[tr] / y[tr, 1]); c0 = C.wmed(y[tr, 1] / P0[tr, 1], P0[tr, 1] / y[tr, 1]); sel = (fold == f) & m; P[sel, 1] = q[sel] * c / c0
a, a0, ar = E.ape(P, y)[:, 1], E.ape(P0, y)[:, 1], E.ape(BP.REF_BEST, y)[:, 1]
print('KONAN CV correction (alpha %g, >=%d cells): CV %.4f vs stack %.4f (d %.4f) vs best pipeline %.4f (d %.4f) | per fold vs stack %s' % (ALPHA, MINC, np.nanmean(a), np.nanmean(a0), np.nanmean(a) - np.nanmean(a0), np.nanmean(ar), np.nanmean(a) - np.nanmean(ar), [round(float(np.nanmean(a[fold == f]) - np.nanmean(a0[fold == f])), 4) for f in range(5)]), flush=True)
np.savez_compressed(D + 'stack_konan_cv.npz', P=P)
