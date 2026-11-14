"""CV lane: does the HEX information the pipeline gained (hex6 + graph) predict the CV residual? Nested 5-fold linear correction of log CV on
features known at test: d log HEX (BEST vs hybrid B), log HEX of BEST, hex6 (f_active10), log CD of BEST, d log CV already applied, and products with source format.
Fit on the training folds' OOF rows (ridge, small grid of alpha chosen inside), applied to the held-out fold."""
import sys, re
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
E = BP.E; y = BP.y; fold = BP.fold; P0 = BP.REF_BEST; B = BP.B; ids = BP.ids
src = np.array([0.0 if re.match(r'^\d{4}-\d{2}$', re.sub(r'\s+', ' ', i.strip())) else 1.0 for i in ids])
f6 = np.nan_to_num(BP.f6, nan=np.nanmedian(BP.f6))
F = np.c_[np.log(P0[:, 2]) - np.log(B[:, 2]), np.log(P0[:, 2]), f6, np.log(P0[:, 0]), np.log(P0[:, 1]) - np.log(B[:, 1]), src]
F = np.c_[F, F[:, :5] * src[:, None]]
ly = np.log(np.where(y[:, 1] > 0, y[:, 1], np.nan)); res = ly - np.log(P0[:, 1])
def ridge(Xtr, ytr, a):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Z = (Xtr - mu) / sd; w = np.linalg.solve(Z.T @ Z + a * np.eye(Z.shape[1]), Z.T @ (ytr - ytr.mean())); return lambda X: ((X - mu) / sd) @ w + ytr.mean()
P = P0.copy(); good = BP.good
for f in range(5):
    tr = (fold != f) & good & np.isfinite(res); best = None
    for a in (1, 10, 100, 1000, 10000):  # alpha by inner folds
        sc = []
        for g in [q for q in range(5) if q != f]:
            t2 = tr & (fold != g); v2 = tr & (fold == g); m = ridge(F[t2], res[t2], a); p = P0[v2, 1] * np.exp(m(F[v2])); sc.append(np.mean(np.abs(p - y[v2, 1]) / y[v2, 1]))
        if best is None or np.mean(sc) < best[0]: best = (np.mean(sc), a)
    m = ridge(F[tr], res[tr], best[1]); P[fold == f, 1] = P0[fold == f, 1] * np.exp(m(F[fold == f])); print('fold', f, 'alpha', best[1], flush=True)
a, a0 = E.ape(P, y)[:, 1], E.ape(P0, y)[:, 1]
print('CV from HEX/CD info, nested: %.4f vs %.4f d %.4f | folds %s' % (np.nanmean(a), np.nanmean(a0), np.nanmean(a) - np.nanmean(a0), [round(float(np.nanmean(a[fold == f]) - np.nanmean(a0[fold == f])), 4) for f in range(5)]))
print('corr of residual with features (all rows):', [round(float(np.corrcoef(F[np.isfinite(res), j], res[np.isfinite(res)])[0, 1]), 3) for j in range(6)])
