"""Audit A1, part 3: extreme rows. How much of the score do they hold, are they label errors, and what would removing them do? (lane GPU 0, 2026-11-06)"""
import re, sys
import numpy as np
sys.path.insert(0, 'experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
E = BP.E
ids, y, P, fold, donors = BP.ids, BP.y, BP.REF_BEST, BP.fold, BP.donors
err = E.ape(P, y); K = ['CD', 'CV', 'HEX']
print('## share of each metric error in the worst rows')
for j, k in enumerate(K):
    e = np.sort(err[:, j][~np.isnan(err[:, j])])[::-1]; c = np.cumsum(e) / e.sum()
    print(k, {f'top {n}': round(float(c[n - 1]), 4) for n in (1, 5, 10, 20, 50, 90, 450, 900)}, 'mean with top 10 removed', round(float(e[10:].mean()), 3), 'with top 50 removed', round(float(e[50:].mean()), 3), 'all', round(float(e.mean()), 3), 'worst', e[:6].round(1).tolist())
print('\n## worst 25 rows per metric: id, fold, label, prediction, APE, the frame other-metric APE, same-donor rows labels')
for j, k in enumerate(K):
    o = np.argsort(-np.nan_to_num(err[:, j]))[:25]
    print(k)
    for i in o:
        d = np.flatnonzero((donors == donors[i]) & (np.arange(len(y)) != i))
        print(f'  {ids[i]:<22} f{fold[i]} label {y[i, 0]:6.0f} {y[i, 1]:.3f} {y[i, 2]:.3f} pred {P[i, 0]:6.0f} {P[i, 1]:.3f} {P[i, 2]:.3f} ape {err[i, 0]:6.1f} {err[i, 1]:6.1f} {err[i, 2]:6.1f}', 'donor', [(round(float(y[q, 0])), round(float(y[q, 1]), 2), round(float(y[q, 2]), 2)) for q in d[:2]])
print('\n## label tails')
for j, k in enumerate(K):
    v = y[:, j]; print(k, 'min', np.sort(v)[:12].round(3).tolist(), 'max', np.sort(v)[-12:].round(3).tolist())
print('\n## fellow-eye and neighbour consistency for extreme labels; how many frames have extreme residual in log, by threshold')
res = np.abs(np.log(P) - np.log(np.where(y > 0, y, np.nan)))
for t in (0.3, 0.4, 0.5, 0.7, 1.0):
    print('|log res| >', t, {k: int((res[:, j] > t).sum()) for j, k in enumerate(K)}, 'any', int((res > t).any(1).sum()))
print('\n## test share: would these rows matter on 1000 test rows? Mean with worst rows of each metric trimmed at APE thresholds')
for t in (50, 100, 200, 400):
    print('APE clipped at', t, 'mean', round(float(np.nanmean(np.minimum(err, t), 0).mean()), 3), {k: round(float(np.nanmean(np.minimum(err[:, j], t))), 3) for j, k in enumerate(K)})
