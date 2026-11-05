"""LOFO read of all tessellation features together against B's log residual (2026-11-04), on the rows finished so far. Ridge and gradient boosting (L1) per metric,
p' = B exp(beta pred), beta from the training folds, floored at 0. Next to the same stack on the hex6 detector feature, and on both. Row bootstrap CI on the delta."""
import json, os, sys, warnings
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
K = list(E.K); b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']
z = np.load(sys.argv[1]); names = [str(n) for n in z['names']]; F = z['F'][idx].astype(float); seen = z['seen'][idx]
h6 = np.load('/workspace/clear-ec-research/experiments/green_track_20261028/deploy/hex6_blend_params_12_oof_feature.npz'); hx = h6['feature'][idx].astype(float); print('hex6 raw range', np.nanmin(hx), np.nanmax(hx), 'nan', int(np.isnan(hx).sum()))
ly = np.log(np.where(y > 0, y, np.nan)); lB = np.log(B); r = ly - lB
rows = np.where(seen)[0]; print('rows', len(rows))
def tf(X):
    X = X.copy(); X[X <= 0] = np.nan; X = np.log(X); med = np.nanmedian(X, 0); bad = ~np.isfinite(X); X[bad] = np.take(med, np.nonzero(bad)[1]); return (X - X.mean(0)) / (X.std(0) + 1e-6)
Xt = tf(F[rows]); Xh = np.nan_to_num(hx[rows], nan=np.nanmedian(hx[rows]))[:, None]; Xh = (Xh - Xh.mean()) / Xh.std()
for j, k in enumerate(K):
    s = np.isfinite(r[rows, j]); print(k, 'hex6 r_y', round(float(np.corrcoef(Xh[s, 0], ly[rows[s], j])[0, 1]), 3), 'r_resid', round(float(np.corrcoef(Xh[s, 0], r[rows[s], j])[0, 1]), 3))
sets = dict(tess=Xt, hex6=Xh, both=np.c_[Xt, Xh])
def fit_pred(X, kind):
    out = np.zeros((len(rows), 3)); bs = []
    for f in range(5):
        tr = fold[rows] != f; te = ~tr
        for j in range(3):
            ok = tr & np.isfinite(r[rows, j]); t = r[rows[ok], j]
            if kind == 'ridge':
                A = X[ok]; lam = 300.0; w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (t - t.mean())); pr, pe = A @ w + t.mean(), X[te] @ w + t.mean()
            else:
                m = HGB(loss='absolute_error', max_depth=3, max_iter=150, learning_rate=0.03, min_samples_leaf=40, l2_regularization=5.0, random_state=0); m.fit(X[ok], t); pr, pe = m.predict(X[ok]), m.predict(X[te])
            out[te, j] = pe; bs.append(0.0)
    return out
def score(pr):
    pp = np.log(B[rows]) + pr; res = {}; tot = 0; alld = []
    for j, k in enumerate(K):
        s = y[rows, j] > 0; d = (np.abs(np.exp(pp[s, j]) - y[rows[s], j]) - np.abs(B[rows[s], j] - y[rows[s], j])) / y[rows[s], j] * 100; res[k] = round(float(d.mean()), 3)
        g = np.random.default_rng(j).integers(0, len(d), (800, len(d))); alld.append(d[g].mean(1))
    m = np.mean(alld, 0); res['mean'] = round(float(np.mean([res[k] for k in K])), 3); res['ci'] = [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]
    F_ = []
    for f in range(5):
        v = []
        for j in range(3):
            q = (fold[rows] == f) & (y[rows, j] > 0); t = y[rows[q], j]; v.append(float(((np.abs(np.exp(pp[q, j]) - t) - np.abs(B[rows[q], j] - t)) / t * 100).mean()))
        F_.append(round(float(np.mean(v)), 3))
    res['folds'] = F_
    return res
rep = {}
for nm, X in sets.items():
    for kind in ('ridge', 'gbm'):
        rep[f'{nm}_{kind}'] = score(fit_pred(X, kind)); print(nm, kind, json.dumps(rep[f'{nm}_{kind}']), flush=True)
json.dump(rep, open(sys.argv[1] + '.stack.json', 'w'), indent=1)
