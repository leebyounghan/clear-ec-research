"""Two ID formats suggest two sources (2026-11-04). Do they differ in how B errs, and can an image tell which it is? Per format: signed log residual, residual spread, slope of log y on log B, error.
Oracle: a separate log-linear calibration per format (fit on other folds). Then a logistic classifier from label-free SSL token statistics (LOFO), and the same calibration routed by its prediction.
Year x style offsets were 1-3 % of MSE (H4), but format was never a group there."""
import json, re, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
K = list(E.K); b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']; ids = [re.sub(r'\s+', ' ', i.strip()) for i in b['ids']]; N = len(ids)
fmt = np.array([0 if re.fullmatch(r'\d{4}-\d{2}', i) else 1 for i in ids]); yr = np.array([2000 + int(i[-2:]) if f == 0 else int(i[:4]) for i, f in zip(ids, fmt)])
ly = np.log(np.where(y > 0, y, np.nan)); lB = np.log(B); r = ly - lB
for f, nm in ((0, 'old'), (1, 'new')):
    s = fmt == f; o = {}
    for j, k in enumerate(K):
        q = s & np.isfinite(r[:, j]); o[k] = dict(mean_r=round(float(r[q, j].mean()), 4), median_r=round(float(np.median(r[q, j])), 4), sd_r=round(float(r[q, j].std()), 4), mad=round(float(1.4826 * np.median(np.abs(r[q, j] - np.median(r[q, j])))), 4), slope=round(float(np.polyfit(lB[q, j], ly[q, j], 1)[0]), 3), label_sd=round(float(ly[q, j].std()), 3))
    e = E.official(E.ape(B[s], y[s])); print(nm, 'n', int(s.sum()), {k: round(e[k], 3) for k in K + ['mean']}, json.dumps(o))
for yy in range(2018, 2025):
    a = (fmt == 0) & (yr == yy); c = (fmt == 1) & (yr == yy); print(yy, 'old', int(a.sum()), round(E.official(E.ape(B[a], y[a]))['mean'], 3) if a.sum() > 20 else None, 'new', int(c.sum()), round(E.official(E.ape(B[c], y[c]))['mean'], 3) if c.sum() > 20 else None)
def calib(route):
    p = B.copy()
    for fo in range(5):
        tr, te = fold != fo, fold == fo
        for j in range(3):
            for g in (0, 1):
                q = tr & (fmt == g) & np.isfinite(r[:, j]); A = np.polyfit(lB[q, j], ly[q, j], 1); t = te & (route == g); p[t, j] = np.exp(A[0] * lB[t, j] + A[1])
    return p
def glob():
    p = B.copy()
    for fo in range(5):
        tr, te = fold != fo, fold == fo
        for j in range(3):
            q = tr & np.isfinite(r[:, j]); A = np.polyfit(lB[q, j], ly[q, j], 1); p[te, j] = np.exp(A[0] * lB[te, j] + A[1])
    return p
base = E.official(E.ape(B, y)); d = lambda p: {k: round(E.official(E.ape(p, y))[k] - base[k], 4) for k in K + ['mean']}
print('global stretch by L2 fit', d(glob())); print('per-format, true format (oracle)', d(calib(fmt)))
ms = np.load('/workspace/clear-ec-research/experiments/meta_ceiling_20261103/retrieval/ssl_tokens.npz'); pos = {int(i): n for n, i in enumerate(ms['idx'])}; o = np.array([pos[int(i)] for i in idx]); X = np.c_[ms['mean'][o], ms['std'][o]].astype(np.float64); X = (X - X.mean(0)) / X.std(0)
pr = np.zeros(N)
for fo in range(5):
    tr = fold != fo; m = LogisticRegression(C=0.01, max_iter=300).fit(X[tr], fmt[tr]); pr[~tr] = m.predict_proba(X[~tr])[:, 1]
from sklearn.metrics import roc_auc_score
print('format classifier LOFO AUC', round(float(roc_auc_score(fmt, pr)), 4), 'accuracy', round(float(((pr > 0.5) == fmt).mean()), 4)); print('per-format, predicted format', d(calib((pr > 0.5).astype(int))))
