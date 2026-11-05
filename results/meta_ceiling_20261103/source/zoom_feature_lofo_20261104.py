"""Zoom response as a feature, LOFO on the folds with views (2026-11-04). Features from the seed-46 member, its own validation rows: slope (L(1.25) - L(0.8)) / log(1.25 / 0.8) and curvature L(1.25) + L(0.8) - 2 L(1) per metric.
Pipeline as deployed: B -> hex6 -> correction -> stretch, all fit on the other folds. Correction: L1 fit (IRLS) of the log residual on CD [g_CD, c_CD], CV [g_CV, c_CV, g_CD], HEX [g_HEX, c_HEX, g_CD].
Reference: the same pipeline without the correction. Gate written before reading: mean <= -0.02 on all folds, CI upper bound < 0, at least 4 of 5 folds negative."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import stretch_nested_20261103 as N
E = C.E; K = list(C.K)
A_, B_, W_ = -0.2084, 0.5989, 0.28
Z = Path('/workspace/clear-ec-research/experiments/meta_ceiling_20261103/zoom')
b = E.load_base(); y, B, fold = b['y'], b['B'], b['fold']; pos = {int(i): k for k, i in enumerate(b['idx'])}; N_ = len(y)
gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); have = gf['f_active10_valid'] & ~gf['seen_leaked'] & np.isfinite(gf['f_active10']); good = ~gf['seen_leaked']
H = B.copy(); H[have, 2] = np.exp((1 - W_) * np.log(B[have, 2]) + W_ * (A_ + B_ * gf['f_active10'][have]))
G = np.full((N_, 3), np.nan); Cv = np.full((N_, 3), np.nan); folds = []
for f in range(5):
    p = Z / f'views_f{f}.npz'
    if not p.exists():
        continue
    z = np.load(p); r = np.array([pos[int(i)] for i in z['indices']]); L = z['L']; G[r] = (L[:, 1] - L[:, 0]) / np.log(1.25 / 0.8); Cv[r] = L[:, 1] + L[:, 0] - 2 * z['s1']; folds.append(f); assert (fold[r] == f).all()
print('folds', folds, 'rows', int(np.isfinite(G[:, 0]).sum()), 'median slope', [round(float(np.nanmedian(G[:, j])), 3) for j in range(3)], flush=True)
rows = np.isin(fold, folds) & np.isfinite(G).all(1)
med_g, med_c = np.nanmedian(G, 0), np.nanmedian(Cv, 0)
def feats(j):
    x = [G[:, j] - med_g[j], Cv[:, j] - med_c[j]]
    if j > 0: x.append(G[:, 0] - med_g[0])
    return np.c_[x].T
def irls(X, t, it=20):
    A = np.c_[np.ones(len(X)), X]; w = np.linalg.lstsq(A, t, rcond=None)[0]
    for _ in range(it):
        d = 1 / np.maximum(np.abs(t - A @ w), 1e-3); w = np.linalg.solve((A * d[:, None]).T @ A + 1e-6 * np.eye(A.shape[1]), (A * d[:, None]).T @ t)
    return w
def run(base, use):
    P = np.full_like(base, np.nan); Q = base.copy(); W = {}
    for f in folds:
        tr = rows & (fold != f) & good; te = rows & (fold == f); C_ = base.copy()
        if use:
            for j in use:
                X = feats(j); ok = tr & (y[:, j] > 0); w = irls(X[ok], np.log(y[ok, j]) - np.log(base[ok, j])); C_[rows, j] = base[rows, j] * np.exp(np.c_[np.ones(rows.sum()), X[rows]] @ w); W[f'{f}{K[j]}'] = [round(float(v), 4) for v in w]
        s = N.fit_stretch(C_, y, tr); P[te] = N.apply_stretch(C_[te], s)
    return P, W
g = E.donors(b['ids'])
def report(name, p, ref):
    pb = E.paired_bootstrap(E.ape(p[rows], y[rows]), E.ape(ref[rows], y[rows]), np.unique(g[rows], return_inverse=True)[1]); s = C.mape(p[rows], y[rows]); s0 = C.mape(ref[rows], y[rows])
    per = [round(C.mape(p[rows & (fold == f)], y[rows & (fold == f)])['mean'] - C.mape(ref[rows & (fold == f)], y[rows & (fold == f)])['mean'], 3) for f in folds]
    out = dict(score={k: round(s[k], 3) for k in K + ['mean']}, per_metric={k: round(s[k] - s0[k], 3) for k in K}, delta=round(pb['delta'], 4), ci95=[round(v, 4) for v in pb['ci95']], folds=per); print(name, json.dumps(out), flush=True); return out
rep = dict(folds=folds)
ref, _ = run(H, None); rep['B_hex6_stretch_vs_B'] = report('B + hex6 + stretch vs B', ref, B)
for nm, use in dict(cd_only=[0], cd_cv=[0, 1], all3=[0, 1, 2]).items():
    p, w = run(H, use); rep[nm] = report(f'+ zoom {nm} vs B + hex6 + stretch', p, ref); rep[nm]['weights'] = w
rB, _ = run(B, None); pB, _ = run(B, [0]); rep['on_B_cd_only'] = report('zoom CD on B + stretch, no hex6', pB, rB)
json.dump(rep, open(Z / f'feature_lofo_{len(folds)}folds.json', 'w'), indent=1); print('done', flush=True)
