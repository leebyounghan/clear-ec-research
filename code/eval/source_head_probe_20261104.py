"""Does the feature-to-label mapping differ between the two sources (2026-11-04)? AUC 0.998: an image tells its source. Frozen label-free SSL statistics, PCA 512 whitened, ridge on log y, LOFO:
pooled head vs one head per source vs pooled + source offset. A probe of the head only: if per-source mapping beats pooled here, a source-conditioned head is worth a training run."""
import json, re, sys
import numpy as np
import torch
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
K = list(E.K); b = E.load_base(); y, fold, idx = b['y'], b['fold'], b['idx']; ids = [re.sub(r'\s+', ' ', i.strip()) for i in b['ids']]; N = len(ids)
fmt = np.array([0 if re.fullmatch(r'\d{4}-\d{2}', i) else 1 for i in ids]); ly = np.log(np.where(y > 0, y, np.nan))
ms = np.load('/workspace/clear-ec-research/experiments/meta_ceiling_20261103/retrieval/ssl_tokens.npz'); pos = {int(i): n for n, i in enumerate(ms['idx'])}; o = np.array([pos[int(i)] for i in idx])
X = torch.from_numpy(np.c_[ms['mean'][o], ms['std'][o]].astype(np.float32)); X = X - X.mean(0); U, S, _ = torch.pca_lowrank(X, q=512, center=False, niter=4); P = (U * (N - 1) ** 0.5).numpy().astype(np.float64)
def rg(A, t, lam):
    m = A.mean(0); tm = t.mean(); A0 = A - m; w = np.linalg.solve(A0.T @ A0 + lam * np.eye(A.shape[1]), A0.T @ (t - tm)); return lambda Z: (Z - m) @ w + tm
def run(mode, lam):
    p = np.full((N, 3), np.nan)
    for f in range(5):
        tr, te = fold != f, fold == f
        for j in range(3):
            ok = tr & np.isfinite(ly[:, j])
            if mode == 'pooled':
                p[te, j] = rg(P[ok], ly[ok, j], lam)(P[te])
            else:
                for g in (0, 1):
                    q = ok & (fmt == g); t = te & (fmt == g); p[t, j] = rg(P[q], ly[q, j], lam)(P[t])
    return np.exp(p)
for lam in (100.0, 1000.0, 10000.0):
    for mode in ('pooled', 'per_source'):
        p = run(mode, lam); out = {}
        for nm, s in (('all', np.ones(N, bool)), ('old', fmt == 0), ('new', fmt == 1)):
            e = E.official(E.ape(p[s], y[s])); out[nm] = {k: round(e[k], 3) for k in K + ['mean']}
        print(lam, mode, json.dumps(out), flush=True)
