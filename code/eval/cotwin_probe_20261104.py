"""Co-twin control probe (2026-11-04). Fellow eyes share donor, session and device, so the label difference between them isolates what is
specific to one eye's cells. Questions, on CPU with frozen label-free features:
  1. How much of the within-pair label difference does B explain, vs the between-pair part? Also the hex6 cell feature.
  2. Is there eye-specific signal in frozen SSL token statistics that plain training misses? Ridge fit on within-pair differences only
     (pairs with neither end in the held fold), read on held-out pairs and as a LOFO correction to B. Plain ridge on the same features is the control.
Pairs: new-format fellow eyes (certain); old-format adjacent IDs (about half fellow by CD, reported apart)."""
import json, os, re, sys
import numpy as np
import torch
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
OUT = '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/cotwin'; os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(int(min(12, max(2, 16 - os.getloadavg()[0] - 1))))
K = list(E.K); b = E.load_base(); ids = [re.sub(r'\s+', ' ', i.strip()) for i in b['ids']]; y, B, fold = b['y'], b['B'], b['fold']; N = len(ids)
ly = np.log(np.where(y > 0, y, np.nan)); lB = np.log(B); r = ly - lB
old, new = {}, {}
for n, i in enumerate(ids):
    m = re.fullmatch(r'(\d{4})-(\d{2})', i)
    if m: old[(int(m.group(2)), int(m.group(1)))] = n; continue
    m = re.match(r'(\d{4})-(\d{4})[ -]*([O0][DS])', i)
    if m: new[(int(m.group(1)), int(m.group(2)), m.group(3).replace('0', 'O'))] = n
NP = np.array([(new[k], new[(k[0], k[1], 'OS')]) for k in new if k[2] == 'OD' and (k[0], k[1], 'OS') in new])
OP = np.array([(old[k], old[(k[0], k[1] + 1)]) for k in old if (k[0], k[1] + 1) in old])
def within(P, P_lp, name):
    out = {}
    for j, k in enumerate(K):
        ok = np.isfinite(ly[P[:, 0], j]) & np.isfinite(ly[P[:, 1], j]); a, c = P[ok, 0], P[ok, 1]
        dy, dp = ly[a, j] - ly[c, j], P_lp[a, j] - P_lp[c, j]
        sy = ly[a, j] + ly[c, j]; sp = P_lp[a, j] + P_lp[c, j]
        be = float((dp * dy).sum() / (dp * dp).sum())
        out[k] = dict(n=int(ok.sum()), within_r=round(float(np.corrcoef(dy, dp)[0, 1]), 3), within_slope=round(be, 3), between_r=round(float(np.corrcoef(sy, sp)[0, 1]), 3),
                      sd_dy=round(float(dy.std()), 4), sd_dp=round(float(dp.std()), 4), sd_sy=round(float(sy.std() / 2), 4))
    print(name, json.dumps(out), flush=True); return out
rep = dict(within={}, probe={})
rep['within']['B_new'] = within(NP, lB, 'B new fellow'); rep['within']['B_old'] = within(OP, lB, 'B old adjacent')
z = np.load('/workspace/clear-ec-research/experiments/green_track_20261028/deploy/hex6_blend_params_12_oof_feature.npz', allow_pickle=True)
print('hex6 keys', {k: z[k].shape for k in z.files}, flush=True)
ms = np.load('/workspace/clear-ec-research/experiments/meta_ceiling_20261103/retrieval/ssl_tokens.npz'); pos = {int(i): n for n, i in enumerate(ms['idx'])}; o = np.array([pos[int(i)] for i in b['idx']])
X = torch.from_numpy(np.c_[ms['mean'][o], ms['std'][o]].astype(np.float32)); X = X - X.mean(0)
U, S, _ = torch.pca_lowrank(X, q=512, center=False, niter=4); P = (U * (N - 1) ** 0.5).numpy().astype(np.float64)
def ridge(A, t, lam): return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ t)
def lofo(pred):
    p = B.copy(); bs = []
    for f in range(5):
        tr = fold != f; te = fold == f
        for j in range(3):
            ok = tr & np.isfinite(r[:, j]); be = float((pred[ok, j] * r[ok, j]).sum() / max((pred[ok, j] ** 2).sum(), 1e-12)); be = max(be, 0.0); p[te, j] = B[te, j] * np.exp(be * pred[te, j]); bs.append(be)
    return p, bs
def score(p):
    a, c = E.official(E.ape(B, y)), E.official(E.ape(p, y)); return {k: round(c[k] - a[k], 4) for k in K + ['mean']}, [round(E.official(E.ape(p[fold == f], y[fold == f]))['mean'] - E.official(E.ape(B[fold == f], y[fold == f]))['mean'], 3) for f in range(5)]
for lam in (1e2, 1e3, 1e4, 1e5):
    for nm, pairs in dict(new=NP, new_old=np.r_[NP, OP]).items():
        pred_d = np.zeros((N, 3)); held = {k: [] for k in K}; pred_ly = np.zeros((N, 3)); pred_p = np.zeros((N, 3))
        for f in range(5):
            tr = pairs[(fold[pairs[:, 0]] != f) & (fold[pairs[:, 1]] != f)]; te = fold == f
            for j, k in enumerate(K):
                ok = np.isfinite(r[tr[:, 0], j]) & np.isfinite(r[tr[:, 1], j]); a, c = tr[ok, 0], tr[ok, 1]
                w = ridge(P[a] - P[c], r[a, j] - r[c, j], lam); pred_d[te, j] = P[te] @ w
                w2 = ridge(P[a] - P[c], ly[a, j] - ly[c, j], lam); pred_ly[te, j] = P[te] @ w2
                okp = (fold != f) & np.isfinite(r[:, j]); w3 = ridge(P[okp] - P[okp].mean(0), r[okp, j] - r[okp, j].mean(), lam); pred_p[te, j] = (P[te] - P[okp].mean(0)) @ w3
        heldN = NP[fold[NP[:, 0]] == fold[NP[:, 1]]]
        e = dict(diff_resid=score(lofo(pred_d)[0]), plain_resid=score(lofo(pred_p)[0]))
        d_w = {}
        for j, k in enumerate(K):
            ok = np.isfinite(ly[heldN[:, 0], j]) & np.isfinite(ly[heldN[:, 1], j]); a, c = heldN[ok, 0], heldN[ok, 1]
            d_w[k] = dict(r_diff_ridge_on_dy=round(float(np.corrcoef(ly[a, j] - ly[c, j], pred_ly[a, j] - pred_ly[c, j])[0, 1]), 3), r_diff_ridge_on_dr=round(float(np.corrcoef(r[a, j] - r[c, j], pred_d[a, j] - pred_d[c, j])[0, 1]), 3), r_plain_on_dr=round(float(np.corrcoef(r[a, j] - r[c, j], pred_p[a, j] - pred_p[c, j])[0, 1]), 3))
        e['within_heldout'] = d_w; rep['probe'][f'{nm}_lam{int(lam)}'] = e; print(nm, lam, json.dumps(e), flush=True)
json.dump(rep, open(f'{OUT}/cotwin_probe.json', 'w'), indent=1); print('done', flush=True)
