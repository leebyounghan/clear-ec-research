"""A member's sensitivity to label-preserving perturbations predicts the size of its CD error (r up to 0.42 for added noise). Can an uncertainty signal be spent (2026-11-04)? On the 500 rows, by sensitivity quintile:
residual bias and spread for B, A, P1; best scale and best A-P1 weight per quintile (fit on one half, read on the other, 40 splits). MAPE rewards a lower point when spread is high: p* = exp(mu - sigma^2)."""
import json, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
X = '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/zoom'; K = list(E.K)
z = np.load(f'{X}/perturb_f0.json.npz'); idx, L, y = z['idx'], z['L'], z['y']; V = ('id', 'blur1', 'blur2p5', 'gamma0p8', 'gamma1p25', 'noise', 'rot7', 'shift40')
b = E.load_base(); pos = {int(i): n for n, i in enumerate(b['idx'])}; R = np.array([pos[int(i)] for i in idx]); B, A, P1 = b['B'][R], b['A'][R], b['P1'][R]
D = {v: np.abs(L[:, k] - L[:, 0]) for k, v in enumerate(V) if v != 'id'}
sens = {nm: D[nm] for nm in ('noise', 'gamma0p8', 'shift40', 'rot7', 'blur1')}
sens['combo'] = np.mean([ (D[n] - D[n].mean(0)) / D[n].std(0) for n in ('noise', 'gamma0p8', 'shift40', 'rot7', 'blur1')], 0)
def ape(p, j, q): s = y[q, j] > 0; return np.abs(p[q, j][s] - y[q, j][s]) / y[q, j][s] * 100
rep = {}
for nm, S in sens.items():
    for j, k in enumerate(K):
        qs = np.quantile(S[:, j], (0.2, 0.4, 0.6, 0.8)); bin_ = np.digitize(S[:, j], qs); row = []
        for g in range(5):
            q = bin_ == g; s = y[q, j] > 0; rB = np.log(y[q, j][s]) - np.log(B[q, j][s]); row.append(dict(bias=round(float(np.median(rB)), 3), mad=round(float(1.4826 * np.median(np.abs(rB - np.median(rB)))), 3), B=round(float(ape(B, j, q).mean()), 2), A=round(float(ape(A, j, q).mean()), 2), P1=round(float(ape(P1, j, q).mean()), 2)))
        rep.setdefault(nm, {})[k] = row
        if nm in ('noise', 'combo'): print(nm, k, json.dumps(row), flush=True)
rng = np.random.default_rng(0); res = {}
for nm in ('noise', 'combo', 'gamma0p8', 'shift40'):
    S = sens[nm]; ds = {k: [] for k in K}; dw = {k: [] for k in K}
    for _ in range(60):
        pm = rng.permutation(len(y)); a, c = pm[:250], pm[250:]
        for j, k in enumerate(K):
            qs = np.quantile(S[a, j], (0.2, 0.4, 0.6, 0.8)); ba, bc = np.digitize(S[a, j], qs), np.digitize(S[c, j], qs); sc = np.ones(5); wt = np.zeros(5)
            for g in range(5):
                q = a[ba == g]; q = q[y[q, j] > 0]; grid = np.exp(np.linspace(-0.12, 0.12, 25)); er = [np.mean(np.abs(B[q, j] * v - y[q, j]) / y[q, j]) for v in grid]; sc[g] = grid[int(np.argmin(er))]
                gw = np.linspace(0, 1, 11); er = [np.mean(np.abs(A[q, j] ** w * P1[q, j] ** (1 - w) - y[q, j]) / y[q, j]) for w in gw]; wt[g] = gw[int(np.argmin(er))]
            tt = c[y[c, j] > 0]; bt = bc[y[c, j] > 0]; s0 = np.mean(np.abs(B[tt, j] - y[tt, j]) / y[tt, j]) * 100
            ds[k].append(np.mean(np.abs(B[tt, j] * sc[bt] - y[tt, j]) / y[tt, j]) * 100 - s0); dw[k].append(np.mean(np.abs(A[tt, j] ** wt[bt] * P1[tt, j] ** (1 - wt[bt]) - y[tt, j]) / y[tt, j]) * 100 - s0)
    res[nm] = dict(scale_by_bin={k: round(float(np.mean(v)), 3) for k, v in ds.items()}, weight_by_bin={k: round(float(np.mean(v)), 3) for k, v in dw.items()}); print(nm, json.dumps(res[nm]), flush=True)
json.dump(dict(bins=rep, spend=res), open(f'{X}/perturb_uncertainty.json', 'w'), indent=1)
