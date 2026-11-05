"""Read the tessellation features against B on the rows finished so far (2026-11-04). Pearson with log y and partial with B's log residual, next to the
hex6 detector feature; then a LOFO log blend as for hex6, p' = exp((1 - w) log B + w (a + b log f)), w on the training folds. Row bootstrap CI on the mean delta."""
import json, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
K = list(E.K); b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']
z = np.load(sys.argv[1]); names = [str(n) for n in z['names']]; F = z['F'][idx]; seen = z['seen'][idx]
h6 = np.load('/workspace/clear-ec-research/experiments/green_track_20261028/deploy/hex6_blend_params_12_oof_feature.npz'); hx = h6['feature'][idx].astype(float)
ly = np.log(np.where(y > 0, y, np.nan)); lB = np.log(B); r = ly - lB
print('rows seen', int(seen.sum()))
def blend(f, j, rows):
    lf = np.log(np.maximum(f, 1e-4)); p = lB[:, j].copy(); good = np.isfinite(lf) & np.isfinite(ly[:, j]) & rows
    for fo in range(5):
        tr = good & (fold != fo); te = rows & (fold == fo) & np.isfinite(lf)
        if tr.sum() < 50: continue
        best = (9e9, 0, 0, 0); A = np.polyfit(lf[tr], ly[tr, j], 1)
        for w in np.arange(0, 0.71, 0.04):
            q = (1 - w) * lB[tr, j] + w * (A[0] * lf[tr] + A[1]); e = np.mean(np.abs(np.exp(q) - y[tr, j]) / y[tr, j])
            if e < best[0]: best = (e, w, A[0], A[1])
        p[te] = (1 - best[1]) * lB[te, j] + best[1] * (best[2] * lf[te] + best[3])
    return p
def delta(p, j, rows):
    s = rows & (y[:, j] > 0); e1 = np.abs(np.exp(p[s]) - y[s, j]) / y[s, j] * 100; e0 = np.abs(B[s, j] - y[s, j]) / y[s, j] * 100; d = e1 - e0
    g = np.random.default_rng(0).integers(0, len(d), (1000, len(d))); m = d[g].mean(1)
    return round(float(d.mean()), 4), round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4), [round(float(d[fold[s] == fo].mean()), 3) for fo in range(5)]
rows_hx = seen & np.isfinite(hx)
print('hex6 r with log HEX', round(float(np.corrcoef(np.log(hx[rows_hx & (y[:, 2] > 0)]), ly[rows_hx & (y[:, 2] > 0), 2])[0, 1]), 4), 'partial', round(float(np.corrcoef(np.log(hx[rows_hx & (y[:, 2] > 0)]), r[rows_hx & (y[:, 2] > 0), 2])[0, 1]), 4), 'HEX blend', delta(blend(hx, 2, rows_hx), 2, rows_hx))
rep = {}
for n_, nm in enumerate(names):
    f = F[:, n_].astype(float); ok = seen & np.isfinite(f) & (f > 0)
    if ok.sum() < 100: continue
    lf = np.log(f); row = {}
    for j, k in enumerate(K):
        s = ok & np.isfinite(ly[:, j]); row[k] = (round(float(np.corrcoef(lf[s], ly[s, j])[0, 1]), 3), round(float(np.corrcoef(lf[s], r[s, j])[0, 1]), 3))
    rep[nm] = row
for j, k in enumerate(K):
    top = sorted(rep, key=lambda n: -abs(rep[n][k][1]))[:10]; print(k, 'B r', round(float(np.corrcoef(lB[y[:, j] > 0, j], ly[y[:, j] > 0, j])[0, 1]), 3)); out = {}
    for nm in top:
        f = F[:, names.index(nm)].astype(float); ok = seen & np.isfinite(f) & (f > 0); out[nm] = dict(r_y=rep[nm][k][0], r_resid=rep[nm][k][1], blend=delta(blend(f, j, ok), j, ok))
        print(' ', nm, json.dumps(out[nm]), flush=True)
json.dump(rep, open(sys.argv[1] + '.corr.json', 'w'))
