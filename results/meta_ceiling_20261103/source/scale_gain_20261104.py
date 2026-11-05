"""Is an image's own zoom response a usable signal (2026-11-04)? Physics says d log CD / d log s = -2. An image whose response is flatter is one where the model is reading something other than
cell size. Tests on the 500 fold-0 validation rows: correlation of the per-image slope with the residual and with its size; a label-free per-image gain correction, log p' = m + (log p - m) * (-2 / slope) clipped, m = mean prediction;
LOFO-style half-split linear use of slope in the residual. Also reports recipe A per-epoch scores next to the two running trainings."""
import json, sys
import numpy as np
A = '/workspace/clear-ec-research/experiments/meta_ceiling_20261103'
z = np.load(f'{A}/zoom/response_f0.json.npz'); idx, L, y = z['idx'], z['L'], z['y']; S = np.array((0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4)); ls = np.log(S)
def slope(sel):
    x = ls[list(sel)] - ls[list(sel)].mean(); W = L[:, list(sel)]; return ((W - W.mean(1, keepdims=True)) * x[None, :, None]).sum(1) / (x * x).sum()
def off(p, q=None):
    q = np.ones(len(p), bool) if q is None else q; s = y[q] > 0; e = np.where(s, np.abs(p[q] - y[q]) / np.where(s, y[q], 1.0), np.nan) * 100; m = np.nanmean(e, 0); return dict(CD=round(float(m[0]), 3), CV=round(float(m[1]), 3), HEX=round(float(m[2]), 3), mean=round(float(m.mean()), 3))
ly = np.log(np.where(y > 0, y, np.nan)); P = L[:, 3]; r = ly - P; print('s=1', off(np.exp(P)))
for nm, sel in dict(narrow=(2, 3, 4), mid=(1, 2, 3, 4, 5), wide=(0, 1, 2, 3, 4, 5, 6)).items():
    g = slope(sel); o = {}
    for j, k in enumerate(('CD', 'CV', 'HEX')):
        q = np.isfinite(r[:, j]); o[k] = dict(slope_median=round(float(np.median(g[q, j])), 3), r_slope_resid=round(float(np.corrcoef(g[q, j], r[q, j])[0, 1]), 3), r_slope_abs_resid=round(float(np.corrcoef(g[q, j], np.abs(r[q, j]))[0, 1]), 3), r_slope_pred=round(float(np.corrcoef(g[q, j], P[q, j])[0, 1]), 3), r_cdslope_resid=round(float(np.corrcoef(g[q, 0], r[q, j])[0, 1]), 3))
    print(nm, json.dumps(o))
    m = P[:, 0].mean(); gain = np.clip(-2 / np.minimum(g[:, 0], -0.5), 1.0, 2.0); p = np.exp(P).copy(); p[:, 0] = np.exp(m + (P[:, 0] - m) * gain); print('  per-image physical gain on CD', off(p))
    gm = -2 / np.median(g[:, 0]); p = np.exp(P).copy(); p[:, 0] = np.exp(m + (P[:, 0] - m) * gm); print('  one global physical gain', round(float(gm), 3), off(p))
    # how well does s=1 CD plus the slope explain CD?  half splits
    rng = np.random.default_rng(0); dl = {k: [] for k in ('CD', 'CV', 'HEX')}
    for _ in range(40):
        pm = rng.permutation(len(y)); a, c = pm[:250], pm[250:]
        for j, k in enumerate(('CD', 'CV', 'HEX')):
            f = np.c_[np.ones(len(y)), g[:, 0], g[:, j]]; oa = a[np.isfinite(r[a, j])]; oc = c[np.isfinite(r[c, j])]; w = np.linalg.lstsq(f[oa], r[oa, j], rcond=None)[0]; pc = P[oc, j] + f[oc] @ w
            w0 = r[oa, j].mean(); e1 = np.abs(np.exp(pc) - y[oc, j]) / y[oc, j]; e0 = np.abs(np.exp(P[oc, j] + w0) - y[oc, j]) / y[oc, j]; dl[k].append((e1.mean() - e0.mean()) * 100)
    print('  half-split residual fit on slopes, delta vs offset only', {k: round(float(np.mean(v)), 3) for k, v in dl.items()})
for nm, root in dict(recipeA='experiments/phase2_pooling_replication_20261026/f0_s46_std', clean='experiments/meta_ceiling_20261103/label_clean/f0_s46', formula='experiments/meta_ceiling_20261103/formula_head/f0_s46').items():
    h = json.load(open(f'/workspace/clear-ec-research/{root}/runs/fold0/cpdino_vitb/history.json')); it = h if isinstance(h, list) else h.get('epochs', h.get('history'))
    print(nm, 'epoch mean', [round(e['score']['mean'], 3) for e in it]); print(' CD', [round(e['score']['CD'], 2) for e in it]); print(' CV', [round(e['score']['CV'], 2) for e in it]); print(' HEX', [round(e['score']['HEX'], 2) for e in it])
