"""What exactly carries the window HEX gain, and is it stable (2026-11-05)? Pre-only, K = 3 windows. Single statistics in the LOFO HGB; a linear form (L1, log statistics, deployable without a tree library); booster seeds; slices by source and year; pooled with the re-selected members; dependence shape of the residual on the window hex6 and cell count. Reference: B + hex6 + stretch."""
import json, re, sys, warnings
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import stretch_nested_20261103 as N
E = C.E; K = list(C.K); b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']; g = E.donors(b['ids']); O = C.X / 'meta_ceiling_20261103/sweep'; ids = [re.sub(r'\s+', ' ', i.strip()) for i in b['ids']]
fmt = np.array([0 if re.fullmatch(r'\d{4}-\d{2}', i) else 1 for i in ids]); yr = np.array([2000 + int(i[-2:]) if f == 0 else int(i[:4]) for i, f in zip(ids, fmt)])
z = np.load(O / 'window_multi.npz'); F = z['F'][idx].astype(np.float64); fam = [str(x) for x in z['families']]; ks = z['ks'].tolist(); st = [str(x) for x in z['stats']]; P0 = F[:, fam.index('preonly'), ks.index(3)]
gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); a_, b_, w_ = -0.2084, 0.5989, 0.28; have = gf['f_active10_valid'] & ~gf['seen_leaked'] & np.isfinite(gf['f_active10']); good = ~gf['seen_leaked']; f6 = gf['f_active10'].astype(float)
def hex6(base):
    H = base.copy(); H[have, 2] = np.exp((1 - w_) * np.log(base[have, 2]) + w_ * (a_ + b_ * f6[have])); return H
H = hex6(B)
def pipeline(base):
    o = np.full_like(base, np.nan)
    for f in range(5): o[fold == f] = N.apply_stretch(base[fold == f], N.fit_stretch(base, y, (fold != f) & good))
    return o
REF = pipeline(H); ly = np.log(np.where(y > 0, y, np.nan))
def lg(X): return np.where(X > 0, np.log(np.where(X > 0, X, 1)), np.nan)
def get(names): return lg(np.stack([P0[:, st.index(n)] for n in names], 1))
def fill(X):
    X = X.copy(); med = np.nanmedian(X, 0); bad = ~np.isfinite(X); X[bad] = np.take(med, np.nonzero(bad)[1]); return X
def fit(X, base=None, seed=0):
    base = H if base is None else base; P = np.full_like(base, np.nan)
    for f in range(5):
        tr = (fold != f) & good; C_ = base.copy(); q = tr & (y[:, 2] > 0); t = np.log(y[q, 2]) - np.log(base[q, 2]); m = HGB(loss='absolute_error', max_depth=3, max_iter=200, learning_rate=0.03, min_samples_leaf=80, l2_regularization=5.0, random_state=seed).fit(X[q], t); C_[:, 2] = base[:, 2] * np.exp(m.predict(X)); P[fold == f] = N.apply_stretch(C_[fold == f], N.fit_stretch(C_, y, tr))
    return P
def irls(A, t, it=25):
    w = np.linalg.lstsq(A, t, rcond=None)[0]
    for _ in range(it):
        d = 1 / np.maximum(np.abs(t - A @ w), 1e-3); w = np.linalg.solve((A * d[:, None]).T @ A + 1e-6 * np.eye(A.shape[1]), (A * d[:, None]).T @ t)
    return w
def linear(X):
    P = np.full_like(H, np.nan); W = []
    for f in range(5):
        tr = (fold != f) & good; C_ = H.copy(); q = tr & (y[:, 2] > 0); m = X[q].mean(0); s = X[q].std(0); A = np.c_[np.ones(q.sum()), (X[q] - m) / s]; w = irls(A, np.log(y[q, 2]) - np.log(H[q, 2])); C_[:, 2] = H[:, 2] * np.exp(np.c_[np.ones(len(X)), (X - m) / s] @ w); P[fold == f] = N.apply_stretch(C_[fold == f], N.fit_stretch(C_, y, tr)); W.append(np.round(w, 4).tolist())
    return P, W
def report(nm, p, ref=REF, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask; pb = E.paired_bootstrap(E.ape(p[m], y[m]), E.ape(ref[m], y[m]), np.unique(g[m], return_inverse=True)[1]); s, s0 = E.official(E.ape(p[m], y[m])), E.official(E.ape(ref[m], y[m])); per = [round(E.official(E.ape(p[m & (fold == f)], y[m & (fold == f)]))['mean'] - E.official(E.ape(ref[m & (fold == f)], y[m & (fold == f)]))['mean'], 3) for f in range(5)]; o = dict(n=int(m.sum()), score=round(s['mean'], 3), HEX=round(s['HEX'], 3), HEX_delta=round(s['HEX'] - s0['HEX'], 3), delta=round(pb['delta'], 4), ci=[round(v, 4) for v in pb['ci95']], folds=per); print(nm, json.dumps(o), flush=True); return o
rep = {}; BASIC = ['hex6', 'cv_area', 'cv_dist', 'cd', 'n']; X5 = fill(get(BASIC))
for nm in BASIC: rep['single ' + nm] = report('single ' + nm, fit(fill(get([nm]))))
for sub in (['hex6', 'n'], ['hex6', 'cd'], ['hex6', 'cv_dist'], ['hex6', 'cd', 'n'], ['cd', 'n'], ['cv_area', 'cv_dist'], ['hex6', 'f5', 'f7', 'n']): rep['+'.join(sub)] = report('+'.join(sub), fit(fill(get(sub))))
p0 = fit(X5); rep['all5'] = report('all5', p0); rep['all5 seeds'] = [report(f'all5 seed {s}', fit(X5, seed=s))['delta'] for s in (1, 2)]
pl, wl = linear(X5); rep['linear'] = report('linear all5', pl); print('  linear weights fold 0 (intercept, standardised hex6, cv_area, cv_dist, cd, n)', wl[0], flush=True)
pl2, wl2 = linear(fill(get(['hex6', 'n']))); rep['linear2'] = report('linear hex6+n', pl2); print('  weights', wl2[0], flush=True)
for nm, m in dict(old=fmt == 0, new=fmt == 1).items(): rep['slice ' + nm] = report('slice ' + nm, p0, mask=m)
for yy in range(2018, 2025):
    m = yr == yy
    if m.sum() > 100: rep[f'year {yy}'] = report(f'year {yy}', p0, mask=m)
r0 = ly[:, 2] - np.log(H[:, 2]); v = get(['hex6'])[:, 0]; n_ = get(['n'])[:, 0]; d_ = get(['cd'])[:, 0]
def shape(x, nm):
    s = np.isfinite(x) & np.isfinite(r0); q = np.quantile(x[s], [0, .1, .25, .5, .75, .9, 1]); print(' residual median by', nm, 'bins', [(round(float(np.exp(q[i])), 3), round(float(np.median(r0[s & (x >= q[i]) & (x <= q[i + 1])])), 4)) for i in range(6)], flush=True)
shape(v, 'window hex6'); shape(n_, 'cell count'); shape(d_, 'window CD'); shape(np.log(H[:, 0]) - d_, 'log (frame CD pred / window CD)')
c = np.load(O / 'combo_preds.npz'); key = [k for k in c.files if k.startswith('bagged_per_metric_from_B')][0]; M = c[key].astype(np.float64); HM = hex6(M); RM = pipeline(HM); rep['members + hex6 + stretch'] = report('re-selected members + hex6 + stretch, vs B + hex6 + stretch', RM); PM = fit(X5, HM); rep['members + hex6 + window + stretch'] = report('re-selected members + hex6 + window + stretch, vs B + hex6 + stretch', PM); rep['window on members'] = report('window, on re-selected members', PM, RM); print('vs stored B', E.official(E.ape(PM, y))['mean'] - E.official(E.ape(B, y))['mean'], flush=True)
json.dump(rep, open(O / 'window_hex_detail.json', 'w'), indent=1, default=float); print('done', flush=True)
