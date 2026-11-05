"""Reads the frequency-domain lattice features against the labels on all frames (2026-11-05). Pearson with log y and with the log residual of B + hex6 + stretch per feature; partial after hex6; LOFO gradient boosting (L1) on that residual, inside the deployed pipeline B -> hex6 -> correction -> stretch. Gate written before reading: mean <= -0.02 on top of B + hex6 + stretch, CI upper bound < 0, at least 4 of 5 folds negative."""
import json, sys, warnings
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import stretch_nested_20261103 as N
E = C.E; K = list(C.K); OUT = C.X / 'meta_ceiling_20261103/sweep'
b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']; g = E.donors(b['ids']); N_ = len(y)
z = np.load(OUT / 'freq_lattice.npz'); names = [str(n) for n in z['names']]; F = z['F'][idx].astype(np.float64)
gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); A_, B_, W_ = -0.2084, 0.5989, 0.28; have = gf['f_active10_valid'] & ~gf['seen_leaked'] & np.isfinite(gf['f_active10']); good = ~gf['seen_leaked']; hx = np.where(np.isfinite(gf['f_active10']), gf['f_active10'], np.nan)
H = B.copy(); H[have, 2] = np.exp((1 - W_) * np.log(B[have, 2]) + W_ * (A_ + B_ * gf['f_active10'][have]))
def post(base, tr):
    return N.fit_stretch(base, y, tr)
REF = np.full_like(H, np.nan)
for f in range(5):
    REF[fold == f] = N.apply_stretch(H[fold == f], post(H, (fold != f) & good))
ly = np.log(np.where(y > 0, y, np.nan)); r = ly - np.log(REF); print('B + hex6 + stretch', round(E.official(E.ape(REF, y))['mean'], 3), 'tile coverage: frames with features', int(np.isfinite(F[:, 0]).sum()), flush=True)
rep = {}
def cor(a, c):
    s = np.isfinite(a) & np.isfinite(c); return float(np.corrcoef(a[s], c[s])[0, 1])
def res(a, x):
    s = np.isfinite(a) & np.isfinite(x); out = np.full(len(a), np.nan); A = np.c_[np.ones(s.sum()), x[s]]; out[s] = a[s] - A @ np.linalg.lstsq(A, a[s], rcond=None)[0]; return out
for n, nm in enumerate(names):
    f = F[:, n]; rep[nm] = {K[j]: (round(cor(f, ly[:, j]), 3), round(cor(f, r[:, j]), 3), round(cor(res(f, hx), res(r[:, j], hx)), 3)) for j in range(3)}
for j, k in enumerate(K):
    print(k, '[r with log y, r with residual, partial after hex6]; n =', N_, 'SE about', round(1 / np.sqrt(N_), 3)); print('  hex6 for reference', round(cor(hx, ly[:, j]), 3), round(cor(hx, r[:, j]), 3))
    for nm in sorted(rep, key=lambda q: -abs(rep[q][k][0]))[:6]: print('  top by y   ', nm, rep[nm][k])
    for nm in sorted(rep, key=lambda q: -abs(rep[q][k][1]))[:6]: print('  top by res ', nm, rep[nm][k])
X = F.copy(); med = np.nanmedian(X, 0); bad = ~np.isfinite(X); X[bad] = np.take(med, np.nonzero(bad)[1])
def run(metrics):
    P = np.full_like(H, np.nan)
    for f in range(5):
        tr = (fold != f) & good; C_ = H.copy()
        for j in metrics:
            q = tr & (y[:, j] > 0); t = np.log(y[q, j]) - np.log(H[q, j]); m = HGB(loss='absolute_error', max_depth=3, max_iter=250, learning_rate=0.03, min_samples_leaf=60, l2_regularization=5.0, random_state=0).fit(X[q], t); C_[:, j] = H[:, j] * np.exp(m.predict(X))
        P[fold == f] = N.apply_stretch(C_[fold == f], post(C_, tr))
    return P
def report(name, p):
    pb = E.paired_bootstrap(E.ape(p, y), E.ape(REF, y), g); s, s0 = E.official(E.ape(p, y)), E.official(E.ape(REF, y)); per = [round(E.official(E.ape(p[fold == f], y[fold == f]))['mean'] - E.official(E.ape(REF[fold == f], y[fold == f]))['mean'], 3) for f in range(5)]
    o = dict(score=round(s['mean'], 3), per_delta={k: round(s[k] - s0[k], 3) for k in K}, delta=round(pb['delta'], 4), ci=[round(v, 4) for v in pb['ci95']], folds=per); print(name, json.dumps(o), flush=True); return o
rep2 = {nm: report(nm, run(m)) for nm, m in dict(CD=[0], CV=[1], HEX=[2], all3=[0, 1, 2]).items()}
json.dump(dict(features=rep, stack=rep2), open(OUT / 'freq_eval.json', 'w'), indent=1); print('done', flush=True)
