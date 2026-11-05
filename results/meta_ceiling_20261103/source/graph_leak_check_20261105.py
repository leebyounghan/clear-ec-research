"""Could the graph network's gain be donor leakage (2026-11-05)? Old-format IDs are not donor-grouped: 81 % of adjacent pairs, about half fellow eyes, straddle folds, and their residuals correlate 0.38, 0.31, 0.22. A network that recognises a donor could carry the partner's error. Tests: gain on new-format rows (donor-grouped, no leak possible) against old-format rows; on old-format rows with and without an adjacent ID in another fold; correlation between the prediction and the partner's training residual, against the correlation with the row's own; year slices. Graph ensemble pre46 + pre47, CV and HEX, against B + hex6 + stretch."""
import json, re, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import stretch_nested_20261103 as N
E = C.E; K = list(C.K); b = E.load_base(); y, B, fold = b['y'], b['B'], b['fold']; g = E.donors(b['ids']); O = C.X / 'meta_ceiling_20261103/sweep'; ids = [re.sub(r'\s+', ' ', i.strip()) for i in b['ids']]; N_ = len(y)
fmt = np.array([0 if re.fullmatch(r'\d{4}-\d{2}', i) else 1 for i in ids]); yr = np.array([2000 + int(i[-2:]) if f == 0 else int(i[:4]) for i, f in zip(ids, fmt)])
gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); a_, b_, w_ = -0.2084, 0.5989, 0.28; have = gf['f_active10_valid'] & ~gf['seen_leaked'] & np.isfinite(gf['f_active10']); good = ~gf['seen_leaked']; f6 = gf['f_active10'].astype(float); H = B.copy(); H[have, 2] = np.exp((1 - w_) * np.log(B[have, 2]) + w_ * (a_ + b_ * f6[have]))
def pipe(pr, ms):
    P = np.full_like(H, np.nan)
    for f in range(5):
        tr = (fold != f) & good; C_ = H.copy()
        for j in ms: C_[:, j] = H[:, j] * np.exp(pr[f][:, j])
        P[fold == f] = N.apply_stretch(C_[fold == f], N.fit_stretch(C_, y, tr))
    return P
REF = pipe({f: np.zeros_like(H) for f in range(5)}, ()); R = [np.load(O / 'graph_sweep' / f'{r}.npz') for r in sys.argv[1].split(',')]; pr = {f: np.mean([z[f'f{f}'] for z in R], 0).astype(np.float64) for f in range(5)}; P = pipe(pr, (1, 2))
def rep(nm, m):
    pb = E.paired_bootstrap(E.ape(P[m], y[m]), E.ape(REF[m], y[m]), np.unique(g[m], return_inverse=True)[1]); s, s0 = E.official(E.ape(P[m], y[m])), E.official(E.ape(REF[m], y[m])); print(nm, 'n', int(m.sum()), 'score', round(s0['mean'], 3), '->', round(s['mean'], 3), 'delta', round(pb['delta'], 4), [round(v, 4) for v in pb['ci95']], 'CV', round(s['CV'] - s0['CV'], 3), 'HEX', round(s['HEX'] - s0['HEX'], 3), flush=True)
rep('all', np.ones(N_, bool)); rep('new format (donor-grouped, no leak possible)', fmt == 1); rep('old format', fmt == 0)
old = {}
for n, i in enumerate(ids):
    m = re.fullmatch(r'(\d{4})-(\d{2})', i)
    if m: old[(int(m.group(2)), int(m.group(1)))] = n
nbr = {}
for k, n in old.items():
    for d in (-1, 1):
        q = old.get((k[0], k[1] + d))
        if q is not None: nbr.setdefault(n, []).append(q)
other = np.zeros(N_, bool); same = np.zeros(N_, bool); none = fmt == 0
for n in old.values():
    v = nbr.get(n, [])
    if any(fold[q] != fold[n] for q in v): other[n] = True
    elif v: same[n] = True
    if v: none[n] = False
rep('old, adjacent ID in another fold (partner in training)', other); rep('old, adjacent ID only in same fold', same); rep('old, no adjacent ID', none)
for a in range(2018, 2025):
    if (yr == a).sum() > 100: rep(f'year {a}', yr == a)
# does the held-out prediction follow the partner's residual beyond the row's own?
Oof = np.zeros((N_, 3));
for f in range(5): Oof[fold == f] = pr[f][fold == f]
r0 = np.log(np.where(y > 0, y, np.nan)) - np.log(H); pairs = [(n, q) for n, v in nbr.items() for q in v if fold[q] != fold[n]]; P_ = np.array(pairs)
for j, k in ((1, 'CV'), (2, 'HEX')):
    s = np.isfinite(r0[P_[:, 0], j]) & np.isfinite(r0[P_[:, 1], j]); a, c = P_[s, 0], P_[s, 1]; own = np.corrcoef(Oof[a, j], r0[a, j])[0, 1]; part = np.corrcoef(Oof[a, j], r0[c, j])[0, 1]; rr = np.corrcoef(r0[a, j], r0[c, j])[0, 1]; X = np.c_[np.ones(len(a)), r0[a, j]]; res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]; pc = np.corrcoef(res(Oof[a, j]), res(r0[c, j]))[0, 1]; print(k, 'pred vs own residual', round(float(own), 3), 'pred vs partner residual', round(float(part), 3), 'partner residual vs own', round(float(rr), 3), 'partial of pred on partner, given own', round(float(pc), 3), 'expected from own alone', round(float(own * rr), 3), flush=True)
m1 = fmt == 1; print('new-format r(pred, own residual)', {k: round(float(np.corrcoef(Oof[m1 & np.isfinite(r0[:, j]), j], r0[m1 & np.isfinite(r0[:, j]), j])[0, 1]), 3) for j, k in ((1, 'CV'), (2, 'HEX'))}, 'old', {k: round(float(np.corrcoef(Oof[(~m1) & np.isfinite(r0[:, j]), j], r0[(~m1) & np.isfinite(r0[:, j]), j])[0, 1]), 3) for j, k in ((1, 'CV'), (2, 'HEX'))}, flush=True)
print('done', flush=True)
