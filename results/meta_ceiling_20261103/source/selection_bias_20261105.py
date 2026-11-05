"""Technician selection as a statistic (2026-11-05). The window HEX residual sits in the top decile of window hex6 (+6.6 %, others about 0): when some patch is very regular, the label follows it. If technicians mark the best-looking patch, the label is not the frame's mean but a high quantile of the frame's local distribution: hex6 high, CV low. One Voronoi per frame from pre-only peaks, per-cell sides, area, neighbour distance; hull-sized sliding windows (380 x 300, stride 32), windows with at least 25 complete cells; per frame the mean and p10, p25, p50, p75, p90, p100 of local hex6, area CV, distance CV, CD, plus the same statistics at the most legible window and its rank within the frame. Two peak thresholds. Then LOFO HGB on the residual of B + hex6 + stretch per metric, with the K = 3 window pair as reference. Gate written before reading: better than hex6 + cv_dist of the K = 3 windows (-0.034) by at least 0.01 on the mean, CI upper bound < 0, at least 4 of 5 folds negative."""
import json, os, re, sys, warnings
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.spatial import Delaunay, Voronoi
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import abs_tess_20261104 as T
H_, W_ = T.H, T.W; UM = 1000 / 1296; W, Hh = 380, 300; OUT = '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep/selection_bias.npz'; PK = None
LOCAL = ['hex6', 'cv_area', 'cv_dist', 'cd']; Q = ['mean', 'p10', 'p25', 'p50', 'p75', 'p90', 'max', 'sd']; THR = (0.35, 0.5); NAMES = [f't{int(t * 100)}_{s}_{q}' for t in THR for s in LOCAL for q in Q] + [f't{int(t * 100)}_nwin' for t in THR]
xs = np.arange(0, W_ - W + 1, 32); ys = np.arange(0, H_ - Hh + 1, 32)


def cells(p):
    vor = Voronoi(p); tri = Delaunay(p); nb = {i: set() for i in range(len(p))}
    for a, c in vor.ridge_points: nb[a].add(c); nb[c].add(a)
    keep, S, A, D = [], [], [], []
    for i in range(len(p)):
        g = vor.regions[vor.point_region[i]]
        if -1 in g or not g: continue
        v = vor.vertices[g]
        if (tri.find_simplex(v) < 0).any(): continue
        x, z = v[:, 0], v[:, 1]; keep.append(i); S.append(len(g)); A.append(0.5 * abs(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1))) * UM * UM); D.append(np.mean([np.linalg.norm(p[i] - p[q]) for q in nb[i]]))
    return p[keep], np.array(S), np.array(A), np.array(D)


def one(r):
    pk = PK.get(r); out = []
    for t in THR:
        v = [np.nan] * (len(LOCAL) * len(Q) + 1)
        if pk is not None:
            q = pk[pk[:, 2] >= t, :2].astype(np.float64)
            if len(q) >= 40:
                try:
                    P, S, A, D = cells(q); vals = []
                    for y0 in ys:
                        my = (P[:, 1] >= y0) & (P[:, 1] < y0 + Hh)
                        for x0 in xs:
                            m = my & (P[:, 0] >= x0) & (P[:, 0] < x0 + W)
                            if m.sum() >= 25: a = A[m]; d = D[m]; vals.append(((S[m] == 6).mean(), a.std() / a.mean(), d.std() / d.mean(), 1e6 / a.mean()))
                    if len(vals) >= 3:
                        M = np.array(vals); v = []
                        for j in range(4):
                            c = M[:, j]; v += [c.mean(), *np.percentile(c, (10, 25, 50, 75, 90)), c.max(), c.std()]
                        v.append(float(len(vals)))
                except Exception:
                    pass
        out += v
    return r, out


def main():
    global PK
    PK = T.load_peaks(); workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1))); F = np.full((8954, len(NAMES)), np.nan, np.float32)
    with get_context('fork').Pool(workers) as pool:
        for k, (r, v) in enumerate(pool.imap_unordered(one, range(8954), chunksize=16)):
            F[r] = v
    np.savez(OUT, F=F, names=np.array(NAMES)); print('features done', workers, 'finite', float(np.isfinite(F[:, 0]).mean()), flush=True); evaluate()


def evaluate():
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    import ceiling_meta_cpu_20261103 as C
    import stretch_nested_20261103 as N
    E = C.E; K = list(C.K); b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']; g = E.donors(b['ids']); O = C.X / 'meta_ceiling_20261103/sweep'
    z = np.load(OUT); nm = [str(n) for n in z['names']]; F = z['F'][idx].astype(np.float64); zw = np.load(O / 'window_multi.npz'); Fw = zw['F'][idx].astype(np.float64); fam = [str(x) for x in zw['families']]; ks = zw['ks'].tolist(); st = [str(x) for x in zw['stats']]; P0 = Fw[:, fam.index('preonly'), ks.index(3)]
    gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); a_, b_, w_ = -0.2084, 0.5989, 0.28; have = gf['f_active10_valid'] & ~gf['seen_leaked'] & np.isfinite(gf['f_active10']); good = ~gf['seen_leaked']; f6 = gf['f_active10'].astype(float); H = B.copy(); H[have, 2] = np.exp((1 - w_) * np.log(B[have, 2]) + w_ * (a_ + b_ * f6[have]))
    REF = np.full_like(H, np.nan)
    for f in range(5): REF[fold == f] = N.apply_stretch(H[fold == f], N.fit_stretch(H, y, (fold != f) & good))
    ly = np.log(np.where(y > 0, y, np.nan)); r = ly - np.log(REF); rB = ly - np.log(B)
    lg = lambda X: np.where(X > 0, np.log(np.where(X > 0, X, 1)), np.nan)
    def cor(a, c):
        s = np.isfinite(a) & np.isfinite(c); return round(float(np.corrcoef(a[s], c[s])[0, 1]), 3)
    def res(a, x):
        s = np.isfinite(a) & np.isfinite(x); o = np.full(len(a), np.nan); A = np.c_[np.ones(s.sum()), x[s]]; o[s] = a[s] - A @ np.linalg.lstsq(A, a[s], rcond=None)[0]; return o
    print('reference ten-detector whole-frame hex6: r log HEX', cor(f6, ly[:, 2]), 'log CV', cor(f6, ly[:, 1]), flush=True)
    for s_, j in (('hex6', 2), ('hex6', 1), ('cv_area', 1), ('cv_dist', 1), ('cd', 0)):
        for t in ('t35', 't50'):
            print(s_, '->', K[j], t, {q: (cor(lg(F[:, nm.index(f'{t}_{s_}_{q}')]), ly[:, j]), cor(lg(F[:, nm.index(f'{t}_{s_}_{q}')]), r[:, j]), cor(res(lg(F[:, nm.index(f'{t}_{s_}_{q}')]), f6), res(r[:, j], f6))) for q in ('mean', 'p50', 'p75', 'p90', 'max', 'p10')}, flush=True)
    def fill(X):
        X = X.copy(); med = np.nanmedian(X, 0); bad = ~np.isfinite(X); X[bad] = np.take(med, np.nonzero(bad)[1]); return X
    def fit(X, ms):
        P = np.full_like(H, np.nan)
        for f in range(5):
            tr = (fold != f) & good; C_ = H.copy()
            for j in ms:
                q = tr & (y[:, j] > 0); t = np.log(y[q, j]) - np.log(H[q, j]); m = HGB(loss='absolute_error', max_depth=3, max_iter=200, learning_rate=0.03, min_samples_leaf=80, l2_regularization=5.0, random_state=0).fit(X[q], t); C_[:, j] = H[:, j] * np.exp(m.predict(X))
            P[fold == f] = N.apply_stretch(C_[fold == f], N.fit_stretch(C_, y, tr))
        return P
    def report(nm_, p):
        pb = E.paired_bootstrap(E.ape(p, y), E.ape(REF, y), g); s, s0 = E.official(E.ape(p, y)), E.official(E.ape(REF, y)); per = [round(E.official(E.ape(p[fold == f], y[fold == f]))['mean'] - E.official(E.ape(REF[fold == f], y[fold == f]))['mean'], 3) for f in range(5)]; o = dict(score=round(s['mean'], 3), per_delta={k: round(s[k] - s0[k], 3) for k in K}, delta=round(pb['delta'], 4), ci=[round(v, 4) for v in pb['ci95']], folds=per); print(nm_, json.dumps(o), flush=True); return o
    def G(t, s_, qs): return lg(np.stack([F[:, nm.index(f'{t}_{s_}_{q}')] for q in qs], 1))
    K3 = lg(np.stack([P0[:, st.index(n)] for n in ('hex6', 'cv_dist')], 1)); rep = {}
    rep['K3 hex6 + cv_dist (reference line)'] = report('K3 hex6 + cv_dist -> HEX', fit(fill(K3), [2]))
    for nm_, X in {'t35 hex6 quantiles': G('t35', 'hex6', ('p50', 'p75', 'p90', 'max')), 't50 hex6 quantiles': G('t50', 'hex6', ('p50', 'p75', 'p90', 'max')), 't35 hex6 p90': G('t35', 'hex6', ('p90',)), 't35 hex6 max': G('t35', 'hex6', ('max',)), 't35 hex6 mean': G('t35', 'hex6', ('mean',)), 't35 hex6 + cv_dist quantiles': np.c_[G('t35', 'hex6', ('p50', 'p75', 'p90', 'max')), G('t35', 'cv_dist', ('p10', 'p25', 'p50'))], 't35 + t50 hex6 + cv_dist': np.c_[G('t35', 'hex6', ('p50', 'p75', 'p90', 'max')), G('t35', 'cv_dist', ('p10', 'p25', 'p50')), G('t50', 'hex6', ('p50', 'p75', 'p90', 'max')), G('t50', 'cv_dist', ('p10', 'p25', 'p50'))], 'K3 + t35 quantiles': np.c_[K3, G('t35', 'hex6', ('p50', 'p75', 'p90', 'max')), G('t35', 'cv_dist', ('p10', 'p25', 'p50'))]}.items():
        rep[nm_] = report(nm_ + ' -> HEX', fit(fill(X), [2]))
    best = np.c_[K3, G('t35', 'hex6', ('p50', 'p75', 'p90', 'max')), G('t35', 'cv_dist', ('p10', 'p25', 'p50')), G('t35', 'cv_area', ('p10', 'p25', 'p50')), G('t35', 'cd', ('p25', 'p50', 'p75'))]
    rep['rich CV'] = report('windows + quantiles -> CV', fit(fill(best), [1])); rep['rich CD'] = report('windows + quantiles -> CD', fit(fill(best), [0]))
    json.dump(rep, open(O / 'selection_bias_eval.json', 'w'), indent=1); print('done', flush=True)


if __name__ == '__main__':
    main()
