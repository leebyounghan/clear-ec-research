"""Technician emulator, then a residual read of every legible-window statistic (2026-11-05). The window choice nearly doubled the correlation of point statistics with the labels (CV r 0.21 -> 0.44, CD 0.39 -> 0.57) with no labels used. A technician also marks only cells they can see, in one contiguous patch of about 140 cells. Emulation: from the best-supported peak in the most legible window, grow a patch through peaks >= T, linked at <= 1.7 x spacing, in order of confidence, to N cells; Voronoi statistics on it, for T in 0.5, 0.7 and N in 100, 140. Then LOFO residual fit (L1) on the log residual of B + hex6 + stretch, all window and emulator features, inside the pipeline. Gate written before reading: mean <= -0.02, CI upper bound < 0, at least 4 of 5 folds negative."""
import heapq, json, os, sys, warnings
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import abs_tess_20261104 as T
import legible_window_features_20261105 as L
RULES = [(0.5, 100), (0.5, 140), (0.7, 100), (0.7, 140)]; STAT = ['hex6', 'cv_area', 'cv_dist', 'cd', 'n']; NAMES = [f'emu_t{int(t * 100)}_n{n}_{s}' for t, n in RULES for s in STAT]; PK = None; OUT = '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep/technician_emulator.npz'


def patch(pk, t, n):
    q = pk[pk[:, 2] >= t]
    if len(q) < 20:
        return None
    xy = q[:, :2].astype(np.float64); tree = cKDTree(xy); s = float(np.median(tree.query(xy, k=2)[0][:, 1])); c = np.array([len(v) for v in tree.query_ball_point(xy, 4 * s)], float) * q[:, 2]; start = int(np.argmax(c)); seen = {start}; heap = [(-float(q[start, 2]), start)]; out = []
    while heap and len(out) < n:
        _, i = heapq.heappop(heap); out.append(i)
        for j in tree.query_ball_point(xy[i], 1.7 * s):
            if j not in seen: seen.add(j); heapq.heappush(heap, (-float(q[j, 2]), j))
    return xy[out]


def one(r):
    pk = PK.get(r); out = []
    for t, n in RULES:
        p = None if pk is None else patch(pk, t, n); out += L.stats(p) if p is not None else [np.nan] * 5
    return r, out


def main():
    global PK
    PK = T.load_peaks(); workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1))); F = np.full((8954, len(NAMES)), np.nan, np.float32)
    with get_context('fork').Pool(workers) as pool:
        for r, v in pool.imap_unordered(one, sorted(PK), chunksize=32): F[r] = v
    np.savez(OUT, F=F, names=np.array(NAMES)); print('features done', workers, flush=True); evaluate(F)


def evaluate(E_):
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    sys.path.insert(0, HERE)
    import ceiling_meta_cpu_20261103 as C
    import stretch_nested_20261103 as N
    E = C.E; K = list(C.K); b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']; g = E.donors(b['ids']); O = C.X / 'meta_ceiling_20261103/sweep'
    z = np.load(O / 'legible_window.npz'); nw = [str(n) for n in z['names']]; W = z['F'][idx].astype(float); Em = E_[idx].astype(float)
    gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); a_, b_, w_ = -0.2084, 0.5989, 0.28; have = gf['f_active10_valid'] & ~gf['seen_leaked'] & np.isfinite(gf['f_active10']); good = ~gf['seen_leaked']; f6 = gf['f_active10'].astype(float); H = B.copy(); H[have, 2] = np.exp((1 - w_) * np.log(B[have, 2]) + w_ * (a_ + b_ * f6[have]))
    def pipeline(base):
        o = np.full_like(base, np.nan)
        for f in range(5): o[fold == f] = N.apply_stretch(base[fold == f], N.fit_stretch(base, y, (fold != f) & good))
        return o
    REF = pipeline(H); ly = np.log(np.where(y > 0, y, np.nan)); r = ly - np.log(REF); rB = ly - np.log(B)
    def cor(a, c):
        s = np.isfinite(a) & np.isfinite(c); return round(float(np.corrcoef(a[s], c[s])[0, 1]), 3)
    def res(a, x):
        s = np.isfinite(a) & np.isfinite(x); o = np.full(len(a), np.nan); A = np.c_[np.ones(s.sum()), x[s]]; o[s] = a[s] - A @ np.linalg.lstsq(A, a[s], rcond=None)[0]; return o
    for n_, nm in enumerate(NAMES):
        if nm.endswith('_n'): continue
        v = Em[:, n_]; v = np.where(v > 0, np.log(v), np.nan); j = {'hex6': 2, 'cv_area': 1, 'cv_dist': 1, 'cd': 0}[nm.rsplit('_', 1)[1] if not nm.endswith(('cv_area', 'cv_dist')) else nm[-7:]]; print(nm, K[j], 'r y', cor(v, ly[:, j]), 'r B-resid', cor(v, rB[:, j]), 'r final', cor(v, r[:, j]), 'partial after hex6', cor(res(v, f6), res(r[:, j], f6)), flush=True)
    sel = [n for n in nw if n.startswith(('top_hull', 'top3_hull', 'top_box')) and not n.endswith(('_x', '_y'))]; X = np.c_[np.stack([W[:, nw.index(n)] for n in sel], 1), Em, f6]; X = np.where(X > 0, np.log(np.where(X > 0, X, 1)), X); med = np.nanmedian(X, 0); bad = ~np.isfinite(X); X[bad] = np.take(med, np.nonzero(bad)[1])
    def report(nm, p):
        pb = E.paired_bootstrap(E.ape(p, y), E.ape(REF, y), g); s, s0 = E.official(E.ape(p, y)), E.official(E.ape(REF, y)); per = [round(E.official(E.ape(p[fold == f], y[fold == f]))['mean'] - E.official(E.ape(REF[fold == f], y[fold == f]))['mean'], 3) for f in range(5)]; o = dict(score=round(s['mean'], 3), per_delta={k: round(s[k] - s0[k], 3) for k in K}, delta=round(pb['delta'], 4), ci=[round(v, 4) for v in pb['ci95']], folds=per); print(nm, json.dumps(o), flush=True); return o
    rep = {}
    for nm, ms in dict(CD=[0], CV=[1], HEX=[2], all3=[0, 1, 2]).items():
        P = np.full_like(H, np.nan)
        for f in range(5):
            tr = (fold != f) & good; C_ = H.copy()
            for j in ms:
                q = tr & (y[:, j] > 0); t = np.log(y[q, j]) - np.log(H[q, j]); m = HGB(loss='absolute_error', max_depth=3, max_iter=200, learning_rate=0.03, min_samples_leaf=80, l2_regularization=5.0, random_state=0).fit(X[q], t); C_[:, j] = H[:, j] * np.exp(m.predict(X))
            P[fold == f] = N.apply_stretch(C_[fold == f], N.fit_stretch(C_, y, tr))
        rep[nm] = report('residual fit ' + nm, P)
    json.dump(rep, open(O / 'technician_emulator_eval.json', 'w'), indent=1); print('done', flush=True)


if __name__ == '__main__':
    main()
