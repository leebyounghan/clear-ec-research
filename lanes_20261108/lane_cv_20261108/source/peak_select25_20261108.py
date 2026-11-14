"""CV lane (2026-11-08). Peak selection with a per-image spacing prior, read on the 25 expert-marker frames.
Stored pre-only peaks (conf >= 0.2, NMS 12 px) are re-selected: spacing s from predicted CD (hybrid B, OOF) or from the image's own
high-confidence peaks; adaptive NMS at alpha*s; optional gap filling with lower-confidence candidates farther than beta*s from kept peaks.
Read: miss, FP, jitter/spacing, F1 at 12 px inside the marker hull; CV statistics in the hull, r and LOO-scaled MAPE against device CV.
usage: peak_select25_20261108.py. CPU, 4 workers."""
import json, os, re, sys, warnings
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.spatial import Delaunay, Voronoi, cKDTree
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import abs_tess_20261104 as T
E = C.E; b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']; ids = [re.sub(r'\s+', ' ', i.strip()) for i in b['ids']]; pos = {n: i for i, n in enumerate(ids)}
ann = json.load(open(C.X.parent / 'datasets/clear_ec_official/green_eval/annotations.json')); names = list(ann); gt = np.array([ann[n]['gt']['CV'] for n in names]); MK = [np.asarray(ann[n]['markers_mha'], float) for n in names]
PK = T.load_peaks(); UMPX = 0.7716


def spacing_from_cd(cd):
    return np.sqrt(2e6 / (np.sqrt(3) * cd)) / UMPX


def stats(p):
    if len(p) < 16: return {}
    vor = Voronoi(p); tri = Delaunay(p); ar = []
    for i in range(len(p)):
        g = vor.regions[vor.point_region[i]]
        if -1 in g or not g: continue
        v_ = vor.vertices[g]
        if (tri.find_simplex(v_) < 0).any(): continue
        x, z = v_[:, 0], v_[:, 1]; ar.append(0.5 * abs(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1))))
    if len(ar) < 10: return {}
    ar = np.array(ar); E_ = np.array(vor.ridge_points); d = np.linalg.norm(p[E_[:, 0]] - p[E_[:, 1]], axis=1); d = d[d < 3 * np.median(d)]; nn = cKDTree(p).query(p, k=2)[0][:, 1]; lo, hi = np.percentile(ar, (10, 90)); t = ar[(ar >= lo) & (ar <= hi)]; mad = np.median(np.abs(ar - np.median(ar))) * 1.4826
    return dict(area_cv=ar.std() / ar.mean(), area_trim=t.std() / t.mean(), area_mad=mad / np.median(ar), log_area_sd=np.log(ar).std(), edge_cv=d.std() / d.mean(), nn_cv=nn.std() / nn.mean())


def read(vals):
    o = {}
    for k in ('area_cv', 'area_trim', 'area_mad', 'edge_cv', 'nn_cv', 'log_area_sd'):
        x = np.array([q.get(k, np.nan) for q in vals]); m = np.isfinite(x)
        if m.sum() < 10: continue
        lx, ly = np.log(x[m]), np.log(gt[m]); pr = np.exp(lx - lx.mean() + ly.mean()); o[k] = (round(float(np.corrcoef(lx, ly)[0, 1]), 3), round(float(np.mean(np.abs(pr - gt[m]) / gt[m]) * 100), 2))
    return o


def select(pk, s, t_hi=0.35, alpha=0.0, t_lo=None, beta=0.8):
    """Greedy NMS by confidence at alpha*s on peaks >= t_hi; then fill with peaks in [t_lo, t_hi) farther than beta*s from kept ones."""
    q = pk[pk[:, 2] >= t_hi]; q = q[np.argsort(-q[:, 2])]; keep = []
    if alpha > 0:
        tree_pts = []
        for i in range(len(q)):
            if tree_pts:
                d = np.min(np.linalg.norm(np.asarray(tree_pts) - q[i, :2], axis=1))
                if d < alpha * s: continue
            tree_pts.append(q[i, :2]); keep.append(i)
        q = q[keep]
    if t_lo is not None and t_lo < t_hi:
        c = pk[(pk[:, 2] >= t_lo) & (pk[:, 2] < t_hi)]; c = c[np.argsort(-c[:, 2])]; pts = list(q[:, :2])
        add = []
        for i in range(len(c)):
            d = np.min(np.linalg.norm(np.asarray(pts) - c[i, :2], axis=1)) if pts else 1e9
            if d >= beta * s: pts.append(c[i, :2]); add.append(c[i])
        if add: q = np.r_[q, np.asarray(add)]
    return q[:, :2].astype(float)


def match(p, m, sp, r=12.0):
    if len(p) == 0: return 1.0, 0.0, np.nan, 0.0
    D = np.linalg.norm(p[:, None] - m[None], axis=2); D2 = np.where(D <= r, D, 1e6); a, c = linear_sum_assignment(D2); ok = D2[a, c] < 1e5; tp = int(ok.sum())
    miss = 1 - tp / len(m); fp = 1 - tp / len(p); jit = float(D[a[ok], c[ok]].mean() / sp) if tp else np.nan; f1 = 2 * tp / (len(p) + len(m)); return miss, fp, jit, f1


def evaluate(cfg):
    nm, kw, spsrc = cfg; vals = []; mm = []
    for n, mk in zip(names, MK):
        r = pos[n]; pk = PK[int(idx[r])]; sp_m = float(np.median(cKDTree(mk).query(mk, k=2)[0][:, 1]))
        if spsrc == 'cd': s = spacing_from_cd(B[r, 0])
        elif spsrc == 'img':
            hi = pk[pk[:, 2] >= 0.6, :2].astype(float); s = float(np.median(cKDTree(hi).query(hi, k=2)[0][:, 1])) if len(hi) > 20 else spacing_from_cd(B[r, 0])
        else: s = sp_m
        q = select(pk, s, **kw); hull = Delaunay(mk); qi = q[hull.find_simplex(q) >= 0]
        mm.append(match(qi, mk, sp_m)); vals.append(stats(qi))
    mm = np.array(mm); return nm, dict(miss=round(float(np.nanmean(mm[:, 0])), 3), fp=round(float(np.nanmean(mm[:, 1])), 3), jit=round(float(np.nanmean(mm[:, 2])), 3), f1=round(float(np.nanmean(mm[:, 3])), 3)), read(vals)


def main():
    # check: spacing from CD vs marker spacing
    sc = []; si = []
    for n, mk in zip(names, MK):
        r = pos[n]; sp_m = float(np.median(cKDTree(mk).query(mk, k=2)[0][:, 1])); sc.append((spacing_from_cd(B[r, 0]), spacing_from_cd(y[r, 0]), sp_m))
    sc = np.array(sc); print('spacing px: from predicted CD median', np.median(sc[:, 0]).round(2), 'from label CD', np.median(sc[:, 1]).round(2), 'marker NN median', np.median(sc[:, 2]).round(2), 'r(pred, marker)', np.corrcoef(sc[:, 0], sc[:, 2])[0, 1].round(3), flush=True)
    cfgs = [(f'base t{t}', dict(t_hi=t), 'cd') for t in (0.2, 0.3, 0.35, 0.5)]
    for src in ('cd', 'img'):
        for t in (0.2, 0.3, 0.35):
            for a in (0.5, 0.6, 0.7, 0.8):
                cfgs.append((f'{src} nms a{a} t{t}', dict(t_hi=t, alpha=a), src))
        for t in (0.35, 0.5):
            for a in (0.6, 0.7):
                for bb in (0.7, 0.8, 0.9):
                    cfgs.append((f'{src} nms a{a} t{t} fill0.2 b{bb}', dict(t_hi=t, alpha=a, t_lo=0.2, beta=bb), src))
    cfgs.append(('oracle-spacing nms a0.7 t0.2', dict(t_hi=0.2, alpha=0.7), 'marker'))
    with get_context('fork').Pool(4) as pool: R = pool.map(evaluate, cfgs, chunksize=1)
    out = {}
    for nm, m, rd in R:
        out[nm] = dict(match=m, cv=rd); print(nm.ljust(36), json.dumps(m), json.dumps(rd), flush=True)
    os.makedirs('/workspace/clear-ec-research/experiments/lane_cv_20261108/out', exist_ok=True); json.dump(out, open('/workspace/clear-ec-research/experiments/lane_cv_20261108/out/peak_select25.json', 'w'), indent=1); print('done', flush=True)


if __name__ == '__main__':
    main()
