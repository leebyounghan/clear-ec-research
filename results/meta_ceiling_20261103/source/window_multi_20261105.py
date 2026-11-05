"""Legible-window cell statistics from several detectors, all frames (2026-11-05). The three most legible hull-sized windows of the single pre-only detector gave -0.032 on HEX on top of B + hex6 + stretch (CI -0.046, -0.017; whole frame -0.011, centre window -0.006, hex6 alone 0). Each detector family picks its own windows by peaks >= 0.5 (K = 3 and 5, greedy non-overlap); Voronoi on peaks >= 0.35 in each window; cell-weighted statistics: hex6, shares of 5, 7, <=4, >=8 sides, mean sides, area CV, neighbour-distance CV, NN-distance CV, CD, cells, mean peak confidence. Writes sweep/window_multi.npz (frame x family x K x stat). Forked pool, clamp(16 - load1 - 1, 2, 12)."""
import json, os, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.spatial import Delaunay, Voronoi, cKDTree
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import legible_window_features_20261105 as L
G = '/workspace/clear-ec-research/experiments/green_track_20261028'
FAM = {'preonly': f'{G}/external_srep/preonly', 'preonly_s47': f'{G}/external_srep/s47_preonly', 'preonly_s48': f'{G}/external_srep/s48_preonly', 'sg2': f'{G}/sg2_specialist', 'sg2_s47': f'{G}/sg2_specialist_s47', 'sg2_s48': f'{G}/sg2_specialist_s48', 'sg2_s49': f'{G}/sg2_specialist_s49', 'sg2_s50': f'{G}/sg2_specialist_s50'}
STAT = ['hex6', 'f5', 'f7', 'f4m', 'f8p', 'sides', 'cv_area', 'cv_dist', 'cv_nn', 'cd', 'n', 'conf']; KS = (3, 5); W, H = 380, 300; UM = 1000 / 1296; OUT = '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep/window_multi.npz'; PK = None


def load(path):
    out = {}
    for f in range(5):
        z = np.load(f'{path}/fold{f}/fold{f}_sg2_peaks.npz', allow_pickle=True); pk, of, ix = z['peaks'], z['offsets'], z['indices']
        for k, i in enumerate(ix): out[int(i)] = pk[of[k]:of[k + 1]]
    return out


def stats(pts, conf):
    if len(pts) < 16: return None
    try:
        vor = Voronoi(pts); tri = Delaunay(pts)
    except Exception:
        return None
    nb = {i: set() for i in range(len(pts))}
    for a, c in vor.ridge_points: nb[a].add(c); nb[c].add(a)
    A, S, D = [], [], []
    for i in range(len(pts)):
        g = vor.regions[vor.point_region[i]]
        if -1 in g or not g: continue
        v = vor.vertices[g]
        if (tri.find_simplex(v) < 0).any(): continue
        x, z = v[:, 0], v[:, 1]; A.append(0.5 * abs(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1))) * UM * UM); S.append(len(g)); D.append(np.mean([np.linalg.norm(pts[i] - pts[q]) for q in nb[i]]))
    if len(A) < 8: return None
    A, S, D = np.array(A), np.array(S), np.array(D); nn = cKDTree(pts).query(pts, k=2)[0][:, 1]
    return [float((S == 6).mean()), float((S == 5).mean()), float((S == 7).mean()), float((S <= 4).mean()), float((S >= 8).mean()), float(S.mean()), float(A.std() / A.mean()), float(D.std() / D.mean()), float(nn.std() / nn.mean()), float(1e6 / A.mean()), float(len(A)), float(conf.mean())]


def one(r):
    res = np.full((len(FAM), len(KS), len(STAT)), np.nan, np.float32)
    for a, nm in enumerate(FAM):
        pk = PK[nm].get(r)
        if pk is None: continue
        q = pk[pk[:, 2] >= 0.35]; hi = pk[pk[:, 2] >= 0.5, :2].astype(np.float64)
        if len(hi) < 10 or len(q) < 16: continue
        ws = L.windows(hi, W, H, k=max(KS)); got = []
        for x0, y0 in ws:
            m = (q[:, 0] >= x0) & (q[:, 0] < x0 + W) & (q[:, 1] >= y0) & (q[:, 1] < y0 + H); got.append(stats(q[m, :2].astype(np.float64), q[m, 2]))
        for b, K in enumerate(KS):
            v = [g for g in got[:K] if g is not None]
            if v:
                v = np.array(v); w = v[:, STAT.index('n')]; res[a, b] = (v * w[:, None]).sum(0) / w.sum(); res[a, b, STAT.index('n')] = w.sum()
    return r, res


def main():
    global PK
    PK = {k: load(v) for k, v in FAM.items()}; print('loaded', {k: len(v) for k, v in PK.items()}, flush=True); workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1))); F = np.full((8954, len(FAM), len(KS), len(STAT)), np.nan, np.float32)
    with get_context('fork').Pool(workers) as pool:
        for k, (r, v) in enumerate(pool.imap_unordered(one, range(8954), chunksize=16)):
            F[r] = v
            if (k + 1) % 2000 == 0: print(k + 1, flush=True)
    np.savez(OUT, F=F, families=np.array(list(FAM)), ks=np.array(KS), stats=np.array(STAT)); print('done', workers, flush=True)


if __name__ == '__main__':
    main()
