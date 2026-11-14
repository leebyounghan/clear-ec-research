"""Consensus cell graphs from the three pre-only detectors (2026-11-05). Each detector alone makes different misses and false peaks. Peaks >= 0.2 from all three are merged greedily by confidence within 0.35 spacing; a node keeps its mean position and carries votes / 3 and the spread of confidence as extra features after the nine usual ones. Node rule: at least two votes, or one detector >= 0.5. Windows: the three with the most nodes of at least two votes at mean confidence >= 0.5. Label-free. usage: cell_graph_consensus_20261105.py OUT. Forked pool, clamp(16 - load1 - 1, 2, 12)."""
import os, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.spatial import Delaunay, Voronoi, cKDTree
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import legible_window_features_20261105 as L
import sim_graph_data_20261105 as SG
G = '/workspace/clear-ec-research/experiments/green_track_20261028/external_srep'; FAM = ('preonly', 's47_preonly', 's48_preonly'); OUT = '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep/' + (sys.argv[1] if len(sys.argv) > 1 else 'cell_graphs_cons.npz'); W, H = 380, 300; PK = None


def load(path):
    o = {}
    for f in range(5):
        z = np.load(f'{path}/fold{f}/fold{f}_sg2_peaks.npz', allow_pickle=True); pk, of, ix = z['peaks'], z['offsets'], z['indices']
        for k, i in enumerate(ix): o[int(i)] = pk[of[k]:of[k + 1]]
    return o


def merge(r):
    a = [PK[k].get(r) for k in range(3)]; a = [x if x is not None else np.zeros((0, 3)) for x in a]; P = np.concatenate(a); src = np.concatenate([np.full(len(x), i) for i, x in enumerate(a)])
    if len(P) < 30: return None
    q = P[P[:, 2] >= 0.35, :2]; s = float(np.median(cKDTree(q).query(q, k=2)[0][:, 1])) if len(q) >= 30 else 22.4; order = np.argsort(-P[:, 2]); tree = cKDTree(P[:, :2]); used = np.zeros(len(P), bool); out = []
    for i in order:
        if used[i]: continue
        g = [j for j in tree.query_ball_point(P[i, :2], 0.35 * s) if not used[j]]; best = {}
        for j in g:
            if src[j] not in best or P[j, 2] > P[best[src[j]], 2]: best[src[j]] = j
        idx = list(best.values()); used[g] = True; c = P[idx, 2]; w = c / c.sum(); out.append((float((P[idx, 0] * w).sum()), float((P[idx, 1] * w).sum()), float(c.mean()), len(idx), float(c.std()), float(c.max())))
    return np.array(out)


def one(r):
    m = merge(r)
    if m is None: return r, []
    keep = (m[:, 3] >= 2) | (m[:, 5] >= 0.5); q = m[keep]; hi = m[(m[:, 3] >= 2) & (m[:, 2] >= 0.5), :2]
    if len(hi) < 10 or len(q) < 16: return r, []
    o = []
    for x0, y0 in L.windows(hi, W, H, k=3):
        t = (q[:, 0] >= x0) & (q[:, 0] < x0 + W) & (q[:, 1] >= y0) & (q[:, 1] < y0 + H); g = SG.graph(q[t][:, :3], x0, y0)
        if g is not None: o.append((np.c_[g[0], q[t][:, 3] / 3, q[t][:, 4]].astype(np.float32), g[1]))
    return r, o


def main():
    global PK
    PK = [load(f'{G}/{k}') for k in FAM]; workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1))); res = [None] * 8954
    with get_context('fork').Pool(workers) as pool:
        for r, g in pool.imap_unordered(one, range(8954), chunksize=16): res[r] = g
    X, E, offs, eoff = [], [], [0], [0]; n = m = 0
    for r in range(8954):
        for f, e in res[r] or []:
            X.append(f); E.append(e + n); n += len(f); m += len(e)
        offs.append(n); eoff.append(m)
    np.savez(OUT, X=np.concatenate(X), E=np.concatenate(E), wid=np.zeros(n, np.int8), offs=np.array(offs), eoff=np.array(eoff)); print('finished', workers, 'nodes', n, 'edges', m, flush=True)


if __name__ == '__main__':
    main()
