"""Write consensus peaks of a detector set in the per-fold peak-file format, so cell_graph_prep3 can build graphs from them
(HEX lane, 2026-11-08). Link peaks >= 0.2 within 0.4 x spacing (median NN distance of the densest detector at >= 0.35);
cluster position = confidence-weighted mean; score = sum over detectors of their best confidence in the cluster / D.
usage: fuse_peaks.py OUTDIR SET"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
from multiprocessing import get_context
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source')
OUTD = sys.argv[1]; SETN = sys.argv[2]
sys.argv = [sys.argv[0], 'x', SETN]
import feat_fuse as FF
DETS = FF.SETS[SETN]; D = len(DETS); PK = []


def one(fr):
    qs = [pk.get(fr) for pk in PK]
    if any(q is None for q in qs): return fr, None
    big = max((q[q[:, 2] >= 0.35] for q in qs), key=len)
    if len(big) < 20: return fr, None
    sp = float(np.median(cKDTree(big[:, :2]).query(big[:, :2], k=2)[0][:, 1]))
    P = np.concatenate([q[:, :2] for q in qs]).astype(np.float64); Cf = np.concatenate([q[:, 2] for q in qs]); DI = np.concatenate([np.full(len(q), k) for k, q in enumerate(qs)])
    pr = cKDTree(P).query_pairs(0.4 * sp, output_type='ndarray'); n = len(P)
    Am = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n)) if len(pr) else coo_matrix((n, n)); nc, lab = connected_components(Am, directed=False)
    sw = np.bincount(lab, Cf, nc); X = np.c_[np.bincount(lab, Cf * P[:, 0], nc) / sw, np.bincount(lab, Cf * P[:, 1], nc) / sw]
    key = lab * 64 + DI; o = np.lexsort((-Cf, key)); first = np.r_[True, key[o][1:] != key[o][:-1]]; sc = np.bincount(lab[o][first], Cf[o][first], nc) / D
    k = sc >= 0.1; return fr, np.c_[X[k], sc[k]].astype(np.float32)


def main():
    global PK
    FF.T = 0.2; PK = [FF.load(d) for d in DETS]; res = {}
    with get_context('fork').Pool(int(os.environ.get('NW', '3'))) as pool:
        for r, o in pool.imap_unordered(one, range(8954), chunksize=16): res[r] = o
    z0 = '/workspace/clear-ec-research/experiments/green_track_20261028/external_srep/preonly'
    for f in range(5):
        ix = np.load(f'{z0}/fold{f}/fold{f}_sg2_peaks.npz', allow_pickle=True)['indices']; pk = []; of = [0]
        for i in ix:
            q = res.get(int(i)); q = q if q is not None else np.zeros((0, 3), np.float32); pk.append(q); of.append(of[-1] + len(q))
        os.makedirs(f'{OUTD}/fold{f}', exist_ok=True); np.savez(f'{OUTD}/fold{f}/fold{f}_sg2_peaks.npz', peaks=np.concatenate(pk), offsets=np.array(of), indices=ix)
    print('done', SETN, D, flush=True)


if __name__ == '__main__':
    main()
