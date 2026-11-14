"""Centroid-Delaunay hexagonality from saved segmentation masks (HEX lane, 2026-11-08). CPU only, one process.
For each frame: cell centroids -> Delaunay graph; edges longer than 1.6 x the median edge are dropped (gaps in the
segmentation); a cell counts when it is >= 1.5 median edges from the image border and kept all its edges. Features:
vor_h6/h5/h7 (fraction of counted cells with 6/5/7 neighbours), n_vor, edge-length CV, mean degree, CD from the mean edge
(hexagonal packing), h6 agreement with the mask adjacency is not used. usage: seg_vor.py INDIR OUT.npz (rel to LANE)"""
import sys, os, glob
import numpy as np
from scipy.spatial import Delaunay
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
UM = 1000 / 1296
NAMES = ['vor_h6', 'vor_h5', 'vor_h7', 'n_vor', 'edge_cv', 'deg_mean', 'cd_edge', 'drop_frac']


def feats(M):
    out = [np.nan] * len(NAMES); M = np.asarray(M, np.int64); n = int(M.max())
    if n < 20: return out
    H, W = M.shape; yy, xx = np.indices(M.shape); lab = M.ravel()
    cnt = np.bincount(lab, minlength=n + 1).astype(float); cy = np.bincount(lab, yy.ravel(), n + 1) / np.maximum(cnt, 1); cx = np.bincount(lab, xx.ravel(), n + 1) / np.maximum(cnt, 1)
    keep = cnt > 30; keep[0] = False; ids = np.where(keep)[0]
    if len(ids) < 20: return out
    P = np.c_[cy[ids], cx[ids]]; T = Delaunay(P).simplices
    E = np.unique(np.sort(np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [0, 2]]], 1), axis=0); L = np.hypot(*(P[E[:, 0]] - P[E[:, 1]]).T)
    med = np.median(L); long = L > 1.6 * med; bad = np.zeros(len(ids), bool); bad[E[long].ravel()] = True
    Es = E[~long]; deg = np.bincount(Es.ravel(), minlength=len(ids))
    border = (P[:, 0] < 1.5 * med) | (P[:, 1] < 1.5 * med) | (P[:, 0] > H - 1.5 * med) | (P[:, 1] > W - 1.5 * med)
    c = ~bad & ~border
    if c.sum() < 5: return out
    d = deg[c]; Ls = L[~long]; me = Ls.mean() * UM
    return [np.mean(d == 6), np.mean(d == 5), np.mean(d == 7), float(c.sum()), Ls.std() / Ls.mean(), d.mean(), 1e6 / (me * me * np.sqrt(3) / 2), long.mean()]


def main():
    ind = os.path.join(LANE, sys.argv[1]); out = os.path.join(LANE, sys.argv[2]); FR, FF = [], []
    files = sorted(glob.glob(os.path.join(ind, 'c[0-9][0-9][0-9].npz')))
    for k, p in enumerate(files):
        z = np.load(p); Ms = z['masks']; FR.append(z['frames']); FF.append(np.array([feats(M) for M in Ms], np.float32))
        if k % 10 == 0: print(k, len(files), flush=True)
    np.savez(out, frames=np.concatenate(FR), F=np.concatenate(FF), names=np.array(NAMES)); print('done', out, flush=True)


if __name__ == '__main__':
    main()
