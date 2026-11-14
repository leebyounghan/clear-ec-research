"""Cell-route lane (2026-11-09): centroids and areas of the zero-shot Cellpose-SAM loose masks (lane_hex_20261108/seg_loose, read-only) for all frames.
Writes feats/seg_centroids.npz: indices (dataset index), offsets, pts (x, y, area_px). 2 workers (I/O bound)."""
import os, glob
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy import ndimage as ndi
D = '/workspace/clear-ec-research/experiments/lane_hex_20261108/seg_loose/'
def chunk(fn):
    z = np.load(fn); fr = z['frames']; M = z['masks']; out = []
    for k in range(len(fr)):
        m = M[k]; lab = np.unique(m); lab = lab[lab > 0]
        if len(lab) == 0: out.append((int(fr[k]), np.zeros((0, 3), np.float32))); continue
        one = np.ones_like(m, np.float32); cy, cx = np.array(ndi.center_of_mass(one, m, lab)).T; ar = ndi.sum(one, m, lab)
        out.append((int(fr[k]), np.c_[cx, cy, ar].astype(np.float32)))
    return out
if __name__ == '__main__':
    fns = sorted(glob.glob(D + 'c[0-9][0-9][0-9].npz')); R = {}
    with get_context('fork').Pool(2) as pool:
        for j, res in enumerate(pool.imap_unordered(chunk, fns)):
            for d, p in res: R[d] = p
            if j % 20 == 0: print(j, len(R), flush=True)
    ix = np.array(sorted(R)); of = np.r_[0, np.cumsum([len(R[d]) for d in ix])]
    np.savez_compressed('/workspace/clear-ec-research/experiments/lane_cellroute_20261109/feats/seg_centroids.npz', indices=ix, offsets=of, pts=np.concatenate([R[d] for d in ix]))
    print('done frames', len(ix), 'points', of[-1], flush=True)
