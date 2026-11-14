"""Do augmented frames stay inside the measured distribution? 300 frames per source, donors as in training. Label-free (lane GPU 0, 2026-11-06)."""
import sys
from multiprocessing import Pool
import numpy as np, cv2
sys.path[:0] = ['experiments/lane_gpu0_20261106/source', 'experiments/meta_ceiling_20261103/source', 'experiments/phase2_pooling_20261020/source']
import measure_frames_20261106 as M
import calib2_20261106 as C
import recipeA_measured_aug2_20261106 as W

def one(arg):
    i, j, seed = arg
    cv2.setNumThreads(1)
    raw = np.memmap(M.RAW, mode='r', dtype=np.uint8, shape=(M.TOTAL, M.H, M.W))
    s = W.stats()
    a = W.transfer(np.asarray(raw[i]), i, j, int(s['src'][i]), np.random.default_rng(seed))
    st, lp = M.frame_stats(a)
    return st, lp

if __name__ == '__main__':
    s = W.stats(); r = np.random.default_rng(3); args = []
    for v in (0, 1):
        ix = np.flatnonzero(s['src'] == v)
        args += [(int(i), int(j), int(i)) for i, j in zip(r.choice(ix, 300, replace=False), r.choice(ix, 300))]
    with Pool(2) as p:
        out = p.map(one, args, chunksize=8)
    I = np.array([a[0] for a in args]); J = np.array([a[1] for a in args])
    LP = np.stack([o[1] for o in out]).astype(np.float64); ST = np.stack([o[0] for o in out])
    names = list(M.NAMES); sg = C.sigma_floor(LP); fc = C.focus2(LP, (0.07, 0.14)); Q = [1, 5, 25, 50, 75, 95, 99]
    for v, n in ((0, 'A'), (1, 'B')):
        k = s['src'][I] == v; real = s['src'] == v
        for nm, aug, rr, tgt in (('sigma', sg[k], s['sigma'][real], s['sigma'][J[k]]), ('focus', fc[k], s['focus'][real], s['focus'][J[k]]), ('p50', ST[k, names.index('p50')], s['p50'][real], s['p50'][J[k]]), ('p99', ST[k, names.index('p99')], s['p99'][real], s['p99'][J[k]])):
            print(n, nm.ljust(6), 'real', np.percentile(rr, Q).round(2).tolist(), 'augmented', np.percentile(aug, Q).round(2).tolist(), 'r(achieved, donor)', round(float(np.corrcoef(aug, tgt)[0, 1]), 3))
