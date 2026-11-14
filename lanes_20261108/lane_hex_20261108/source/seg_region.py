"""HEX from saved Cellpose-SAM masks, by region and neighbour radius (HEX lane, 2026-11-08). CPU only.
Per frame: adjacency at R = 2, 3, 5 px; interior cells (not on the frame edge, ring background < 0.25, area > 30);
3x3 grid of windows; features: hex6 over all interior cells per R, hex6 in the centre window, in the window with most
interior cells, pooled over the top-3 windows, over cells of regular size (0.5-2x median area), and with neighbours that
are themselves regular (no tiny fragments). usage: seg_region.py CHUNKDIR(rel LANE) OUT(rel LANE) [workers] [max_chunks]"""
import sys, glob, os, numpy as np
from multiprocessing import Pool
from scipy import ndimage as ndi
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
RMAX = 5; OFFS = [(dy, dx, dy * dy + dx * dx) for dy in range(-RMAX, RMAX + 1) for dx in range(0, RMAX + 1) if (dx > 0 or dy > 0) and dy * dy + dx * dx <= RMAX * RMAX]
NAMES = ['h6_r2', 'h6_r3', 'h6_r5', 'h6_center', 'h6_bestwin', 'h6_top3', 'h6_regular', 'h6_regnb', 'n_top3', 'cv_top3', 'h6_r3_top3', 'h5_top3', 'h7_top3']


def frame_feats(M):
    M = M.astype(np.int32); H, W = M.shape; n = int(M.max()); out = [np.nan] * len(NAMES)
    if n < 10: return out
    KS, DS = [], []
    for dy, dx, d2 in OFFS:
        y0, y1 = max(0, -dy), H - max(0, dy); a = M[y0:y1, 0:W - dx]; b = M[y0 + dy:y1 + dy, dx:W]
        k = (a > 0) & (b > 0) & (a != b)
        if not k.any(): continue
        u = np.unique(np.minimum(a[k], b[k]).astype(np.int64) * (n + 1) + np.maximum(a[k], b[k])); KS.append(u); DS.append(np.full(len(u), d2))
    allk, alld = np.concatenate(KS), np.concatenate(DS); o = np.lexsort((alld, allk)); allk, alld = allk[o], alld[o]
    keys, i0 = np.unique(allk, return_index=True); d2s = alld[i0]
    pa, pb = keys // (n + 1), keys % (n + 1)
    area = np.bincount(M.ravel(), minlength=n + 1).astype(float)
    edge = np.unique(np.r_[M[0], M[-1], M[:, 0], M[:, -1]])
    R = 3; dil = ndi.grey_dilation(M, size=(2 * R + 1, 2 * R + 1)); ring = (M == 0) & (dil > 0)
    bg = np.bincount(dil[ring], minlength=n + 1).astype(float); per = np.bincount(dil[(dil > 0) & (dil != M)], minlength=n + 1).astype(float)
    ok = np.ones(n + 1, bool); ok[0] = False; ok[edge] = False; ok &= bg / np.maximum(per, 1) < 0.25; ok &= area > 30
    if ok.sum() < 20: return out
    med = np.median(area[ok]); reg = (area > 0.5 * med) & (area < 2.0 * med)
    def nb(r2, sel=None):
        m = d2s <= r2
        if sel is not None: m &= sel[pa] & sel[pb]
        return np.bincount(np.r_[pa[m], pb[m]], minlength=n + 1)
    nb2, nb3, nb5 = nb(4), nb(9), nb(25); nbr = nb(9, area > 0.25 * med)
    cy = ndi.mean(np.indices(M.shape)[0], M, np.arange(n + 1)); cx = ndi.mean(np.indices(M.shape)[1], M, np.arange(n + 1))
    wy = np.clip((np.nan_to_num(cy) / (H / 3)).astype(int), 0, 2); wx = np.clip((np.nan_to_num(cx) / (W / 3)).astype(int), 0, 2); win = wy * 3 + wx
    cnt = np.bincount(win[ok], minlength=9); top = np.argsort(-cnt)[:3]; intop = ok & np.isin(win, top)
    def h(nbv, sel, k=6): return float(np.mean(nbv[sel] == k)) if sel.sum() >= 5 else np.nan
    a3 = area[intop]
    return [h(nb2, ok), h(nb3, ok), h(nb5, ok), h(nb3, ok & (win == 4)), h(nb3, ok & (win == top[0])), h(nb3, intop), h(nb3, ok & reg),
            h(nbr, ok), float(intop.sum()), float(a3.std() / a3.mean()) if len(a3) > 5 else np.nan, h(nb2, intop), h(nb3, intop, 5), h(nb3, intop, 7)]


def do_chunk(p):
    z = np.load(p); return z['frames'], np.array([frame_feats(M) for M in z['masks']], np.float32)


if __name__ == '__main__':
    fs = sorted(glob.glob(f'{LANE}/{sys.argv[1]}/c[0-9][0-9][0-9].npz'))
    if len(sys.argv) > 4: fs = fs[:int(sys.argv[4])]
    W = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    with Pool(W) as pool: res = pool.map(do_chunk, fs, chunksize=1)
    np.savez_compressed(os.path.join(LANE, sys.argv[2]), frames=np.concatenate([r[0] for r in res]), F=np.concatenate([r[1] for r in res]), names=np.array(NAMES))
    print('done', len(fs), 'chunks', flush=True)
