"""Per-image cell tessellation fitted to the image, no training (2026-11-04). Starts from pre-only detector peaks and uses a physics prior:
endothelial cells are a gapless tiling separated by dark lines.
  border map  multi-scale dark-ridge response (largest Hessian eigenvalue) on a locally normalised frame, scales tied to the cell spacing
  flood       seeded watershed of the border map from peaks >= t0
  merge       neighbouring regions whose shared boundary has no ridge support and whose union is a plausible cell lose their boundary (false peaks)
  split       oversized regions with a strong interior ridge are divided (missed cells)
Cell = region: centroid, pixel area, neighbours sharing >= min_edge boundary pixels. This gives real boundaries, where Voronoi CV is biased 0.49x.
Read on the 25 expert marker sets: miss, FP, jitter, F1 at 12 px against the shipped pre-only numbers (0.150, 0.165, 0.215, 0.862), and CV and HEX
inside the marker hull against the device values (r and LOO-scaled MAPE), next to Voronoi on the same peaks and the expert oracle.
usage: abs_tess_20261104.py pilot OUT_DIR. Forked pool, clamp(16 - load1 - 1, 2, 12), OMP 1."""
import json, os, sys, itertools
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import Delaunay, Voronoi, cKDTree
from skimage.segmentation import watershed
cv2.setNumThreads(0)
ROOT = '/workspace/clear-ec-research'
RAW = f'{ROOT}/datasets/clear_ec_fullframe_20260904/img_full_raw.dat'; H, W = 972, 1296
PRE = f'{ROOT}/experiments/green_track_20261028/external_srep/preonly'
UM = 1000 / 1296


def load_peaks():
    out = {}
    for f in range(5):
        z = np.load(f'{PRE}/fold{f}/fold{f}_sg2_peaks.npz', allow_pickle=True); pk, of, ix = z['peaks'], z['offsets'], z['indices']
        for k, i in enumerate(ix):
            out[int(i)] = pk[of[k]:of[k + 1]]
    return out


def gf(a, s):
    return cv2.GaussianBlur(a, (0, 0), s, borderType=cv2.BORDER_REFLECT)


def border_map(a, s):
    x = a.astype(np.float32); bg = gf(x, 1.5 * s); sd = np.sqrt(gf((x - bg) ** 2, 2 * s)) + 3.0; n = (x - bg) / sd
    R = np.zeros_like(n)
    for sig in (0.06 * s, 0.09 * s, 0.13 * s):
        g = gf(n, max(sig, 0.8)); gy, gx = np.gradient(g); gyy, gyx = np.gradient(gy); gxy, gxx = np.gradient(gx)
        lam = 0.5 * (gxx + gyy + np.sqrt((gxx - gyy) ** 2 + 4 * gxy * gxy)) * sig * sig
        R = np.maximum(R, lam)
    return R


def rag(lab):
    """Shared boundary pixels per adjacent region pair: returns dict (i, j) -> (ys, xs of one side)."""
    a = lab[:, :-1].ravel(); b = lab[:, 1:].ravel(); yy, xx = np.mgrid[0:H, 0:W - 1]; m = (a != b) & (a > 0) & (b > 0)
    P = [(np.minimum(a[m], b[m]), np.maximum(a[m], b[m]), yy.ravel()[m], xx.ravel()[m])]
    a = lab[:-1, :].ravel(); b = lab[1:, :].ravel(); yy, xx = np.mgrid[0:H - 1, 0:W]; m = (a != b) & (a > 0) & (b > 0)
    P.append((np.minimum(a[m], b[m]), np.maximum(a[m], b[m]), yy.ravel()[m], xx.ravel()[m]))
    i, j, y, x = (np.concatenate(c) for c in zip(*P))
    key = i.astype(np.int64) * 100000 + j; order = np.argsort(key, kind='stable'); key, y, x = key[order], y[order], x[order]
    u, st = np.unique(key, return_index=True); st = np.r_[st, len(key)]
    return {(int(u[k] // 100000), int(u[k] % 100000)): (y[st[k]:st[k + 1]], x[st[k]:st[k + 1]]) for k in range(len(u))}


def tessellate(a, pts, t0=0.35, t_merge=0.35, t_split=0.6, grow=1.6, spacing=None):
    tissue = gf(a.astype(np.float32), 4) > 12
    seeds = pts[pts[:, 2] >= t0, :2]
    s = spacing or float(np.median(cKDTree(seeds).query(seeds, k=2)[0][:, 1]))
    R = border_map(a, s)
    mk = np.zeros((H, W), np.int32); yi, xi = np.clip(seeds[:, 1].round().astype(int), 0, H - 1), np.clip(seeds[:, 0].round().astype(int), 0, W - 1); mk[yi, xi] = np.arange(1, len(seeds) + 1)
    lab = watershed(gf(R, 0.8), mk, mask=tissue)
    # reference: ridge strength on confident edges of this image
    ed = rag(lab); strength = {k: float(np.mean(R[v])) for k, v in ed.items()}
    ref = np.median(list(strength.values())) if strength else 1.0
    area = np.bincount(lab.ravel()); med = np.median(area[1:][area[1:] > 0])
    # agglomerate weakest first
    parent = np.arange(len(area))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    n_merge = 0
    for (i, j), v in sorted(strength.items(), key=lambda kv: kv[1]):
        if v >= t_merge * ref:
            break
        a_, b_ = find(i), find(j)
        if a_ != b_ and area[a_] + area[b_] <= grow * med:
            parent[b_] = a_; area[a_] += area[b_]; area[b_] = 0; n_merge += 1
    lut = np.array([find(i) for i in range(len(parent))]); lab = lut[lab]
    # split oversized regions on an interior ridge
    n_split = 0; nxt = int(lab.max()) + 1; ob = ndi.find_objects(lab)
    for r, sl in enumerate(ob, 1):
        if sl is None or (lab[sl] == r).sum() < 1.5 * med:
            continue
        m = lab[sl] == r; dist = ndi.distance_transform_edt(m); g = gf(R, 1.0)[sl]
        # two seeds: strongest interior local maxima of distance minus ridge
        sc = dist / (dist.max() + 1e-6) - g / (g[m].max() + 1e-6); pk = (sc == ndi.maximum_filter(sc, size=int(0.6 * s) | 1)) & m & (dist > 0.2 * s)
        yy, xx = np.nonzero(pk)
        if len(yy) < 2:
            continue
        top = np.argsort(-sc[yy, xx])[:2]; mk2 = np.zeros(m.shape, np.int32); mk2[yy[top[0]], xx[top[0]]] = 1; mk2[yy[top[1]], xx[top[1]]] = 2
        sub = watershed(g, mk2, mask=m); bd = (sub == 1) & ndi.binary_dilation(sub == 2)
        if bd.sum() >= 3 and R[sl][bd].mean() >= t_split * ref and min((sub == 1).sum(), (sub == 2).sum()) >= 0.35 * med:
            part = lab[sl]; part[sub == 2] = nxt; nxt += 1; n_split += 1
    ids, lab = np.unique(lab, return_inverse=True); lab = lab.reshape(H, W)
    if ids[0] != 0:
        lab = lab + 1
    return lab, dict(spacing=s, ref=float(ref), n_seed=int(len(seeds)), n_merge=n_merge, n_split=n_split)


def cells(lab, min_edge=3):
    n = int(lab.max()); idx = np.arange(1, n + 1); area = ndi.sum_labels(np.ones_like(lab), lab, idx); cy, cx = zip(*ndi.center_of_mass(np.ones_like(lab), lab, idx)) if n else ((), ())
    ed = rag(lab); nb = np.zeros(n + 1, int)
    for (i, j), v in ed.items():
        if len(v[0]) >= min_edge:
            nb[i] += 1; nb[j] += 1
    # regions touching background or the frame are not complete cells
    touch = np.zeros(n + 1, bool); bgd = ndi.binary_dilation(lab == 0); touch[np.unique(lab[bgd])] = True; touch[np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])] = True; touch[0] = False
    return np.c_[cx, cy], area, nb[1:], touch[1:]


def measure(gt, det, sp):
    hull = Delaunay(gt)
    if len(det) == 0: return dict(miss=1., fp=0., jitter=np.nan, f1_12px=0.)
    dmin, _ = cKDTree(gt).query(det); d_ = det[(hull.find_simplex(det) >= 0) | (dmin <= 0.5 * sp)]
    d, j = cKDTree(d_).query(gt); used = set(); tp = []
    for dd, jj in sorted(zip(d, j)):
        if dd <= 0.5 * sp and jj not in used: used.add(jj); tp.append(dd)
    d12 = sum(1 for x in tp if x <= 12)
    return dict(miss=1 - len(tp) / len(gt), fp=(len(d_) - len(tp)) / len(gt), jitter=float(np.median(tp) / sp), f1_12px=2 * d12 / (len(gt) + len(d_)))


def voronoi_est(pts):
    vor = Voronoi(pts); tri = Delaunay(pts); ar, si = [], []
    for i in range(len(pts)):
        r = vor.regions[vor.point_region[i]]
        if -1 in r or not r: continue
        v = vor.vertices[r]
        if (tri.find_simplex(v) < 0).any(): continue
        x, y = v[:, 0], v[:, 1]; ar.append(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))); si.append(len(r))
    ar = np.array(ar); si = np.array(si)
    return float(ar.std() / ar.mean()), float((si == 6).mean()), len(ar)


def task(args):
    name, r, gt, pk, grid = args
    a = np.asarray(np.memmap(RAW, dtype=np.uint8, mode='r', shape=(8954, H, W))[r]); sp = float(np.median(cKDTree(gt).query(gt, k=2)[0][:, 1])); hull = Delaunay(gt)
    rows = {}
    for t0, tm, ts in grid:
        lab, info = tessellate(a, pk, t0, tm, ts); c, ar, nb, tch = cells(lab); e = measure(gt, c, sp)
        ins = (hull.find_simplex(c) >= 0) & ~tch
        e.update(cv=float(ar[ins].std() / ar[ins].mean()) if ins.sum() >= 12 else np.nan, hex=float((nb[ins] == 6).mean()) if ins.sum() >= 12 else np.nan, n=int(ins.sum()), **info); rows[f'{t0}_{tm}_{ts}'] = e
    base = {}
    q = pk[pk[:, 2] >= 0.35, :2].astype(np.float64); base['pre'] = measure(gt, q, sp)
    i2 = (hull.find_simplex(q) >= 0); cv, hx, n = voronoi_est(q[i2]); base['pre'].update(cv=cv, hex=hx, n=n)
    cv, hx, n = voronoi_est(gt); base['oracle'] = dict(cv=cv, hex=hx, n=n)
    return name, rows, base


def main():
    out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    ann = json.load(open(f'{ROOT}/datasets/clear_ec_official/green_eval/annotations.json'))
    with np.load(f'{ROOT}/datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True) as z: ids = z['ids'].astype(str)
    pos = {n: i for i, n in enumerate(ids)}; P = load_peaks()
    grid = list(itertools.product((0.2, 0.35, 0.5), (0.0, 0.25, 0.4, 0.55, 0.7), (0.6, 99.0)))
    workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1)))
    tasks = [(n, pos[n], np.asarray(e['markers_mha'], float), P[pos[n]], grid) for n, e in ann.items()]
    with get_context('fork').Pool(workers) as pool: res = list(pool.imap_unordered(task, tasks))
    gt = {n: ann[n]['gt'] for n, _, _ in res}
    def summary(get, needs=True):
        o = {k: float(np.nanmean([get(n, rows, base)[k] for n, rows, base in res])) for k in ('miss', 'fp', 'jitter', 'f1_12px')} if needs else {}
        for m, key in (('cv', 'CV'), ('hex', 'HEX')):
            x = np.array([get(n, rows, base)[m] for n, rows, base in res]); y = np.array([gt[n][key] for n, _, _ in res]); ok = np.isfinite(x) & (x > 0)
            lx, ly = np.log(x[ok]), np.log(y[ok]); pred = np.array([np.exp((lx[i] - np.delete(lx, i).mean()) + np.delete(ly, i).mean()) for i in range(ok.sum())])
            o[m + '_r'] = float(np.corrcoef(x[ok], y[ok])[0, 1]); o[m + '_loo_mape'] = float(np.mean(np.abs(pred - y[ok]) / y[ok]) * 100); o[m + '_ratio'] = float(np.median(x[ok] / y[ok]))
        return o
    rep = dict(n=len(res), workers=workers, baseline_pre=summary(lambda n, r, b: b['pre']), oracle=summary(lambda n, r, b: b['oracle'], False), grid={})
    print('pre', json.dumps(rep['baseline_pre'])); print('oracle', json.dumps(rep['oracle']), flush=True)
    for t0, tm, ts in grid:
        k = f'{t0}_{tm}_{ts}'; s = summary(lambda n, r, b: r[k]); s['merges'] = float(np.mean([r[k]['n_merge'] for _, r, _ in res])); s['splits'] = float(np.mean([r[k]['n_split'] for _, r, _ in res])); rep['grid'][k] = s
        print(k, json.dumps({a: round(v, 3) for a, v in s.items()}), flush=True)
    json.dump(dict(rep=rep, per_image={n: dict(rows=r, base=b) for n, r, b in res}), open(f'{out}/pilot25.json', 'w'), indent=1, default=float)
    print('done', flush=True)


if __name__ == '__main__':
    main()
