"""Procedural specular-microscopy corneal endothelium renderer (route B, 2026-11-03): exact cells, no human labels.

Geometry: a jittered hexagonal lattice (spacing from CD; 27 px median Delaunay spacing at the median label CD 2830
in our detections) turned into a power diagram (per-cell weights -> polymegathism / CV) and relaxed by a few Lloyd
steps (regularity -> HEX). Appearance: bright domed cell bodies, dark borders of variable width, per-cell
brightness, granular texture, guttae, a smooth illumination field with optional vignette (the "framed" style),
spatially varying defocus, noise and gamma. Labels: a 538x408 ROI box (centre jitter sd 137 x 92 px like the
technician boxes); CD = cells in ROI / ROI area, CV = sd/mean of their areas, HEX = share with six neighbours,
computed on complete cells only. Outputs: image uint8, instance map, centres, ROI, CD/CV/HEX.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

H, W = 972, 1296
UM_PER_PX = 0.77                      # from 27 px spacing at CD 2830 (hexagonal packing)
ROI_W, ROI_H = 538, 408


def up(a):
    return ndi.zoom(a, (H / a.shape[0], W / a.shape[1]), order=1)


def spacing_px(cd):
    area_um2 = 1e6 / cd
    return np.sqrt(2 * area_um2 / np.sqrt(3)) / UM_PER_PX


def centres(rng, cd, jitter, pad=60):
    a = spacing_px(cd)
    th = rng.uniform(0, np.pi / 3)
    n = int(np.hypot(H, W) / a) + 6
    i, j = np.meshgrid(np.arange(-n, n), np.arange(-n, n))
    x = a * (i + 0.5 * (j % 2)); y = a * np.sqrt(3) / 2 * j
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    p = (R @ np.c_[x.ravel(), y.ravel()].T).T + [W / 2, H / 2]
    keep = (p[:, 0] > -pad) & (p[:, 0] < W + pad) & (p[:, 1] > -pad) & (p[:, 1] < H + pad)
    p = p[keep] + rng.normal(0, jitter * a, (keep.sum(), 2))
    # local density modulation (smooth field) so CD varies a little across the frame
    return p, a


def power_labels(p, w, k=12, step=1, h=None, wd=None):
    h, wd = h or H, wd or W
    yy, xx = np.mgrid[0:h:step, 0:wd:step]
    q = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    d, nb = cKDTree(p).query(q, k=k)
    pw = d ** 2 - w[nb]
    lab = nb[np.arange(len(q)), pw.argmin(1)]
    return lab.reshape(xx.shape)


def lloyd(lab, n, step=1):
    idx = np.arange(n)
    yy, xx = np.mgrid[0:lab.shape[0], 0:lab.shape[1]] * step
    cnt = ndi.sum(np.ones_like(lab), lab, idx)
    cx = ndi.sum(xx, lab, idx) / np.maximum(cnt, 1); cy = ndi.sum(yy, lab, idx) / np.maximum(cnt, 1)
    return np.c_[cx, cy], cnt


def geometry(rng, cd, jitter, size_sd, relax, pad=60):
    """Random points (Poisson) relaxed by `relax` Lloyd steps on a power diagram: 0 steps ~ Poisson-Voronoi
    (HEX ~0.3, CV ~0.5), more steps -> more regular. `jitter` re-perturbs after relaxation (pleomorphism)."""
    a = spacing_px(cd)
    n = int((H + 2 * pad) * (W + 2 * pad) / (np.sqrt(3) / 2 * a ** 2))
    p = rng.uniform([-pad, -pad], [W + pad, H + pad], (n, 2))
    w = rng.lognormal(0, size_sd, n); w = (w / w.mean() - 1) * a ** 2 * 0.5
    for _ in range(relax):                                     # Lloyd at half resolution (4x faster)
        lab = power_labels(p, w, step=2)
        c, cnt = lloyd(lab, n, step=2)
        p = np.where((cnt > 0)[:, None], c, p)
    p = p + rng.normal(0, jitter * a, p.shape)
    return power_labels(p, w), p, a


def neighbours(lab):
    """Cell adjacency from 4-connected label changes, vectorised (unique pair codes)."""
    n = int(lab.max()) + 1
    u = np.r_[lab[:, :-1].ravel(), lab[:-1, :].ravel()].astype(np.int64)
    v = np.r_[lab[:, 1:].ravel(), lab[1:, :].ravel()].astype(np.int64)
    m = u != v
    lo, hi = np.minimum(u[m], v[m]), np.maximum(u[m], v[m])
    code = np.unique(lo * n + hi)
    a, b = code // n, code % n
    deg = np.bincount(np.r_[a, b], minlength=n)
    nb = {}
    for x, y_ in zip(a.tolist(), b.tolist()):
        nb.setdefault(x, set()).add(y_); nb.setdefault(y_, set()).add(x)
    return nb


def labels(rng, lab, p):
    cx = W / 2 + np.clip(rng.normal(0, 137), -300, 300); cy = H / 2 + np.clip(rng.normal(0, 92), -230, 230)
    x0, y0 = int(cx - ROI_W / 2), int(cy - ROI_H / 2)
    n = lab.max() + 1
    area = np.bincount(lab.ravel(), minlength=n)
    edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    yy, xx = np.mgrid[0:H, 0:W]
    cx_ = ndi.mean(xx, lab, np.arange(n)); cy_ = ndi.mean(yy, lab, np.arange(n))
    inroi = (cx_ >= x0) & (cx_ < x0 + ROI_W) & (cy_ >= y0) & (cy_ < y0 + ROI_H) & (area > 0)
    inroi[edge] = False
    nb = neighbours(lab)
    ids = np.where(inroi)[0]
    a_um = area[ids] * UM_PER_PX ** 2
    cd = len(ids) / (ROI_W * ROI_H * UM_PER_PX ** 2 / 1e6)
    cv = a_um.std() / a_um.mean()
    hexf = np.mean([len(nb.get(i, ())) == 6 for i in ids])
    return dict(CD=float(cd), CV=float(cv), HEX=float(hexf), roi=[x0, y0, ROI_W, ROI_H], n_cells_roi=int(len(ids)))


def appearance(rng, lab, a, framed):
    n = lab.max() + 1
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    border = np.zeros((H, W), bool)
    border[:, 1:] |= lab[:, 1:] != lab[:, :-1]; border[1:, :] |= lab[1:, :] != lab[:-1, :]
    bw = rng.uniform(0.35, 1.1)
    dist = ndi.distance_transform_edt(~border).astype(np.float32)
    body = 1 - rng.uniform(0.45, 0.9) * np.exp(-(dist / (bw + rng.uniform(0.02, 0.1) * a)) ** 2)               # dark border, domed interior
    cell_b = rng.normal(1.0, rng.uniform(0.03, 0.12), n).astype(np.float32)
    img = body * cell_b[lab]
    img += rng.normal(0, rng.uniform(0.02, 0.08), (H, W)).astype(np.float32) * body          # granularity
    img = ndi.gaussian_filter(img, rng.uniform(0.7, 1.8))
    img = img * rng.uniform(0.35, 0.85) + rng.uniform(0.15, 0.6)                          # haze / low contrast
    for _ in range(rng.poisson(rng.uniform(0, 6))):                                          # guttae
        gx, gy, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.4, 1.5) * a
        img *= 1 - 0.7 * np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * r ** 2)))
    # illumination: tilted band + smooth random field
    th = rng.uniform(0, np.pi); u = (xx - W / 2) * np.cos(th) + (yy - H / 2) * np.sin(th)
    illum = np.exp(-(u / rng.uniform(250, 700)) ** 2) * rng.uniform(0.5, 0.9) + rng.uniform(0.15, 0.45)
    field = ndi.gaussian_filter(rng.normal(0, 1, (H // 16, W // 16)), 3)
    illum *= np.exp(0.35 * up(field / (field.std() + 1e-6)))
    if framed:
        r = np.hypot((xx - W / 2 - rng.normal(0, 60)) / (W * rng.uniform(0.48, 0.6)), (yy - H / 2) / (H * rng.uniform(0.48, 0.62)))
        r = r + 0.12 * up(ndi.gaussian_filter(rng.normal(0, 1, (H // 32, W // 32)), 2) * 3)
        illum *= np.clip(1.6 - 1.5 * r ** 3, 0, 1)
    img = img * illum
    # spatially varying defocus: blend sharp and blurred by a smooth mask
    blur = ndi.gaussian_filter(img, rng.uniform(1.5, 4.0))
    m = ndi.gaussian_filter(rng.uniform(0, 1, (H // 32, W // 32)), 2)
    m = np.clip(up((m - m.min()) / (np.ptp(m) + 1e-6)) * rng.uniform(0.3, 1.0), 0, 1)
    img = (1 - m) * img + m * blur
    img = np.clip(img, 0, None) ** rng.uniform(0.8, 1.3)
    img = img / np.percentile(img, 99.7) * rng.uniform(150, 235)
    img += rng.normal(0, rng.uniform(2, 7), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def sample(seed, cd=None, jitter=None, size_sd=None, relax=None, framed=None):
    rng = np.random.default_rng(seed)
    cd = cd or float(np.exp(rng.normal(np.log(2700), 0.19)))
    jitter = jitter if jitter is not None else rng.uniform(0.0, 0.12)
    size_sd = size_sd if size_sd is not None else rng.uniform(0.0, 0.5)
    relax = relax if relax is not None else int(rng.integers(1, 7))
    framed = bool(rng.random() < 0.48) if framed is None else framed
    lab, p, a = geometry(rng, cd, jitter, size_sd, relax)
    y = labels(rng, lab, p)
    img = appearance(rng, lab, a, framed)
    n = lab.max() + 1
    yy, xx = np.mgrid[0:H, 0:W]
    area = np.bincount(lab.ravel(), minlength=n)
    ctr = np.c_[ndi.mean(xx, lab, np.arange(n)), ndi.mean(yy, lab, np.arange(n))][area > 0]
    return img, lab, ctr, y, dict(cd_param=cd, jitter=jitter, size_sd=size_sd, relax=relax, framed=framed, spacing_px=a)


if __name__ == '__main__':
    import sys, json
    from PIL import Image
    out = sys.argv[1]
    for s in range(int(sys.argv[2]) if len(sys.argv) > 2 else 6):
        img, lab, ctr, y, prm = sample(1000 + s)
        Image.fromarray(img).resize((648, 486)).save(f'{out}/synth_{s}.png')
        print(json.dumps(dict(seed=1000 + s, **{k: round(v, 3) if isinstance(v, float) else v for k, v in {**y, **prm}.items()})))
