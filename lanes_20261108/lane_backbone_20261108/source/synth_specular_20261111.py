#!/usr/bin/env python3
"""Synthetic specular-microscopy frames with exact labels (backbone lane, 2026-11-11; coordinator, KG ds-simulated-tessellations,
bucht2011simulation, kucharski2023gan). CPU only, deterministic per frame index.

Mosaic: hexagonal lattice (spacing from a CD drawn from our label distribution, scale 1000/1296 um/px as the legible-window code),
random rotation, Gaussian jitter, a smooth density warp, random point deletion (merged large cells), 0-2 Lloyd steps.
Exact labels from the Voronoi cells whose centroid lies in the legible mask: CD = 1e6 / int(mean area in um^2), CV = area CV
(std / mean; the device CV is about 2x this, find-device-cv-twice-voronoi), HEX = fraction of six-sided cells.
Render (972 x 1296 uint8, like the raw cache): per-pixel distance to the nearest cell border -> bright cell bodies with thin dark
borders (v2; checked against real crops), v3: cell contrast, cell-to-cell brightness and edge blur set from measured real-frame statistics
(synth_appearance_stats_20261111.py) with focus varying across the frame, low-frequency shading, darker out-of-focus patches,
an optional dark out-of-focus edge band, dark spots, an optional specular glare and an optional black frame border; brightness and
contrast drawn over the P1/P2 ranges (raw mean 50-130, std 25-55). Generator parameters are fixed ranges, not tuned on OOF.

Usage:
  synth_specular_20261111.py sample OUTDIR [N]                 PNG previews + labels
  synth_specular_20261111.py gen PREFIX N [--workers K]        PREFIX_frames.u8 (N x 972 x 1296 memmap), PREFIX_labels.npz
"""
import json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import cv2
from scipy.spatial import Voronoi

H, W = 972, 1296
UM = 1000.0 / 1296.0  # um per pixel (legible_window_features_20261105.py)
SEED0 = 1111
# empirical label quantiles (rows with three positive labels): CD p1/p10/p50/p90/p99 = 1395/2247/2833/3311/3759
CD_Q = np.array([1395., 2247., 2833., 3311., 3759.]); CD_P = np.array([0.01, 0.10, 0.50, 0.90, 0.99])
# HEX label quantiles at the same probabilities (0.35/0.44/0.54/0.66/0.75); jitter that gives a target hex6 on a lattice without
# relaxation (sweep of 16 jitters x 8 mosaics, tmp/hex_sweep.py, 2026-11-11): hex6 0.864/0.760/0.676/0.586/0.487/0.447/0.414/0.370/0.346/0.333
HEX_Q = np.array([0.35, 0.44, 0.54, 0.66, 0.75])
HEX_MAP_J = np.array([0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.36]); HEX_MAP_H = np.array([0.864, 0.760, 0.676, 0.586, 0.487, 0.447, 0.414, 0.370, 0.346, 0.333])


def smooth_field(rng, shape, cells):
    """Smooth random field, unit std, correlation length ~ shape / cells."""
    g = rng.standard_normal((cells[0] + 2, cells[1] + 2)).astype(np.float32)
    f = cv2.resize(g, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return (f - f.mean()) / (f.std() + 1e-6)


def voronoi_polys(pts, box):
    vor = Voronoi(pts)
    polys, idx = [], []
    x0, y0, x1, y1 = box
    for i, r in enumerate(vor.point_region):
        reg = vor.regions[r]
        if not reg or -1 in reg:
            continue
        v = vor.vertices[reg]
        if v[:, 0].min() < x0 or v[:, 0].max() > x1 or v[:, 1].min() < y0 or v[:, 1].max() > y1:
            continue
        polys.append(v); idx.append(i)
    return polys, np.array(idx, int)


def poly_area_centroid(v):
    x, y = v[:, 0], v[:, 1]; xs, ys = np.roll(x, -1), np.roll(y, -1)
    c = x * ys - xs * y; a = 0.5 * c.sum()
    if abs(a) < 1e-9:
        return 0.0, v.mean(0)
    return abs(a), np.array([((x + xs) * c).sum() / (6 * a), ((y + ys) * c).sum() / (6 * a)])


def mosaic(rng):
    u = rng.random()
    cd = float(np.interp(u, np.r_[0, CD_P, 1], np.r_[1100., CD_Q, 4100.]))
    a_px = 1e6 / cd / UM ** 2; d = np.sqrt(2 * a_px / np.sqrt(3))
    m = 3 * d; ang = rng.uniform(0, np.pi)
    span = np.hypot(H, W) / 2 + m
    nx = int(2 * span / d) + 2; ny = int(2 * span / (d * np.sqrt(3) / 2)) + 2
    jj, ii = np.meshgrid(np.arange(nx) - nx / 2, np.arange(ny) - ny / 2)
    px = (jj + 0.5 * (ii % 2)) * d; py = ii * d * np.sqrt(3) / 2
    c, s = np.cos(ang), np.sin(ang)
    P = np.stack([c * px - s * py + W / 2, s * px + c * py + H / 2], -1).reshape(-1, 2)
    P = P[(P[:, 0] > -m) & (P[:, 0] < W + m) & (P[:, 1] > -m) & (P[:, 1] < H + m)]
    h_t = float(np.interp(rng.random(), np.r_[0, CD_P, 1], np.r_[0.30, HEX_Q, 0.80]))  # target hex6 drawn from our HEX label distribution
    jit = float(np.interp(h_t + 0.04, HEX_MAP_H[::-1], HEX_MAP_J[::-1]))  # +0.04: warp, deletion and the legible subset lower hex6 by ~0.04 (480-frame check)
    prm = dict(cd_target=cd, spacing_px=float(d), hex_target=h_t, jitter=jit, warp=float(rng.uniform(0.0, 0.3)),
               delete=float(rng.choice([0.0, 0.0, rng.uniform(0.0, 0.04)])), lloyd=0)
    # density warp: smooth displacement, correlation length ~ 4 d (local size variation, no frame-wide gradient)
    gh, gw = max(2, int((H + 2 * m) / (4 * d))), max(2, int((W + 2 * m) / (4 * d)))
    fx = smooth_field(rng, (int(H + 2 * m), int(W + 2 * m)), (gh, gw)); fy = smooth_field(rng, (int(H + 2 * m), int(W + 2 * m)), (gh, gw))
    qi = np.clip((P[:, 1] + m).astype(int), 0, fx.shape[0] - 1); qj = np.clip((P[:, 0] + m).astype(int), 0, fx.shape[1] - 1)
    P = P + prm['warp'] * d * np.stack([fx[qi, qj], fy[qi, qj]], -1)
    P = P + rng.standard_normal(P.shape) * prm['jitter'] * d
    if prm['delete'] > 0:
        P = P[rng.random(len(P)) >= prm['delete']]
    box = (-m, -m, W + m, H + m)
    for _ in range(prm['lloyd']):
        polys, idx = voronoi_polys(P, box)
        for v, i in zip(polys, idx):
            P[i] = poly_area_centroid(v)[1]
    polys, idx = voronoi_polys(P, box)
    return polys, prm


def render(i):
    rng = np.random.default_rng(np.random.SeedSequence([SEED0, i]))
    polys, prm = mosaic(rng)
    d = prm['spacing_px']
    # border map and cell id map
    border = np.zeros((H, W), np.uint8); cid = np.zeros((H, W), np.int32)
    cells = []
    for k, v in enumerate(polys):
        a, cen = poly_area_centroid(v)
        pts = np.round(v).astype(np.int32)
        cv2.fillPoly(cid, [pts], k + 1)
        cells.append((a, cen, len(v)))
    for v in polys:
        cv2.polylines(border, [np.round(v).astype(np.int32)], True, 255, 1)
    dist = cv2.distanceTransform(255 - border, cv2.DIST_L2, 3).astype(np.float32)
    r = dist / (0.5 * d)  # ~0 at the border, ~1 at the cell centre
    polarity = True  # bright cell bodies, thin dark borders (checked against real P1 and P2 crops, compare.png; v1 had a wrong inverted mode)
    bw = rng.uniform(0.06, 0.16)  # border half-width in units of the inradius (~1-2 px at CD 2800: thin dark gaps as in real frames)
    prof = 1.0 - np.exp(-(r / bw) ** 2)
    cellv = rng.normal(0, rng.uniform(0.05, 0.2), len(polys) + 1).astype(np.float32); cellv[0] = 0  # mild cell-to-cell brightness (v3: measured)
    pat = prof + cellv[cid] * prof
    # grain (sub-cell texture) and blur; focus varies across the frame (sharp and softer copies blended by a smooth mask)
    grain = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), rng.uniform(0.8, 2.5))
    pat = pat + rng.uniform(0.0, 0.3) * pat.std() * grain / (grain.std() + 1e-6)
    s1 = rng.uniform(1.2, 2.4)
    sharp = cv2.GaussianBlur(pat, (0, 0), s1); softer = cv2.GaussianBlur(pat, (0, 0), s1 + rng.uniform(0.5, 2.5))
    wf = 1.0 / (1.0 + np.exp(-1.5 * smooth_field(rng, (H, W), (2, 3)) - rng.uniform(-1, 1)))
    pat = wf * sharp + (1 - wf) * softer
    # the cell pattern is a weak modulation of the illumination (v3: relative contrast 0.03-0.13, measured on real frames)
    img = 1.0 + rng.uniform(0.03, 0.13) * (pat - pat.mean()) / (pat.std() + 1e-6)
    # low-frequency shading (smooth field + a bright central lobe + edge vignetting)
    sh = smooth_field(rng, (H, W), (2, 3)) * rng.uniform(0.1, 0.35)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = rng.uniform(0.3, 0.7) * W, rng.uniform(0.3, 0.7) * H; sx, sy = rng.uniform(0.25, 0.6) * W, rng.uniform(0.25, 0.6) * H
    lobe = np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
    base = rng.uniform(0.2, 0.7); shade = np.exp(sh) * (base + (1 - base) * lobe)
    # out-of-focus patches
    oof = np.clip((smooth_field(rng, (H, W), (3, 4)) - rng.uniform(0.6, 2.2)) / 1.2, 0, 1); oof = cv2.GaussianBlur(oof, (0, 0), 25)
    soft = cv2.GaussianBlur(img, (0, 0), rng.uniform(4, 9)) * rng.uniform(0.6, 0.9)  # out of focus = blurred and darker
    img = img * (1 - oof) + soft * oof
    # optional dark out-of-focus band along one edge (as the dark top/bottom bands of real frames)
    band = np.zeros((H, W), np.float32)
    if rng.random() < 0.4:
        side = int(rng.integers(0, 4)); wfr = rng.uniform(0.08, 0.25)
        coord = [yy / H, 1 - yy / H, xx / W, 1 - xx / W][side]
        band = np.clip((wfr - coord) / (0.5 * wfr), 0, 1); band = band * band * (3 - 2 * band)
        img = img * (1 - band) + cv2.GaussianBlur(img, (0, 0), 8) * band * rng.uniform(0.2, 0.5)
    img = img * shade
    # dark spots (guttae-like) and optional glare
    for _ in range(int(rng.integers(0, 10))):
        cv2.circle(img, (int(rng.uniform(0, W)), int(rng.uniform(0, H))), int(rng.uniform(4, 14)), float(img.mean() * rng.uniform(0.2, 0.6)), -1)
    if rng.random() < 0.3:  # specular reflex: bright elongated glare
        g = np.exp(-0.5 * (((xx - rng.uniform(0, W)) / rng.uniform(20, 120)) ** 2 + ((yy - rng.uniform(0, H)) / rng.uniform(8, 40)) ** 2))
        img = img + g * img.max() * rng.uniform(0.5, 1.5)
    # brightness / contrast over the P1-P2 range, noise
    img = img / (img.mean() + 1e-6) * rng.uniform(50, 130)  # v3: scale only, so the relative cell contrast set above is kept
    img = img + rng.standard_normal((H, W)).astype(np.float32) * rng.uniform(1.0, 3.5)
    # optional black frame border (P1 frames: median 10 % black)
    black = np.zeros((H, W), bool)
    if rng.random() < 0.5:
        edge = smooth_field(rng, (H, W), (2, 3)) * rng.uniform(2, 10)
        mx, my = rng.uniform(0, 0.08) * W, rng.uniform(0, 0.12) * H
        black = (xx < mx + edge) | (xx > W - mx + edge) | (yy < my + edge) | (yy > H - my + edge)
        bm = cv2.GaussianBlur(black.astype(np.float32), (0, 0), 3)
        img = img * (1 - bm) + rng.uniform(0, 12) * bm
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    # legible mask: in focus, well lit, not black
    legible = (oof < 0.2) & (band < 0.2) & (shade > np.quantile(shade, rng.uniform(0.25, 0.5))) & ~black
    legible = cv2.erode(legible.astype(np.uint8), np.ones((int(d), int(d)), np.uint8)) > 0
    A, S = [], []
    for k, (a, cen, ns) in enumerate(cells):
        x, y = int(cen[0]), int(cen[1])
        if 0 <= x < W and 0 <= y < H and legible[y, x]:
            A.append(a * UM * UM); S.append(ns)
    if len(A) < 30:  # too little legible area: label from all cells inside the frame
        A = [a * UM * UM for a, cen, ns in cells if 0 <= cen[0] < W and 0 <= cen[1] < H]; S = [ns for a, cen, ns in cells if 0 <= cen[0] < W and 0 <= cen[1] < H]
        prm['label_fallback'] = 1
    A, S = np.array(A), np.array(S)
    lab = np.array([1e6 / int(A.mean()), A.std() / A.mean(), (S == 6).mean()], np.float32)
    prm.update(n_label_cells=int(len(A)), polarity=int(polarity), legible_frac=float(legible.mean()), black_frac=float(black.mean()))
    return img, lab, prm


def _gen_one(args):
    i, path, n = args
    img, lab, prm = render(i)
    mm = np.memmap(path, mode='r+', dtype=np.uint8, shape=(n, H, W)); mm[i] = img; mm.flush(); del mm
    return i, lab, prm


def main():
    cmd = sys.argv[1]
    if cmd == 'sample':
        out = sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 6; os.makedirs(out, exist_ok=True); rows = []
        for i in range(n):
            t = time.time(); img, lab, prm = render(i); dt = time.time() - t
            cv2.imwrite(f'{out}/synth_{i:02d}.png', cv2.resize(img, (648, 486), interpolation=cv2.INTER_AREA)); cv2.imwrite(f'{out}/synth_{i:02d}_crop.png', img[300:556, 400:656])
            rows.append(dict(i=i, CD=round(float(lab[0]), 1), CV=round(float(lab[1]), 4), HEX=round(float(lab[2]), 4), sec=round(dt, 3), mean=round(float(img.mean()), 1), std=round(float(img.std()), 1), **prm))
            print(json.dumps(rows[-1]), flush=True)
        return
    if cmd == 'gen':
        prefix, n = sys.argv[2], int(sys.argv[3]); k = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 8
        path = prefix + '_frames.u8'
        np.memmap(path, mode='w+', dtype=np.uint8, shape=(n, H, W)).flush()
        labs = np.zeros((n, 3), np.float32); prms = [None] * n; t0 = time.time()
        import multiprocessing as mp
        with mp.get_context('fork').Pool(k) as pool:
            for c, (i, lab, prm) in enumerate(pool.imap_unordered(_gen_one, [(i, path, n) for i in range(n)], chunksize=32)):
                labs[i] = lab; prms[i] = prm
                if (c + 1) % 2000 == 0:
                    print(json.dumps(dict(done=c + 1, n=n, sec=round(time.time() - t0))), flush=True)
        np.savez_compressed(prefix + '_labels.npz', labels=labs, params=np.array([json.dumps(p) for p in prms]), shape=np.array([n, H, W]),
                            label_names=np.array(['CD', 'CV_area', 'HEX']), seed0=SEED0)
        q = lambda j: np.percentile(labs[:, j], [1, 10, 50, 90, 99]).round(4).tolist()
        print(json.dumps(dict(done=n, sec=round(time.time() - t0), CD=q(0), CV_area=q(1), HEX=q(2))), flush=True)


if __name__ == '__main__':
    main()
