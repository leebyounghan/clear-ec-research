"""Green day 3 / E1 (pre-registered 2026-11-03): CPSAM segmentation as an independent per-cell source.
  segment --model {v2,gold26,pseudo25} --rows-from A --rows-to B [--limit N]
      Cellpose-SAM on full frames (CLAHE as the regression pipeline, min-max to uint8). Per image stores per-cell
      centroid, area (px^2), mask-adjacency neighbour count (after 2-px label expansion), border flag, organiser
      hexagonality per cell (IoU with the fitted hexagon, verbatim organiser logic via p10) and the cell count.
      Chunked npz every 200 images under experiments/green_day3_20261103/e1_cpsam/<model>/.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import cv2
import torch

ROOT = Path('experiments/green_day3_20261103'); OUT = ROOT/'e1_cpsam'
RAW = Path('datasets/clear_ec_fullframe_20260904/img_full_raw.dat'); N_TOTAL, HGT, WID = 8954, 972, 1296
MODELS = {'v2': 'cpsam_models/cpsam_v2', 'gold26': 'datasets/clear_ec_official/models/cpsam_gold26', 'pseudo25': 'datasets/clear_ec_official/models/cpsam_pseudo25'}


def log(msg, **kw):
    print(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), msg=msg, **kw)), flush=True)


def prep(a):
    c = cv2.createCLAHE(2.0, (WID//32, HGT//32)).apply(np.ascontiguousarray(a))
    lo, hi = np.percentile(c, (1, 99))
    return np.clip((c.astype(np.float32)-lo)/(hi-lo+1e-6)*255, 0, 255).astype(np.uint8)


def organiser_hexagonality_per_cell(masks, stats):
    """Verbatim organiser logic (calculate_hexagonality), returned per cell instead of averaged."""
    from skimage.draw import polygon as skpoly
    out = np.zeros(len(stats), dtype=np.float32)
    for i, cell in enumerate(stats):
        min_row, min_col, max_row, max_col = cell.bbox
        msc = masks[min_row:max_row, min_col:max_col].astype(np.float32)
        msc[msc != cell.label] = 0.2; msc[msc == cell.label] = 1.0
        Ma = cell.axis_major_length*0.8; mi = cell.axis_minor_length*0.8; ang = cell.orientation
        hp = np.array([[Ma/2, 0], [Ma/4, -mi/2], [-Ma/4, -mi/2], [-Ma/2, 0], [-Ma/4, mi/2], [Ma/4, mi/2]])
        R = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]]); rh = hp@R
        hm = np.zeros(msc.shape); cy, cx = cell.centroid
        th = rh+np.array([np.ceil(cy-min_row), np.ceil(cx-min_col)])
        rr, cc = skpoly(th[:, 0], th[:, 1], hm.shape); hm[rr, cc] = 1
        inter = np.logical_and(msc == 1.0, hm).sum(); uni = np.logical_or(msc == 1.0, hm).sum()
        out[i] = inter/uni if uni > 0 else 0
    return out


def per_cell(m, hexfun=None):
    """Per-cell stats from a label mask (0 = background)."""
    from skimage.measure import regionprops
    from skimage.segmentation import expand_labels
    n = int(m.max())
    if n == 0:
        return dict(n=0)
    props = regionprops(m)
    lab = np.array([p.label for p in props]); area = np.array([p.area for p in props], dtype=np.float32)
    cen = np.array([p.centroid for p in props], dtype=np.float32)[:, ::-1]        # (x, y)
    bb = np.array([p.bbox for p in props]); border = (bb[:, 0] <= 0) | (bb[:, 1] <= 0) | (bb[:, 2] >= HGT) | (bb[:, 3] >= WID)
    e = expand_labels(m, 2)
    nb = [set() for _ in range(n+1)]
    for ax, sh in ((0, 1), (1, 1)):
        r = np.roll(e, sh, axis=ax); d = (e != r) & (e > 0) & (r > 0)
        for a_, b_ in zip(e[d].ravel(), r[d].ravel()):
            nb[a_].add(int(b_)); nb[b_].add(int(a_))
    sides = np.array([len(nb[l]) for l in lab], dtype=np.int16)
    hexv = organiser_hexagonality_per_cell(m, props)
    return dict(n=n, label=lab.astype(np.int32), centroid=cen, area=area, sides=sides, border=border, hexagonality=hexv)


def segment(a):
    from cellpose import models
    hexfun = None
    torch.set_num_threads(4)
    out = OUT/a.model; out.mkdir(parents=True, exist_ok=True)
    mdl = models.CellposeModel(gpu=True, pretrained_model=MODELS[a.model])
    mm = np.memmap(RAW, dtype=np.uint8, mode='r', shape=(N_TOTAL, HGT, WID))
    rows = np.arange(a.rows_from, min(a.rows_to, N_TOTAL))
    if a.limit:
        rows = rows[:a.limit]
    t0 = time.time(); chunk = {}; done = 0
    for k, r in enumerate(rows):
        c0 = (int(r)//200)*200
        path = out/f'rows_{c0:05d}.npz'
        if path.exists() and not a.limit:
            continue
        img = prep(mm[r])
        m, flows, styles = mdl.eval(img, batch_size=32, normalize=True, flow_threshold=a.flow_threshold, cellprob_threshold=a.cellprob_threshold)
        m = np.asarray(m).astype(np.int32)
        st = per_cell(m, hexfun); st['row'] = int(r); st['seconds'] = round(time.time()-t0, 1)
        chunk[int(r)] = st; done += 1
        last_of_chunk = (k+1 == len(rows)) or (int(rows[k+1])//200)*200 != c0
        if last_of_chunk:
            np.savez_compressed(path if not a.limit else out/f'smoke_rows_{c0:05d}.npz', rows=np.array(sorted(chunk)), stats=np.array([chunk[i] for i in sorted(chunk)], dtype=object))
            chunk = {}
        if done % 50 == 0 or done == 1:
            log('segment', model=a.model, done=done, of=len(rows), row=int(r), cells=st['n'], seconds=round(time.time()-t0, 1), sec_per_image=round((time.time()-t0)/done, 2))
    log('segment done', model=a.model, done=done, seconds=round(time.time()-t0, 1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['segment']); ap.add_argument('--model', default='v2', choices=list(MODELS))
    ap.add_argument('--rows-from', type=int, default=0); ap.add_argument('--rows-to', type=int, default=N_TOTAL); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--flow-threshold', type=float, default=0.4); ap.add_argument('--cellprob-threshold', type=float, default=0.0)
    a = ap.parse_args(); segment(a)
