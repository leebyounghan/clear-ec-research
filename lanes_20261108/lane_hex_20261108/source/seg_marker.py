"""Does hex6 inside the technician's box track the HEX label better than hex6 over the frame? (HEX lane, 2026-11-08)
25 expert-marker frames: Cellpose-SAM loose+CLAHE masks (CPU), per-cell neighbours (R=3) and interior flag; hex6 over the
whole frame, inside the true box (538x408), inside predicted boxes (box with most interior cells, most cells, highest mask
coverage), and marker hex6. usage: seg_marker.py [masks_npz_to_reuse]"""
import sys, os, json
import numpy as np
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'; sys.path.insert(0, f'{LANE}/source')
os.chdir('/workspace/clear-ec-research')
from scipy import ndimage as ndi
from scipy.spatial import Delaunay
import hexlib as L
ann = json.load(open('/workspace/clear-ec-research/datasets/clear_ec_official/green_eval/annotations.json'))
ids = [str(q) for q in L.BP.ids]; pos = {i: r for r, i in enumerate(ids)}
keys = [k for k in ann if k in pos]; FR = np.array([int(L.idx[pos[k]]) for k in keys])
OUTM = f'{LANE}/feats/seg_marker_masks.npz'
if len(sys.argv) > 1 and os.path.exists(OUTM):
    MS = np.load(OUTM)['masks']
else:
    import cv2
    from cellpose import models
    MM = np.memmap('/workspace/clear-ec-research/datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
    model = models.CellposeModel(gpu=False, pretrained_model='cpsam_models/cpsam_v2'); cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    MS = np.zeros((len(FR), 972, 1296), np.uint16)
    for k, f in enumerate(FR):
        MS[k] = np.asarray(model.eval([cl.apply(np.asarray(MM[f]))], batch_size=8, flow_threshold=0.8, cellprob_threshold=-2.0)[0][0]); print('seg', k, flush=True)
    np.savez_compressed(OUTM, frames=FR, keys=np.array(keys), masks=MS)
R = 3; OFFS = [(dy, dx) for dy in range(-R, R + 1) for dx in range(0, R + 1) if (dx > 0 or dy > 0) and dy * dy + dx * dx <= R * R]


def percell(M):
    M = M.astype(np.int32); H, W = M.shape; n = int(M.max()); pairs = []
    for dy, dx in OFFS:
        y0, y1 = max(0, -dy), H - max(0, dy); a = M[y0:y1, 0:W - dx]; b = M[y0 + dy:y1 + dy, dx:W]
        k = (a > 0) & (b > 0) & (a != b); pairs.append(np.c_[a[k], b[k]])
    P = np.unique(np.sort(np.concatenate(pairs), 1), axis=0); nb = np.bincount(P.ravel(), minlength=n + 1)
    area = np.bincount(M.ravel(), minlength=n + 1).astype(float); edge = np.unique(np.r_[M[0], M[-1], M[:, 0], M[:, -1]])
    dil = ndi.grey_dilation(M, size=(2 * R + 1, 2 * R + 1)); ring = (M == 0) & (dil > 0)
    bg = np.bincount(dil[ring], minlength=n + 1).astype(float); per = np.bincount(dil[(dil > 0) & (dil != M)], minlength=n + 1).astype(float)
    ok = np.ones(n + 1, bool); ok[0] = False; ok[edge] = False; ok &= bg / np.maximum(per, 1) < 0.25; ok &= area > 30
    lab = np.arange(n + 1); cy = np.nan_to_num(ndi.mean(np.indices(M.shape)[0], M, lab)); cx = np.nan_to_num(ndi.mean(np.indices(M.shape)[1], M, lab))
    return nb, ok, cx, cy, area


def inbox(cx, cy, b): return (cx >= b[0]) & (cx < b[2]) & (cy >= b[1]) & (cy < b[3])


def h6(nb, sel): return float(np.mean(nb[sel] == 6)) if sel.sum() >= 5 else np.nan


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


BW, BH = 538, 408; rows = []
for k, key in enumerate(keys):
    a = ann[key]; b = a['roi_box_mha']; M = MS[k]; nb, ok, cx, cy, area = percell(M)
    mk = np.array(a['markers_mha'], float)[:, :2]; hull = Delaunay(mk); inh = hull.find_simplex(np.c_[cx, cy]) >= 0
    cand = [(x, y, x + BW, y + BH) for x in range(0, 1296 - BW + 1, 16) for y in range(0, 972 - BH + 1, 16)]
    cov = M > 0; II = np.pad(cov.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    sc_int = [np.sum(ok & inbox(cx, cy, c)) for c in cand]; sc_all = [np.sum((area > 30) & inbox(cx, cy, c)) for c in cand]
    sc_cov = [II[c[3], c[2]] - II[c[1], c[2]] - II[c[3], c[0]] + II[c[1], c[0]] for c in cand]
    bi, ba, bc = cand[int(np.argmax(sc_int))], cand[int(np.argmax(sc_all))], cand[int(np.argmax(sc_cov))]
    hx = a['gt']['HEX'] if isinstance(a['gt'], dict) else float(eval(a['gt'])['HEX'])
    from scipy.spatial import cKDTree
    rows.append(dict(key=key, label=hx, whole=h6(nb, ok), box=h6(nb, ok & inbox(cx, cy, b)), hull=h6(nb, ok & inh), pred_int=h6(nb, ok & inbox(cx, cy, bi)),
                     pred_all=h6(nb, ok & inbox(cx, cy, ba)), pred_cov=h6(nb, ok & inbox(cx, cy, bc)), n_box=int(np.sum(ok & inbox(cx, cy, b))), n_whole=int(ok.sum()),
                     iou_int=iou(bi, b), iou_all=iou(ba, b), iou_cov=iou(bc, b)))
lab = np.array([r['label'] for r in rows])
print('frames', len(rows), 'median interior cells: whole', np.median([r['n_whole'] for r in rows]), 'box', np.median([r['n_box'] for r in rows]))
for c in ('whole', 'box', 'hull', 'pred_int', 'pred_all', 'pred_cov'):
    v = np.array([r[c] for r in rows]); m = np.isfinite(v)
    s = np.sum(v[m] * lab[m]) / np.sum(v[m] ** 2)   # one scale, as the stretch would fit
    print('%-9s n %2d r %.3f  MAPE after one scale %.2f  bias %+.3f' % (c, m.sum(), np.corrcoef(v[m], lab[m])[0, 1], np.mean(np.abs(s * v[m] - lab[m]) / lab[m]) * 100, np.mean(v[m] - lab[m])))
print('IoU with true box: interior-count box %.2f, all-cell box %.2f, coverage box %.2f' % tuple(np.mean([r[q] for r in rows]) for q in ('iou_int', 'iou_all', 'iou_cov')))
