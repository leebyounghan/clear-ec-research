"""Does the technician pick the clearest patch (2026-11-05)? E9 asked whether labels can pick the ROI (they cannot), and nobody asked whether the image can: a technician marks cells where they can see them. For the 25 expert sets: per-window legibility maps, no labels, no training.
Candidates scored over box-sized windows (538 x 408 px, stride 32) and marker-hull-sized windows (380 x 300 px): ridge strength; ridge contrast (ridge p80 - p20); local contrast; high-frequency energy; pre-only peak count at 0.5; mean peak confidence; ring height of the tile spectrum; regularity of peak spacing (1 / CV of NN distance); brightness; centrality. Rank of the window nearest the expert region among all windows, IoU of the top window with the expert region vs mean IoU. Gate: median rank <= 0.15 with top-window IoU clearly above random. E1 also reads hex6 and CV inside the top window on the 25 against the device values, next to whole-frame."""
import json, os, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import cv2
import numpy as np
from scipy.spatial import Delaunay, Voronoi, cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abs_tess_20261104 as T
H, W = T.H, T.W
ann = json.load(open(f'{T.ROOT}/datasets/clear_ec_official/green_eval/annotations.json'))
with np.load(f'{T.ROOT}/datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True) as z: ids = z['ids'].astype(str)
pos = {n: i for i, n in enumerate(ids)}; PK = T.load_peaks()
def integral(a): return cv2.integral(a.astype(np.float64))
def box_mean(I, x0, y0, x1, y1): return (I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]) / max((x1 - x0) * (y1 - y0), 1)
def iou(a, c):
    ix = max(0, min(a[2], c[2]) - max(a[0], c[0])); iy = max(0, min(a[3], c[3]) - max(a[1], c[1])); i = ix * iy; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (c[2] - c[0]) * (c[3] - c[1]) - i)
def maps(a, pk):
    q = pk[pk[:, 2] >= 0.35, :2]; s = float(np.median(cKDTree(q).query(q, k=2)[0][:, 1])) if len(q) > 30 else 24.0; R = T.border_map(a, s); f = a.astype(np.float32); u = cv2.createCLAHE(2.0, (W // 32, H // 32)).apply(a).astype(np.float32); bg = cv2.GaussianBlur(f, (0, 0), 1.5 * s); con = np.abs(f - bg) / (cv2.GaussianBlur(f, (0, 0), 3 * s) + 5); lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(f, (0, 0), 1.0), cv2.CV_32F)) / (cv2.GaussianBlur(f, (0, 0), 3 * s) + 5)
    P5 = np.zeros((H, W), np.float64); P = np.zeros((H, W), np.float64); hi = pk[pk[:, 2] >= 0.5]
    for x, y_, p in hi: P5[min(int(y_), H - 1), min(int(x), W - 1)] += 1
    for x, y_, p in pk: P[min(int(y_), H - 1), min(int(x), W - 1)] += p
    # regularity: inverse local deviation of NN distance
    reg = np.zeros((H, W), np.float64); ct = np.zeros((H, W), np.float64)
    if len(q) > 30:
        d = cKDTree(q).query(q, k=4)[0][:, 1:]; dev = np.abs(d.mean(1) - np.median(d.mean(1))) / np.median(d.mean(1))
        for (x, y_), v in zip(q, dev): reg[min(int(y_), H - 1), min(int(x), W - 1)] += v; ct[min(int(y_), H - 1), min(int(x), W - 1)] += 1
    return dict(ridge=R, contrast=con, hf=lap, bright=f, peaks5=P5, conf=P), reg, ct, s
def task(n):
    e = ann[n]; a = np.asarray(np.memmap(T.RAW, dtype=np.uint8, mode='r', shape=(8954, H, W))[pos[n]]); pk = PK[pos[n]]; M, reg, ct, s = maps(a, pk); I = {k: integral(v) for k, v in M.items()}; Ir, Ic = integral(reg), integral(ct); ex = e['roi_box_mha']; m = np.asarray(e['markers_mha'], float); hull = [m[:, 0].min(), m[:, 1].min(), m[:, 0].max(), m[:, 1].max()]; out = {}
    for nm, (w, h, tgt) in dict(box=(538, 408, ex), hull=(380, 300, hull)).items():
        wins = [(x, y_, x + w, y_ + h) for y_ in range(0, H - h + 1, 32) for x in range(0, W - w + 1, 32)]; IoU = np.array([iou(c, tgt) for c in wins]); near = int(np.argmax(IoU)); sc = {}
        for k in I: sc[k] = np.array([box_mean(I[k], *c) for c in wins])
        sc['regular'] = -np.array([box_mean(Ir, *c) / (box_mean(Ic, *c) + 1e-9) for c in wins]); sc['central'] = -np.array([np.hypot((c[0] + c[2]) / 2 - W / 2, (c[1] + c[3]) / 2 - H / 2) for c in wins]); sc['dark'] = -sc['bright']
        o = {}
        for k, v in sc.items():
            top = int(np.argmax(v)); o[k] = (float((v > v[near]).mean()), float(IoU[top]))
        o['_random_iou'] = float(IoU.mean()); o['_near_iou'] = float(IoU[near]); out[nm] = o
    return n, out
if __name__ == '__main__':
    workers = int(min(8, max(2, 16 - os.getloadavg()[0] - 1)))
    with get_context('fork').Pool(workers) as pool: res = pool.map(task, list(ann))
    rep = {}
    for nm in ('box', 'hull'):
        keys = [k for k in res[0][1][nm] if not k.startswith('_')]; rd = float(np.mean([r[1][nm]['_random_iou'] for r in res])); o = {k: dict(median_rank=round(float(np.median([r[1][nm][k][0] for r in res])), 3), share_top20=round(float(np.mean([r[1][nm][k][0] < 0.2 for r in res])), 2), iou_top=round(float(np.mean([r[1][nm][k][1] for r in res])), 3)) for k in keys}; rep[nm] = dict(random_iou=round(rd, 3), near_iou=round(float(np.mean([r[1][nm]['_near_iou'] for r in res])), 3), scores=o); print(nm, 'random IoU', round(rd, 3)); [print('  ', k, json.dumps(v)) for k, v in sorted(o.items(), key=lambda kv: kv[1]['median_rank'])]
    json.dump(rep, open('/workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep/legibility_roi.json', 'w'), indent=1); print('done', flush=True)
