"""Renderer v4 (2026-11-04). A real-trained detector on v3 gave miss 0.28 / FP 0.017 / jitter 0.098 against real
0.17 / 0.17 / 0.22; it ignored ~90 % of the small confusers and lost 46 % of the darkest third of cells. Changes: the floor of
border contrast goes up and its top is capped (no black dashes); confusers become cell-sized (centre-surround bright blobs and
faint rings of about one cell interior, on borders and inside cells); shapes get real variance, since merged neighbour pairs
(oversized, elongated cells) make a detector fire twice inside one cell and an optional anisotropic stretch elongates a whole
frame; stronger interior asymmetry for jitter. Geometry changes, so centres, instance map and ROI labels are recomputed and
saved per image in <out>/meta. Annotated frames never serve as the illumination source.
usage: synth_render_v4_20261103.py IN_DIR OUT_DIR N [SEED0]. Forked pool sized clamp(16 - load1 - 1, 2, 8).
"""
import json, os, re, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import cv2
import numpy as np
from scipy import ndimage as ndi
cv2.setNumThreads(0)
H, W = 972, 1296
RAW = '/workspace/clear-ec-research/datasets/clear_ec_fullframe_20260904/img_full_raw.dat'
TUNE = dict(border=(0.04, 0.34), tex=(0.012, 0.05), blob_n=(5, 70), gut_n=(0, 8), folds=(0, 3), defocus=(1.0, 6.5), noise=(1, 3.5), mean=(65, 140))
SRC = None


def gf(a, s):
    return cv2.GaussianBlur(a, (0, 0), s, borderType=cv2.BORDER_REFLECT)


def smooth(rng, s):
    """Unit-range smooth random field with correlation length ~s px."""
    h, w = max(4, int(H / s)), max(4, int(W / s))
    f = cv2.resize(rng.normal(0, 1, (h, w)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    return (f - f.min()) / (np.ptp(f) + 1e-6)


def illumination(rng, framed):
    idx = SRC['framed'] if framed else SRC['plain']
    mm = np.memmap(RAW, dtype=np.uint8, mode='r', shape=(8954, H, W))
    a = np.asarray(mm[int(rng.choice(idx))], dtype=np.float32)
    if rng.random() < 0.5:
        a = a[:, ::-1]
    if rng.random() < 0.5:
        a = a[::-1]
    low = gf(a, rng.uniform(22, 38))
    return low / max(float(np.percentile(low, 98)), 1.0)


def reshape_cells(rng, lab):
    """Optional whole-frame anisotropic stretch, then merge random neighbour pairs into oversized elongated cells."""
    info = dict(aspect=1.0, merged=0)
    if rng.random() < 0.5:
        asp, t = rng.uniform(1.05, 1.4), rng.uniform(0, np.pi)
        c, s_ = np.cos(t), np.sin(t)
        A = np.array([[c, -s_], [s_, c]]) @ np.diag([asp, 1 / asp]) @ np.array([[c, s_], [-s_, c]])
        M = np.c_[A, np.array([W / 2, H / 2]) - A @ np.array([W / 2, H / 2])].astype(np.float32)
        lab = cv2.warpAffine(lab.astype(np.float32), M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT).astype(np.int64)
        info['aspect'] = float(asp)
    n = int(lab.max()) + 1
    u = np.r_[lab[:, :-1].ravel(), lab[:-1, :].ravel()]; v = np.r_[lab[:, 1:].ravel(), lab[1:, :].ravel()]
    m = u != v
    code = np.unique(np.minimum(u[m], v[m]) * n + np.maximum(u[m], v[m]))
    a_, b_ = code // n, code % n
    order = rng.permutation(len(code))
    used = np.zeros(n, bool); mapping = np.arange(n); want = int(len(np.unique(lab)) * rng.uniform(0.02, 0.08)); k = 0
    for i in order:
        if k >= want:
            break
        if not used[a_[i]] and not used[b_[i]]:
            mapping[b_[i]] = a_[i]; used[a_[i]] = used[b_[i]] = True; k += 1
    info['merged'] = k
    return mapping[lab], info


def centres_and_labels(lab, roi):
    n = int(lab.max()) + 1
    yy, xx = np.mgrid[0:H, 0:W]
    area = np.bincount(lab.ravel(), minlength=n).astype(np.float64)
    live = area > 0; safe = np.where(live, area, 1)
    cx = np.bincount(lab.ravel(), xx.ravel(), n) / safe; cy = np.bincount(lab.ravel(), yy.ravel(), n) / safe
    edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    u = np.r_[lab[:, :-1].ravel(), lab[:-1, :].ravel()]; v = np.r_[lab[:, 1:].ravel(), lab[1:, :].ravel()]
    m = u != v
    code = np.unique(np.minimum(u[m], v[m]) * n + np.maximum(u[m], v[m]))
    deg = np.bincount(np.r_[code // n, code % n], minlength=n)
    x0, y0, w, h = [int(t) for t in roi]
    ok = live & (cx >= x0) & (cx < x0 + w) & (cy >= y0) & (cy < y0 + h); ok[edge] = False
    a_um = area[ok] * 0.77 ** 2
    y = [ok.sum() / (w * h * 0.77 ** 2 / 1e6), a_um.std() / a_um.mean(), (deg[ok] == 6).mean()] if ok.sum() > 2 else [np.nan] * 3
    return np.c_[cx[live], cy[live]].astype(np.float32), np.array(y)


def impulses(pts, s, amp):
    """Gaussian blobs of peak height `amp` (array or scalar) at integer (x, y) points."""
    f = np.zeros((H, W), np.float32)
    if len(pts):
        x = np.clip(pts[:, 0].astype(int), 0, W - 1); y = np.clip(pts[:, 1].astype(int), 0, H - 1)
        np.add.at(f, (y, x), amp)
    return gf(f, s) * (2 * np.pi * s ** 2)


def appearance(rng, lab, a, framed):
    n = int(lab.max()) + 1
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    area = np.bincount(lab.ravel(), minlength=n).astype(np.float32)
    live = np.where(area > 0)[0]; area[area == 0] = 1
    cx = np.bincount(lab.ravel(), xx.ravel(), n).astype(np.float32) / area
    cy = np.bincount(lab.ravel(), yy.ravel(), n).astype(np.float32) / area
    # per-edge border depth and width; some edges nearly vanish
    code = np.zeros((H, W), np.int64)
    m = lab[:, 1:] != lab[:, :-1]
    code[:, 1:][m] = (np.minimum(lab[:, 1:], lab[:, :-1]) * n + np.maximum(lab[:, 1:], lab[:, :-1]) + 1)[m]
    m = lab[1:, :] != lab[:-1, :]
    code[1:, :][m] = (np.minimum(lab[1:, :], lab[:-1, :]) * n + np.maximum(lab[1:, :], lab[:-1, :]) + 1)[m]
    bmask = code > 0
    uniq, inv = np.unique(code[bmask], return_inverse=True)
    e_depth = np.clip(rng.lognormal(0, rng.uniform(0.12, 0.3), len(uniq)), 0.5, 1.5).astype(np.float32)
    e_depth[rng.random(len(uniq)) < rng.uniform(0.02, 0.08)] *= rng.uniform(0.1, 0.4)
    e_width = rng.lognormal(0, rng.uniform(0.15, 0.4), len(uniq)).astype(np.float32)
    dist, near = cv2.distanceTransformWithLabels((~bmask).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    d_by, w_by = np.ones(near.max() + 1, np.float32), np.ones(near.max() + 1, np.float32)
    d_by[near[bmask]] = e_depth[inv]; w_by[near[bmask]] = e_width[inv]
    lo, hi = rng.uniform(0.09, 0.16), rng.uniform(0.16, 0.27)
    depth = np.minimum((lo + (hi - lo) * smooth(rng, rng.uniform(100, 300))) * d_by[near], 0.38)
    width = (rng.uniform(0.8, 2.0) + rng.uniform(0.02, 0.09) * a) * w_by[near]
    img = 1 - np.clip(depth, 0, 0.95) * np.exp(-(dist / width) ** 2)
    # asymmetric bodies: per-cell brightness, shading gradient, off-centre dome
    img *= rng.normal(1.0, rng.uniform(0.008, 0.05), n).astype(np.float32)[lab]
    th = rng.uniform(0, 2 * np.pi, n); g = rng.uniform(0, rng.uniform(0.12, 0.42), n)
    gx, gy = (g * np.cos(th)).astype(np.float32), (g * np.sin(th)).astype(np.float32)
    img *= 1 + (gx[lab] * (xx - cx[lab]) + gy[lab] * (yy - cy[lab])) / a
    off = rng.uniform(0.2, 0.6, n) * a; ph = rng.uniform(0, 2 * np.pi, n)
    pick = live[rng.random(len(live)) < rng.uniform(0.3, 0.7)]
    img += impulses(np.c_[cx[pick] + off[pick] * np.cos(ph[pick]), cy[pick] + off[pick] * np.sin(ph[pick])], rng.uniform(0.15, 0.3) * a, rng.uniform(0.05, 0.18))
    # sub-cell texture
    tex = gf(rng.normal(0, 1, (H, W)).astype(np.float32), rng.uniform(1.2, 3.5)); tex /= tex.std() + 1e-6
    tex2 = gf(rng.normal(0, 1, (H, W)).astype(np.float32), rng.uniform(4, 9)); tex2 /= tex2.std() + 1e-6
    img += rng.uniform(*TUNE['tex']) * tex + rng.uniform(0.008, 0.035) * tex2
    # confusers, ~0.1-0.2 per cell in total
    junction = (lab[:-1, :-1] != lab[1:, :-1]).astype(int) + (lab[:-1, :-1] != lab[:-1, 1:]) + (lab[1:, 1:] != lab[1:, :-1]) + (lab[1:, 1:] != lab[:-1, 1:]) >= 3
    jy, jx = np.where(junction)
    if len(jx):
        k = rng.choice(len(jx), int(len(live) * rng.uniform(0.03, 0.08)), replace=False)
        img += impulses(np.c_[jx[k], jy[k]], rng.uniform(2.0, 4.5), rng.uniform(0.1, 0.5, len(k)).astype(np.float32))
    pick = live[rng.random(len(live)) < rng.uniform(0.03, 0.08)]
    img += impulses(np.c_[cx[pick] + off[pick] * np.sin(ph[pick]), cy[pick] - off[pick] * np.cos(ph[pick])], rng.uniform(2.5, 5.5), rng.uniform(0.1, 0.5, len(pick)).astype(np.float32))
    # cell-sized confusers: centre-surround bright blobs and faint rings of about one cell interior
    sb = rng.uniform(0.17, 0.27) * a
    jpts = np.c_[jx, jy] if len(jx) else np.zeros((0, 2))
    bpick = live[rng.random(len(live)) < rng.uniform(0.04, 0.1)]
    pts = np.r_[np.c_[cx[bpick] + off[bpick] * np.cos(ph[bpick] + 2), cy[bpick] + off[bpick] * np.sin(ph[bpick] + 2)],
                jpts[rng.choice(len(jpts), min(len(jpts), int(len(live) * rng.uniform(0.04, 0.1))), replace=False)] if len(jpts) else np.zeros((0, 2))]
    amp = rng.uniform(0.1, 0.32, len(pts)).astype(np.float32)
    img += impulses(pts, sb, amp) - 0.4 * impulses(pts, 1.7 * sb, amp)
    ring = np.zeros((H, W), np.float32)
    for i in live[rng.random(len(live)) < rng.uniform(0.03, 0.09)]:
        r = max(3, int(rng.uniform(0.22, 0.42) * a))
        cv2.circle(ring, (int(cx[i] + off[i] * np.cos(ph[i] - 1)), int(cy[i] + off[i] * np.sin(ph[i] - 1))), r, 1.0, 1, cv2.LINE_AA)
    img *= 1 - rng.uniform(0.06, 0.2) * gf(ring, 1.0) * 2
    chord = np.zeros((H, W), np.float32)
    for i in live[rng.random(len(live)) < rng.uniform(0.02, 0.06)]:
        t = rng.uniform(0, np.pi); r = 0.35 * a
        cv2.line(chord, (int(cx[i] - r * np.cos(t)), int(cy[i] - r * np.sin(t))), (int(cx[i] + r * np.cos(t)), int(cy[i] + r * np.sin(t))), 1.0, 1, cv2.LINE_AA)
    img *= 1 - rng.uniform(0.05, 0.2) * gf(chord, 1.1) * 2
    gut, rim = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    for _ in range(int(len(live) * rng.uniform(0.001, 0.006))):
        c = (int(rng.integers(W)), int(rng.integers(H))); r = int(rng.uniform(0.3, 1.0) * a)
        cv2.circle(gut, c, r, 1.0, -1); cv2.circle(rim, c, r + 2, 1.0, 2)
    img *= 1 - rng.uniform(0.3, 0.75) * gf(gut, rng.uniform(1.5, 3.5))
    img += rng.uniform(0.03, 0.15) * gf(rim, 2.2)
    # fold bands: curved, dark or bright, soften the cells under them
    fold_soft = np.zeros((H, W), np.float32)
    if rng.random() < 0.8:
        for _ in range(int(rng.integers(*TUNE['folds']))):
            t = rng.uniform(0, np.pi); u = (xx - rng.uniform(0, W)) * np.cos(t) + (yy - rng.uniform(0, H)) * np.sin(t)
            v = -(xx - W / 2) * np.sin(t) + (yy - H / 2) * np.cos(t)
            u = u + rng.uniform(10, 90) * np.sin(v / rng.uniform(120, 500) + rng.uniform(0, 6.28))
            band = np.exp(-(u / rng.uniform(10, 70)) ** 2)
            img *= 1 + rng.choice([-1, 1], p=[0.7, 0.3]) * rng.uniform(0.06, 0.28) * band
            fold_soft = np.maximum(fold_soft, band)
    # defocus, milder than v2
    s1, s2 = rng.uniform(0.8, 1.6), rng.uniform(2.0, 5.0)
    b1, b2 = gf(img, s1), gf(img, s2)
    m = np.clip(smooth(rng, rng.uniform(150, 500)) * rng.uniform(0.4, 1.2) - rng.uniform(0.1, 0.5), 0, 1)
    m = np.maximum(m, fold_soft * rng.uniform(0.2, 0.7))
    img = (1 - m) * b1 + m * b2
    img = np.clip(img, 0.02, None) ** rng.uniform(0.8, 1.25)
    ill = illumination(rng, framed) ** (rng.uniform(0.45, 0.75) if framed else rng.uniform(0.6, 1.0))
    img = img * ill
    img = img / max(float(img.mean()), 1e-3) * rng.uniform(*TUNE['mean'])
    # bright fragments near the dark frame edge
    if framed:
        fy, fx = np.where((ill > 0.05) & (ill < 0.35))
        if len(fx):
            k = rng.choice(len(fx), min(len(fx), int(len(live) * rng.uniform(0.01, 0.03))), replace=False)
            img += impulses(np.c_[fx[k], fy[k]], rng.uniform(2.0, 5.0), rng.uniform(25, 80, len(k)).astype(np.float32))
    img = img * (1 + rng.normal(0, rng.uniform(0.004, 0.02), (H, W)).astype(np.float32))
    img += rng.normal(0, rng.uniform(*TUNE['noise']), (H, W)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def task(args):
    in_dir, out, name, seed = args
    dst = f'{out}/images/{name}.png'
    if os.path.exists(dst):
        return 0
    with np.load(f'{in_dir}/meta/{name}.npz') as z:
        lab = z['instance'].astype(np.int64); prm = json.loads(str(z['params'])); roi = z['roi'].copy(); target = z['target'].copy()
    rng = np.random.default_rng(seed)
    lab, info = reshape_cells(rng, lab)
    ctr, y = centres_and_labels(lab, roi)
    from PIL import Image
    np.savez_compressed(f'{out}/meta/{name}.npz', centres=ctr, instance=lab.astype(np.uint16), roi=roi, labels=y, target=target, params=json.dumps({**prm, **info}))
    Image.fromarray(appearance(rng, lab, prm['spacing_px'], prm['framed'])).save(dst)
    return 1


def main():
    global SRC
    in_dir, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 999000
    os.makedirs(f'{out}/images', exist_ok=True)
    os.makedirs(f'{out}/meta', exist_ok=True)
    sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
    import eval_oof_20261031 as E
    b = E.load_base(); st = E.image_stats(b)
    ann_ids = {re.sub(r'\s+', ' ', k.strip()) for k in json.load(open('/workspace/clear-ec-research/datasets/clear_ec_official/green_eval/annotations.json'))}
    ids = [re.sub(r'\s+', ' ', str(i).strip()) for i in b['ids']]
    ann = np.array([i not in ann_ids for i in ids])                     # never use an annotated frame as a source
    assert (~ann).sum() >= 20, (~ann).sum()
    SRC = dict(framed=b['idx'][ann & (st['black_frac'] > 0.01)], plain=b['idx'][ann & (st['black_frac'] <= 0.01)])
    names = sorted(x[:-4] for x in os.listdir(f'{in_dir}/meta') if x.endswith('.npz'))[:n]
    workers = int(min(8, max(2, 16 - os.getloadavg()[0] - 1)))
    tasks = [(in_dir, out, nm, seed0 + k) for k, nm in enumerate(names)]
    with get_context('fork').Pool(workers) as pool:
        done = sum(pool.imap_unordered(task, tasks, chunksize=4))
    print(json.dumps(dict(rendered=int(done), total=len(names), workers=workers)), flush=True)


if __name__ == '__main__':
    main()
