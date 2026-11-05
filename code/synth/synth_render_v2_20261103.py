"""Renderer v2 appearance (2026-11-03): v1 was far easier than real frames for a real-trained detector (image 00000:
miss 0.11, FP 0.01, jitter 0.07 x spacing vs real miss 0.17, FP 0.17, jitter 0.22). Re-renders the v1 geometry (cell map)
with what makes real frames hard: weak and uneven borders, cell texture comparable to the borders, specular blobs and
guttae that look like cells, Descemet-fold bands, broad defocus, and illumination taken from a real frame low-passed
far below the cell scale (real vignette and folds, no real cell positions). Annotated frames never serve as the source.
usage: synth_render_v2_20261103.py IN_DIR OUT_DIR N [SEED0]. Forked pool sized clamp(16 - load1 - 1, 2, 8).
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


def appearance(rng, lab, a, framed):
    n = int(lab.max()) + 1
    border = np.zeros((H, W), np.uint8)
    border[:, 1:] |= (lab[:, 1:] != lab[:, :-1]); border[1:, :] |= (lab[1:, :] != lab[:-1, :])
    dist = cv2.distanceTransform(1 - border, cv2.DIST_L2, 3)
    # border depth varies by region and by cell; some borders almost vanish
    lo, hi = rng.uniform(0.02, 0.09), rng.uniform(0.1, 0.24)
    depth = (lo + (hi - lo) * smooth(rng, rng.uniform(100, 300))) * rng.lognormal(0, rng.uniform(0.08, 0.3), n).astype(np.float32)[lab]
    width = rng.uniform(0.8, 2.0) + rng.uniform(0.02, 0.09) * a
    img = 1 - np.clip(depth, 0, 0.95) * np.exp(-(dist / width) ** 2)
    img *= rng.normal(1.0, rng.uniform(0.008, 0.05), n).astype(np.float32)[lab]
    # texture inside cells at sub-cell scale, comparable to the borders
    t = rng.normal(0, 1, (H, W)).astype(np.float32)
    tex = gf(t, rng.uniform(1.2, 3.5)); tex /= tex.std() + 1e-6
    tex2 = gf(rng.normal(0, 1, (H, W)).astype(np.float32), rng.uniform(4, 9)); tex2 /= tex2.std() + 1e-6
    img += rng.uniform(*TUNE['tex']) * tex + rng.uniform(0.008, 0.035) * tex2
    # nuclei or dimples at some cell centres
    if rng.random() < 0.5:
        img -= rng.uniform(0.02, 0.1) * np.exp(-((dist - 0.45 * a) / (0.2 * a)) ** 2) * (rng.random(n).astype(np.float32)[lab] < 0.5)
    # specular blobs (false-positive bait) and guttae
    yy, xx = None, None
    blobs = np.zeros((H, W), np.float32)
    for _ in range(int(rng.integers(*TUNE['blob_n']))):
        blobs[int(rng.integers(H)), int(rng.integers(W))] = rng.uniform(0.6, 1.6)
    img += gf(blobs, rng.uniform(2.0, 5.0)) * rng.uniform(15, 60)
    gut = np.zeros((H, W), np.float32)
    for _ in range(int(rng.integers(*TUNE['gut_n'])) if TUNE['gut_n'][1] else 0):
        cv2.circle(gut, (int(rng.integers(W)), int(rng.integers(H))), int(rng.uniform(0.3, 1.2) * a), 1.0, -1)
    img *= 1 - rng.uniform(0.3, 0.8) * gf(gut, rng.uniform(2, 5))
    # fold bands: curved, dark or bright, soften the cells under them
    fold_soft = np.zeros((H, W), np.float32)
    if rng.random() < 0.8:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        for _ in range(int(rng.integers(*TUNE['folds']))):
            th = rng.uniform(0, np.pi); u = (xx - rng.uniform(0, W)) * np.cos(th) + (yy - rng.uniform(0, H)) * np.sin(th)
            v = -(xx - W / 2) * np.sin(th) + (yy - H / 2) * np.cos(th)
            u = u + rng.uniform(10, 90) * np.sin(v / rng.uniform(120, 500) + rng.uniform(0, 6.28))
            band = np.exp(-(u / rng.uniform(10, 70)) ** 2)
            img *= 1 + rng.choice([-1, 1], p=[0.7, 0.3]) * rng.uniform(0.06, 0.28) * band
            fold_soft = np.maximum(fold_soft, band)
    # broad defocus in three levels
    s1, s2 = rng.uniform(1.0, 2.2), rng.uniform(TUNE['defocus'][0] + 1, TUNE['defocus'][1])
    b1, b2 = gf(img, s1), gf(img, s2)
    m = np.clip(smooth(rng, rng.uniform(150, 500)) * rng.uniform(0.6, 1.5) - rng.uniform(0, 0.4), 0, 1)
    m = np.maximum(m, fold_soft * rng.uniform(0.3, 0.9))
    img = (1 - m) * b1 + m * b2
    img = np.clip(img, 0.02, None) ** rng.uniform(0.8, 1.25)
    img = img * illumination(rng, framed) ** rng.uniform(0.6, 1.0)
    img = img / max(float(img.mean()), 1e-3) * rng.uniform(*TUNE['mean'])
    img = img * (1 + rng.normal(0, rng.uniform(0.004, 0.02), (H, W)).astype(np.float32))
    img += rng.normal(0, rng.uniform(*TUNE['noise']), (H, W)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def task(args):
    in_dir, out, name, seed = args
    dst = f'{out}/images/{name}.png'
    if os.path.exists(dst):
        return 0
    with np.load(f'{in_dir}/meta/{name}.npz') as z:
        lab = z['instance'].astype(np.int64)
        prm = json.loads(str(z['params']))
    rng = np.random.default_rng(seed)
    from PIL import Image
    Image.fromarray(appearance(rng, lab, prm['spacing_px'], prm['framed'])).save(dst)
    return 1


def main():
    global SRC
    in_dir, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 777000
    os.makedirs(f'{out}/images', exist_ok=True)
    if not os.path.lexists(f'{out}/meta'):
        os.symlink(os.path.abspath(f'{in_dir}/meta'), f'{out}/meta')
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
