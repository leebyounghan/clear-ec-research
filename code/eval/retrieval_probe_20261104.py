"""Can an image alone find its adjacent-ID or fellow-eye image? (2026-11-04). Label-free descriptors of the raw frame, on CPU:
  thumb   per-image standardised 48 x 36 thumbnail (illumination and frame shape)
  hist    square-root gray histogram, 64 bins (exposure, LUT)
  fixed   high-pass (frame - sigma 16 blur) averaged over 8 px blocks (dust and fixed-pattern noise, if any survive)
For known pairs (new-format fellow eyes; old-format adjacent IDs, about half of which are fellow eyes judging by CD), the rank of the
partner among all 8953 other images. Random is a median rank of about 4476. Forked pool, clamp(16 - load1 - 1, 2, 12)."""
import json, os, re, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import cv2
import numpy as np
cv2.setNumThreads(0)
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
RAW = '/workspace/clear-ec-research/datasets/clear_ec_fullframe_20260904/img_full_raw.dat'
OUT = '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/retrieval'
def desc(i):
    a = np.asarray(np.memmap(RAW, dtype=np.uint8, mode='r', shape=(8954, 972, 1296))[i], dtype=np.float32)
    t = cv2.resize(a, (48, 36), interpolation=cv2.INTER_AREA); t = (t - t.mean()) / (t.std() + 1e-6)
    h = np.sqrt(np.histogram(a, 64, (0, 256))[0] / a.size).astype(np.float32)
    hp = a - cv2.GaussianBlur(a, (0, 0), 16); f = cv2.resize(hp, (162, 121), interpolation=cv2.INTER_AREA); f = (f - f.mean()) / (f.std() + 1e-6)
    return t.ravel().astype(np.float32), h, f.ravel().astype(np.float32)
def main():
    os.makedirs(OUT, exist_ok=True)
    b = E.load_base(); ids = [re.sub(r'\s+', ' ', i.strip()) for i in b['ids']]; y = b['y']
    workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1)))
    with get_context('fork').Pool(workers) as pool:
        res = list(pool.imap(desc, b['idx'].tolist(), chunksize=32))
    T, H, F = (np.stack([r[k] for r in res]) for k in range(3))
    np.savez(f'{OUT}/raw_desc.npz', idx=b['idx'], T=T, H=H, F=F)
    old, new = {}, {}
    for n, i in enumerate(ids):
        m = re.fullmatch(r'(\d{4})-(\d{2})', i)
        if m: old[(int(m.group(2)), int(m.group(1)))] = n; continue
        m = re.match(r'(\d{4})-(\d{4})[ -]*([O0][DS])', i)
        if m: new[(int(m.group(1)), int(m.group(2)), m.group(3).replace('0', 'O'))] = n
    pairs = dict(new_fellow=[(new[k], new[(k[0], k[1], 'OS')]) for k in new if k[2] == 'OD' and (k[0], k[1], 'OS') in new],
                 old_adjacent=[(old[k], old[(k[0], k[1] + 1)]) for k in old if (k[0], k[1] + 1) in old])
    # the strongest fellow candidates among old adjacent pairs, using labels only to read the result
    P = np.array(pairs['old_adjacent']); close = np.abs(np.log(np.maximum(y[P[:, 0], 0], 1)) - np.log(np.maximum(y[P[:, 1], 0], 1))) < 0.03
    pairs['old_adjacent_cd_close'] = P[close].tolist()
    rng = np.random.default_rng(0); rep = dict(workers=workers)
    def unit(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    D = {'thumb': unit(T), 'hist': unit(H), 'fixed': unit(F)}
    D['thumb+hist'] = unit(np.c_[unit(T), unit(H)])
    for nm, Q in D.items():
        rep[nm] = {}
        for pn, pr in pairs.items():
            pr = np.array(pr); pr = pr[rng.permutation(len(pr))[:400]]
            s = Q[pr[:, 0]] @ Q.T
            s[np.arange(len(pr)), pr[:, 0]] = -9
            rank = (s > s[np.arange(len(pr)), pr[:, 1]][:, None]).sum(1) + 1
            rep[nm][pn] = dict(n=len(pr), median_rank=int(np.median(rank)), top1=round(float((rank == 1).mean()), 3), top10=round(float((rank <= 10).mean()), 3), top100=round(float((rank <= 100).mean()), 3))
        print(nm, json.dumps(rep[nm]), flush=True)
    json.dump(rep, open(f'{OUT}/raw_rank.json', 'w'), indent=1)
if __name__ == '__main__':
    main()
