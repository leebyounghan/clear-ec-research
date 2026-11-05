"""Are there near-duplicate frames in the training set (2026-11-04)? If a frame was acquired or exported twice, a held-out test frame may have a twin in train. Top-1 similarity for each frame on raw thumbnails, on high-pass fixed-pattern maps and on the
exact pixels of the best thumbnail candidates; label agreement of the closest pairs against random pairs. Report only: nothing here is used by a model."""
import json, os, re, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
os.environ.setdefault('OMP_NUM_THREADS', '4')
OUT = '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/retrieval'
RAW = '/workspace/clear-ec-research/datasets/clear_ec_fullframe_20260904/img_full_raw.dat'
b = E.load_base(); y, fold, ids = b['y'], b['fold'], [re.sub(r'\s+', ' ', i.strip()) for i in b['ids']]; N = len(ids); ly = np.log(np.where(y > 0, y, np.nan))
d = np.load(f'{OUT}/raw_desc.npz'); pos = {int(i): n for n, i in enumerate(d['idx'])}; o = np.array([pos[int(i)] for i in b['idx']])
def unit(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
T = unit(d['T'][o].astype(np.float32)); F = unit(d['F'][o].astype(np.float32))
def top(Q):
    best = np.zeros(N); arg = np.zeros(N, int)
    for s in range(0, N, 512):
        S = Q[s:s + 512] @ Q.T; S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -9; arg[s:s + 512] = S.argmax(1); best[s:s + 512] = S.max(1)
    return best, arg
mm = np.memmap(RAW, dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
for nm, Q in dict(thumb=T, fixed=F).items():
    best, arg = top(Q); print(nm, 'top-1 cosine quantiles', np.round(np.quantile(best, [.5, .9, .99, .999, 1]), 4).tolist(), flush=True)
    order = np.argsort(-best)[:60]; seen = set(); rows = []
    for i in order:
        j = arg[i]; k = tuple(sorted((int(i), int(j))))
        if k in seen: continue
        seen.add(k); a, c = np.asarray(mm[b['idx'][i]], np.float32), np.asarray(mm[b['idx'][j]], np.float32); px = float(np.corrcoef(a.ravel()[::7], c.ravel()[::7])[0, 1]); same = float((a == c).mean())
        rows.append(dict(a=ids[i], c=ids[j], cos=round(float(best[i]), 4), pixel_r=round(px, 4), identical=round(same, 4), y_a=[round(float(v), 3) for v in y[i]], y_c=[round(float(v), 3) for v in y[j]], folds=[int(fold[i]), int(fold[j])]))
    print(json.dumps(rows[:12]), flush=True)
    s = best > np.quantile(best, .99); print(' top 1 % pairs |log CD diff| median', round(float(np.nanmedian(np.abs(ly[s, 0] - ly[arg[s], 0]))), 3), 'random', round(float(np.nanmedian(np.abs(ly[:, 0] - np.random.default_rng(0).permutation(ly[:, 0])))), 3), 'pixel r max', max(r['pixel_r'] for r in rows), 'identical max', max(r['identical'] for r in rows))
print('done', flush=True)
