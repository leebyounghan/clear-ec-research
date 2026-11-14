"""Fold-0 CV parity after the CV lane's 6-component freeze (cell-route lane, 2026-11-11). Test harness, one GPU, <= 4 CPU threads.
The CV inputs the entry computes from the image run live with fold-0 weights, through the same calls as recipe_infer.Recipe.predict:
recipe A TTA (A), the Phase-1 five members (P1), plain DINOv3 s46/s47/s49 epochs 16-20 (DINO_logcv). The CV per-image graph / ridge / quartet /
frozen-head parts have full-data weights only, so they come from lane_cv deploy/cv_inputs_oof.npz (as in test_gpu_fold0.py). The CV column is then
Recipe.cv_column with params/fold0.json (rebuilt from the new npz: w0, sA[0], stretch0). Same frames as test_gpu_fold0.py. usage: test_cv_fold0.py N OUT"""
import sys, json, time, functools
import numpy as np, torch
torch.set_num_threads(4)
_load = torch.load
@functools.lru_cache(maxsize=None)
def _cached(path, wo): return _load(path, map_location='cpu', weights_only=wo)
def cached_load(f, *a, **k):
    if isinstance(f, (str,)) and (f.endswith('.pt') or 'cpsam_v2' in f): return _cached(f, bool(k.get('weights_only', False)))
    return _load(f, *a, **k)
torch.load = cached_load
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference')
import recipe_infer as RI, members as M
E = 'experiments/'
zc = np.load(E + 'lane_cd_20261108/reads/freeze_cd_oof_20261109.npz'); zv = np.load(E + 'lane_cv_20261108/deploy/cv_inputs_oof.npz', allow_pickle=True)
assert np.array_equal(np.asarray(zc['idx']).astype(int), np.asarray(zv['idx']).astype(int))
idx = zc['idx'].astype(int); fold = zc['fold']; r0 = np.where(fold == 0)[0]
CVK = ['graph_cv', 'ridge_cv', 'qplain_cv', 'qorder_cv', 'fz_std', 'fz_spectrum', 'fz_cov_sqrt', 'fz_autocorr']
MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
N = int(sys.argv[1]); OUT = sys.argv[2]; rows = r0[np.linspace(0, len(r0) - 1, N).astype(int)]
R = RI.Recipe('manifests/fold0.json', 'params/fold0.json'); m = R.m; sA = float(zv['sA'][0]); s1 = float(zv['phase1_scalar_cv'])
assert R.cv_weight('orderdino') == 0 and R.cv_weight('dino') > 0
rel = lambda a, b: float(abs(a / b - 1)); res = []
for k in rows:
    d = int(idx[k]); M.clear_cache(); t = time.time(); img = np.asarray(MM[d], np.uint8); x = M.preprocess(img)
    A = M.recipeA_tta(x, m['recipeA_tta'], R.p['A_scalars']); P1 = R.p1(img); dino = float(np.log(M.dino_cv(x, m['dino_cv'])[1]))
    parts = dict(A=A, P1=P1, DINO_logcv=dino, orderdino_logcv=np.nan, **{q: float(zv[q][k]) for q in CVK})
    cv = R.cv_column(parts)
    row = dict(frame=d, seconds=round(time.time() - t, 1), A_cv=rel(A[1], zv['a_raw_cv'][k] * sA), P1_cv=rel(P1[1], zv['p1_cv'][k] * s1),
               DINO_logcv=rel(dino, zv['dino_logcv'][k]), CV_final=rel(cv, zv['cv_oof'][k]), cv=cv, cv_oof=float(zv['cv_oof'][k]))
    res.append(row); print(json.dumps(row), flush=True)
keys = ['A_cv', 'P1_cv', 'DINO_logcv', 'CV_final']; Aa = {q: np.array([r[q] for r in res]) for q in keys}
summ = dict(frames=len(res), seconds_median=float(np.median([r['seconds'] for r in res])), max={q: float(v.max()) for q, v in Aa.items()},
            median={q: float(np.median(v)) for q, v in Aa.items()}, p95={q: float(np.percentile(v, 95)) for q, v in Aa.items()})
print('SUMMARY', json.dumps(summ), flush=True); json.dump(dict(summary=summ, rows=res), open(E + f'lane_cellroute_20261109/out/{OUT}.json', 'w'), indent=1)
