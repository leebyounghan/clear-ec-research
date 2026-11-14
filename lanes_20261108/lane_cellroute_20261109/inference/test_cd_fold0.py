"""Fold-0 CD parity after the CD lane's seed audit (cell-route lane, 2026-11-11). Test harness, one GPU, <= 4 CPU threads.
D runs live through members.dino_cd with the fold-0 manifest's two seeds (s46 lane_gpu3, s47 lane_gpu2; 4 flips x epochs 16-20 each, the runs the CD
lane's tta files were made from), compared with the stored two-seed D of reads/freeze_cd_oof_20261111.npz. The CD column is Recipe.cd_column with the
live D and the stored P1, A, V, mask (each reproduced live earlier: test_gpu_fold0.py, 120 frames) and params/fold0.json (freeze_cd_params_20261111
per_fold 0), against stored oof_cd. Same frames as test_gpu_fold0.py. usage: test_cd_fold0.py N OUT"""
import sys, json, time, functools
import numpy as np, torch
torch.set_num_threads(4)
_load = torch.load
@functools.lru_cache(maxsize=None)
def _cached(path, wo): return _load(path, map_location='cpu', weights_only=wo)
def cached_load(f, *a, **k):
    if isinstance(f, (str,)) and f.endswith('.pt'): return _cached(f, bool(k.get('weights_only', False)))
    return _load(f, *a, **k)
torch.load = cached_load
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference')
import recipe_infer as RI, members as M
E = 'experiments/'; Z = np.load(E + 'lane_cd_20261108/reads/freeze_cd_oof_20261111.npz'); S = np.load(E + 'lane_cd_20261108/reads/seedmean_d_oof_20261111.npz')
idx = Z['idx'].astype(int); r0 = np.where(Z['fold'] == 0)[0]
MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
N = int(sys.argv[1]); OUT = sys.argv[2]; rows = r0[np.linspace(0, len(r0) - 1, N).astype(int)]
R = RI.Recipe('manifests/fold0.json', 'params/fold0.json'); assert len(R.m['dino_cd']) == 2
rel = lambda a, b: float(abs(a / b - 1)); res = []
for k in rows:
    d = int(idx[k]); M.clear_cache(); t = time.time(); x = M.preprocess(np.asarray(MM[d], np.uint8))
    D = float(M.dino_cd(x, R.m['dino_cd'])[0]); D46 = float(M.dino_cd(x, R.m['dino_cd'][0])[0]); D47 = float(M.dino_cd(x, R.m['dino_cd'][1])[0])
    s = np.full(17, np.nan); s[5] = Z['mask_cd'][k]; s[3] = Z['mask_n_int'][k]
    cd = R.cd_column(dict(P1=[float(Z['P1'][k]), np.nan, np.nan], A_cd=float(Z['A'][k]), D=D, V=float(Z['V'][k]), seg17=s))
    row = dict(frame=d, seconds=round(time.time() - t, 1), D=rel(D, Z['D'][k]), D46=rel(D46, S['D46'][k]), D47=rel(D47, S['D47'][k]), CD_final=rel(cd, Z['oof_cd'][k]), cd=cd, oof_cd=float(Z['oof_cd'][k]))
    res.append(row); print(json.dumps(row), flush=True)
keys = ['D', 'D46', 'D47', 'CD_final']; A = {q: np.array([r[q] for r in res]) for q in keys}
summ = dict(frames=len(res), seconds_median=float(np.median([r['seconds'] for r in res])), max={q: float(v.max()) for q, v in A.items()},
            median={q: float(np.median(v)) for q, v in A.items()}, p95={q: float(np.percentile(v, 95)) for q, v in A.items()})
print('SUMMARY', json.dumps(summ), flush=True); json.dump(dict(summary=summ, rows=res), open(E + f'lane_cellroute_20261109/out/{OUT}.json', 'w'), indent=1)
