"""Deploy-mode GPU run (cell-route lane, 2026-11-10): full-data weights on N training frames spread over the folds, hex6 feature live. Inputs whose
full-data build is missing are stubbed from stored OOF values (listed). Not a parity test (full-data models saw these frames): reports the deploy vs OOF
gap per column (median and max relative difference), the hex6 feature against the stored OOF f6, and the per-image GPU time. usage: test_gpu_deploy.py N OUT"""
import sys, json, os, functools
import numpy as np, torch
torch.set_num_threads(4)
_load = torch.load
@functools.lru_cache(maxsize=None)
def _cached(path, wo): return _load(path, map_location='cpu', weights_only=wo)
def cached_load(f, *a, **k):
    if isinstance(f, str) and (f.endswith('.pt') or 'cpsam_v2' in f): return _cached(f, bool(k.get('weights_only', False)))
    return _load(f, *a, **k)
torch.load = cached_load
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference')
import recipe_infer as RI, members as M
E = 'experiments/'; zc = np.load(E + 'lane_cd_20261108/reads/freeze_cd_oof_20261111.npz'); zh = np.load(E + 'lane_hex_20261108/preds/hex_inputs_oof.npz'); zv = np.load(E + 'lane_cv_20261108/deploy/cv_inputs_oof.npz', allow_pickle=True)
crref = np.load(E + 'lane_cellroute_20261109/out/corr_hex_on_joint_attnp.npz')['P']
MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
N = int(sys.argv[1]); OUT = sys.argv[2]; rows = np.linspace(0, len(zc['idx']) - 1, N).astype(int)
R = RI.Recipe('manifests/deploy.json', 'params/deploy.json'); res = []
vs = R.m['vitl_seeds']; v_ok = all(os.path.exists(p) for s in vs for p in s['single'] + [s['tta']]); c_ok = all(os.path.exists(p) for p in R.m['cps5'])
for k in rows:
    d = int(zc['idx'][k]); M.clear_cache(); prov, skip = {}, []
    if not v_ok: prov['V'] = zc['V'][k]; skip.append('V')
    if not c_ok: prov['cps5'] = zh['cps5'][k]
    o = R.predict(np.asarray(MM[d]), provided=prov, skip=tuple(skip)); p = o['parts']
    rel = lambda a, b: float(abs(a / b - 1))
    row = dict(frame=d, fold=int(zc['fold'][k]), seconds=o['seconds'], stubs=o['stubs'], CD=rel(o['CD'], zc['oof_cd'][k]), CV=rel(o['CV'], zv['cv_oof'][k]), HEX=rel(o['HEX'], crref[k]),
               f6_deploy=p['f6'], f6_oof=float(zh['f6'][k]), have=[bool(p['have']), bool(zh['have'][k])],
               deploy=[float(o['CD']), float(o['CV']), float(o['HEX'])], oof=[float(zc['oof_cd'][k]), float(zv['cv_oof'][k]), float(crref[k])], cps5_deploy=float(p['cps5']), cps5_oof=float(zh['cps5'][k]))
    res.append(row); print(json.dumps(row), flush=True)
A = {q: np.array([r[q] for r in res]) for q in ('CD', 'CV', 'HEX')}; f = np.array([[r['f6_deploy'], r['f6_oof']] for r in res if r['have'][0] and r['have'][1]])
summ = dict(frames=len(res), stubs=res[0]['stubs'], seconds_median=float(np.median([r['seconds'] for r in res[1:]])), median_rel={q: float(np.median(v)) for q, v in A.items()},
            max_rel={q: float(np.max(v)) for q, v in A.items()}, f6_corr=float(np.corrcoef(f[:, 0], f[:, 1])[0, 1]) if len(f) > 2 else None, f6_median_abs_diff=float(np.median(np.abs(f[:, 0] - f[:, 1]))) if len(f) else None)
print('SUMMARY', json.dumps(summ), flush=True); json.dump(dict(summary=summ, rows=res), open(E + f'lane_cellroute_20261109/out/{OUT}.json', 'w'), indent=1)
