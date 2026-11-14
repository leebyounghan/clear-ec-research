"""GPU fold-0 reproduction of recipe_infer.py on a larger sample (cell-route lane, 2026-11-10). Test harness: reads stored OOF values to compare and to
inject the inputs that have no fold-0 weights. Run on one GPU (CUDA_VISIBLE_DEVICES), <= 4 CPU threads. Unlike the CPU layer-3 test, the 13 non-pre-only
detectors run for real with their fold-0 checkpoints. Stubs: CV per-image parts, g_s10x3, g_hex46 (full-data weights only), dino_cvw3 (fold runs kept
epochs 16-20 only), f6 / have (hex6 deploy corrector is full-data), the ViT-L epoch 8-11 halves (fold runs kept final.pt only).
Checkpoints are read once into RAM (test-only torch.load cache) so 100+ frames do not re-read them. usage: test_gpu_fold0.py N OUTNAME"""
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
E = 'experiments/'; HERE = E + 'lane_cellroute_20261109/inference/'
zc = np.load(E + 'lane_cd_20261108/reads/freeze_cd_oof_20261111.npz'); zh = np.load(E + 'lane_hex_20261108/preds/hex_inputs_oof.npz'); zv = np.load(E + 'lane_cv_20261108/deploy/cv_inputs_oof.npz', allow_pickle=True)
hexref = np.load(E + 'lane_hex_20261108/preds/chain_joint_attnp.npz')['final_hex']; crref = np.load(E + 'lane_cellroute_20261109/out/corr_hex_on_joint_attnp.npz')['P']
zf = np.load(E + 'lane_cellroute_20261109/feats/cellroute_feats.npz'); NM = list(zf['names']); CRF = zf['F'][:, [NM.index(c) for c in RI.DK.NAMES]]
idx = zc['idx'].astype(int); fold = zc['fold']; r0 = np.where(fold == 0)[0]; CVK = ['graph_cv', 'ridge_cv', 'qplain_cv', 'qorder_cv', 'fz_std', 'fz_spectrum', 'fz_cov_sqrt', 'fz_autocorr']
rel = lambda a, b: float(np.nanmax(np.abs(np.asarray(a, float) / np.asarray(b, float) - 1)))
MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
def vsl(run):
    L = np.mean([np.log(np.load(run + f'predictions_epoch_{e:02d}.npz')['pred']) for e in range(8, 12)], 0); return {int(i): v for i, v in zip(np.load(run + 'predictions_epoch_08.npz')['indices'], L)}
VS = [vsl(E + 'lane_backbone_20261108/runs/vitl16_avg_f0_s46/raw/runs/fold0/dinov3_l16t/'), vsl(E + 'lane_cd_20261108/arms/vitl16_f0_s47/raw/runs/fold0/dinov3_l16t/')]
N = int(sys.argv[1]); OUT = sys.argv[2]; rows = r0[np.linspace(0, len(r0) - 1, N).astype(int)]
R = RI.Recipe('manifests/fold0.json', 'params/fold0.json'); res = []
for n, k in enumerate(rows):
    d = int(idx[k]); M.clear_cache()
    prov = {q: (zh[q][k] if q != 'have' else bool(zh[q][k])) for q in ('f6', 'have', 'best', 'graph', 'g_s10x3', 'g_hex46', 'dino_cvw3')}
    prov.update({q: zv[q][k] for q in CVK}); prov.update(V_single_logs=[VS[0][d], VS[1][d]])
    o = R.predict(np.asarray(MM[d]), provided=prov); p = o['parts']
    row = dict(frame=d, seconds=o['seconds'], A=rel(p['A'][2], zh['A'][k]), A_cd=rel(p['A_cd'], zc['A'][k]), D=rel(p['D'], zc['D'][k]), V=rel(p['V'], zc['V'][k]), P1_cd=rel(p['P1'][0], zc['P1'][k]),
               P1_hex=rel(p['P1'][2], zh['P1'][k]), DINO_logcv=rel(p['DINO_logcv'], zv['dino_logcv'][k]) if 'dino_logcv' in zv.files else 0.0,
               orderdino_logcv=rel(p['orderdino_logcv'], zv['orderdino_logcv'][k]) if 'orderdino_logcv' in zv.files else 0.0, cps5=rel(p['cps5'], zh['cps5'][k]),
               mask_cd=rel(p['seg17'][5], zc['mask_cd'][k]) if np.isfinite(zc['mask_cd'][k]) else 0.0, prior_h6c=rel(p['prior_h6c'], zh['prior_h6c'][k]), ridge_k3=rel(p['ridge_k3'], zh['ridge_k3'][k]),
               ridge_fall16=rel(p['ridge_fall16_h6_s15'], zh['ridge_fall16_h6_s15'][k]), cellroute_feats_full=rel(np.asarray(p['cellroute_feats'])[5:], CRF[d][5:]),
               CD_final=rel(o['CD'], zc['oof_cd'][k]), CV_final=rel(o['CV'], zv['cv_oof'][k]), HEX_chain=rel(p['hex_chain'], hexref[k]), HEX_final=rel(o['HEX'], crref[k]), stubs=o['stubs'])
    res.append(row); print(json.dumps({q: (round(v, 6) if isinstance(v, float) else v) for q, v in row.items() if q != 'stubs'}), flush=True)
keys = [q for q in res[0] if q not in ('frame', 'seconds', 'stubs')]; A = {q: np.array([r[q] for r in res], float) for q in keys}
summ = dict(frames=len(res), stubs=res[0]['stubs'], seconds_median=float(np.median([r['seconds'] for r in res])),
            max={q: float(np.nanmax(v)) for q, v in A.items()}, median={q: float(np.nanmedian(v)) for q, v in A.items()}, p95={q: float(np.nanpercentile(v, 95)) for q, v in A.items()})
print('SUMMARY', json.dumps(summ), flush=True); json.dump(dict(summary=summ, rows=res), open(E + f'lane_cellroute_20261109/out/{OUT}.json', 'w'), indent=1)
