"""Deploy-mode smoke test (CPU): every full-data weight in manifests/deploy.json that exists is loaded and run on one frame; ViT-L (not built) and the
GPU-only hex6 feature are stubbed, as are the pieces without a builder. Not a parity test (full-data models saw this frame); it checks that the
deploy paths load, run and give values near the OOF ones. Test harness: reads stored values only to inject stubs and to print a comparison."""
import sys, json
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference')
import recipe_infer as RI, members as M
E = 'experiments/'; zc = np.load(E + 'lane_cd_20261108/reads/freeze_cd_oof_20261109.npz'); zh = np.load(E + 'lane_hex_20261108/preds/hex_inputs_oof.npz')
hexref = np.load(E + 'lane_cellroute_20261109/out/corr_hex_a100.npz')['P'][:, 2]
k = int(np.where(zc['fold'] == 3)[0][7]); d = int(zc['idx'][k]); MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
R = RI.Recipe('manifests/deploy.json', 'params/deploy.json'); M.clear_cache()
prov = {n: (zh[n][k] if n != 'have' else bool(zh[n][k])) for n in ('f6', 'have', 'best', 'graph', 'g_s10x3', 'g_hex46', 'prior_h6c', 'ridge_k3', 'ridge_fall16_h6_s15', 'dino_cvw3')}
prov.update(V=zc['V'][k], QUARTET_cv=np.nan, ORDER_QUARTET_cv=np.nan, frozen_lg=np.nan, CV_final=np.nan)
o = R.predict(np.asarray(MM[d]), provided=prov, skip=('V', 'f6')); p = o['parts']
print(json.dumps(dict(frame=d, seconds=o['seconds'], stubs=o['stubs'], CD_deploy=o['CD'], CD_oof=float(zc['oof_cd'][k]), HEX_deploy=o['HEX'], HEX_oof=float(hexref[k]),
                      A_cd=[p['A_cd'], float(zc['A'][k])], D=[float(p['D']), float(zc['D'][k])], P1_cd=[float(p['P1'][0]), float(zc['P1'][k])], DINO_cv=float(p['DINO_cv']))), flush=True)
