"""Deploy-mode smoke test v2 (CPU, one training frame): full-data weights for everything that exists, including the CV per-image parts; compares
the GRAPH / ridge / quartet outputs with the CV lane's deploy in-sample members (same full-data networks on the stored training peaks).
Stubs: ViT-L (not built), hex6 f6 (GPU-only code), the 13 non-pre-only detector peaks (2 not built; stored peaks used), HEX graph nets / cvw3 / cps5."""
import sys, json
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference')
import recipe_infer as RI, members as M
E = 'experiments/'; zc = np.load(E + 'lane_cd_20261108/reads/freeze_cd_oof_20261109.npz'); zh = np.load(E + 'lane_hex_20261108/preds/hex_inputs_oof.npz')
ins = np.load(E + 'lane_cv_20261108/deploy/cv_deploy_insample_members.npz', allow_pickle=True); zv = np.load(E + 'lane_cv_20261108/deploy/cv_inputs_oof.npz', allow_pickle=True)
k = int(np.where(zc['fold'] == 3)[0][7]); d = int(zc['idx'][k]); g = int(zc['fold'][k]); MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
def sp(det):
    z = np.load(E + f'green_track_20261028/{det}/fold{g}/fold{g}_sg2_peaks.npz', allow_pickle=True); j = list(z['indices']).index(d); return z['peaks'][z['offsets'][j]:z['offsets'][j + 1]].astype(np.float32)
R = RI.Recipe('manifests/deploy.json', 'params/deploy.json'); M.clear_cache()
prov = {n: zh[n][k] for n in ('g_s10x3', 'g_hex46', 'dino_cvw3', 'cps5')}; prov.update(f6=zh['f6'][k], have=bool(zh['have'][k]), V=zc['V'][k], peaks_other13={x: sp(x) for x in RI.ALL16[3:]})
o = R.predict(np.asarray(MM[d]), provided=prov, skip=('V', 'f6', 'peaks_other13')); p = o['parts']; cp = p['cv_parts']
print(json.dumps(dict(frame=d, fold=g, seconds=o['seconds'], stubs=o['stubs'], CD=o['CD'], CV=o['CV'], HEX=o['HEX'], CV_oof=float(zv['cv_oof'][k]),
                      G_mine=np.round(cp['G'], 5).tolist(), G_insample=np.round(ins['G'][k], 5).tolist(), QP_mine=np.round(cp['QP'], 5).tolist(), QP_insample=np.round(ins['QP'][k], 5).tolist(),
                      QO_mine=np.round(cp['QO'], 5).tolist(), QO_insample=np.round(ins['QO'][k], 5).tolist(), ridge_mine=cp['ridge'], RID_insample=np.round(ins['RID'][k], 5).tolist(), fz=cp['fz'])), flush=True)
