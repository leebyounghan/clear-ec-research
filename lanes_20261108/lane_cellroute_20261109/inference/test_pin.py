"""(1) CV lane's pinning test: frame 32 (fold 3) with its STORED fold-detector peaks fed into Recipe.cv_parts (detectors skipped): GRAPH / quartets /
ridge must equal cv_deploy_insample_members (same full-data nets, same peaks). (2) cps5 member code on 2 fold-0 frames with the fold-0 CPSAM 512 run
(epochs 8-12) vs hex_inputs_oof cps5."""
import sys, json, numpy as np, torch
torch.set_num_threads(16)
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference')
import recipe_infer as RI, members as M
E = 'experiments/'; MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
zc = np.load(E + 'lane_cd_20261108/reads/freeze_cd_oof_20261109.npz'); ins = np.load(E + 'lane_cv_20261108/deploy/cv_deploy_insample_members.npz'); zh = np.load(E + 'lane_hex_20261108/preds/hex_inputs_oof.npz')
k = int(np.where(zc['fold'] == 3)[0][7]); d = int(zc['idx'][k])
def st(path):
    z = np.load(path, allow_pickle=True); j = list(z['indices']).index(d); return z['peaks'][z['offsets'][j]:z['offsets'][j + 1]].astype(np.float32)
single3 = [st(E + f'green_track_20261028/external_srep/{s}/fold3/fold3_sg2_peaks.npz') for s in ('preonly', 's47_preonly', 's48_preonly')]
tta3 = [st(E + f'lane_cv_20261108/out/tta_peaks/{s}/fold3/fold3_sg2_peaks.npz') for s in ('s46_tta4', 's47_tta4', 's48_tta4')]; mean3 = st(E + 'lane_cv_20261108/out/tta_peaks/seeds3_tta4/fold3/fold3_sg2_peaks.npz')
R = RI.Recipe('manifests/deploy.json', 'params/deploy.json'); x = M.preprocess(np.asarray(MM[d])); cp = R.cv_parts(x, (mean3, tta3, single3))
dif = lambda a, b: float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))
print(json.dumps(dict(frame=d, G=dif(cp['G'], ins['G'][k]), QP=dif(cp['QP'], ins['QP'][k]), QO=dif(cp['QO'], ins['QO'][k]), ridge_CV=dif(cp['ridge']['CV'], ins['RID'][k][1]), ridge_HEX=dif(cp['ridge']['HEX'], ins['RID'][k][2]))), flush=True)
P0 = json.load(open(RI.HERE + '/manifests/fold0.json'))['cps5']; r0 = np.where(zh['fold'] == 0)[0]
for kk in r0[[0, 900]]:
    dd = int(zh['idx'][kk]); M.clear_cache(); v = M.family_mean(M.preprocess(np.asarray(MM[dd])), P0, 'cpsam_c512')[2]
    print(json.dumps(dict(frame=dd, cps5_mine=float(v), cps5_oof=float(zh['cps5'][kk]), rel=float(abs(v / zh['cps5'][kk] - 1)))), flush=True)
