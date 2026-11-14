"""Quick check (no model forward): Recipe.hex_features on the stored fold-0 single-view peaks of the 16 detectors and the stored hybrid-B CD
-> prior_h6c, ridge_k3, ridge_fall16 vs the HEX lane's stored inputs, 4 fold-0 frames."""
import sys, json, numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference')
import recipe_infer as RI, members as M
E = 'experiments/'; zh = np.load(E + 'lane_hex_20261108/preds/hex_inputs_oof.npz'); B = np.load(E + 'recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True)
Bcd = {int(i): v for i, v in zip(B['indices'], B['B_hybrid_w0p5'][:, 0])}
def sp(det):
    z = np.load(E + f'green_track_20261028/{det}/fold0/fold0_sg2_peaks.npz', allow_pickle=True); pk, of = z['peaks'], z['offsets']; return {int(i): pk[of[k]:of[k + 1]].astype(np.float32) for k, i in enumerate(z['indices'])}
SP = {d: sp(d) for d in RI.ALL16}
MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
R = RI.Recipe.__new__(RI.Recipe); r0 = np.where(zh['fold'] == 0)[0]
for k in r0[::600][:4]:
    d = int(zh['idx'][k]); F, G = R.hex_features(np.asarray(MM[d]), [SP[x][d] for x in RI.ALL16[:3]], {x: SP[x][d] for x in RI.ALL16[3:]}, Bcd[d])
    print(json.dumps(dict(frame=d, prior_h6c=[float(F['prior_h6c'][0]), float(zh['prior_h6c'][k])], ridge_k3_maxdiff=float(np.nanmax(np.abs(F['ridge_k3'][0] - zh['ridge_k3'][k]))),
                          ridge_fall16=[float(F['ridge_fall16_h6_s15'][0]), float(zh['ridge_fall16_h6_s15'][k])], img_maxdiff=float(np.max(np.abs(F['img'][0] - zh['img'][k]))))), flush=True)
