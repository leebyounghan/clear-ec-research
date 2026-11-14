"""HEX graph-net pinning: frame 32 (fold 3) with its STORED fold-detector single-view peaks (pre3 + the other 13) through Recipe.hex_features ->
gnn_hex_infer outputs vs each full-data build's in-sample prediction (NAME_fullfit_pred.npz, best-pipeline row order); plus g_s10x3 / g_hex46."""
import sys, json, numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference')
import recipe_infer as RI, members as M
E = 'experiments/'; MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
zc = np.load(E + 'lane_cd_20261108/reads/freeze_cd_oof_20261109.npz'); B = np.load(E + 'recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True)
k = int(np.where(zc['fold'] == 3)[0][7]); d = int(zc['idx'][k]); g = 3
def st(det):
    z = np.load(E + f'green_track_20261028/{det}/fold{g}/fold{g}_sg2_peaks.npz', allow_pickle=True); j = list(z['indices']).index(d); return z['peaks'][z['offsets'][j]:z['offsets'][j + 1]].astype(np.float32)
R = RI.Recipe('manifests/deploy.json', 'params/deploy.json'); cd = float(B['B_hybrid_w0p5'][list(B['indices']).index(d), 0])
F, G, gout = R.hex_features(np.asarray(MM[d]), [st(x) for x in RI.ALL16[:3]], {x: st(x) for x in RI.ALL16[3:]}, cd)
GM = E + 'lane_hex_20261108/deploy/graph_models/'; out = dict(frame=d, nets_found=sorted(gout))
for key, (name, gset) in R.m['hex_graph_nets'].items():
    ref = np.load(GM + f'{name}_fullfit_pred.npz')['pred'][k]; out[key] = dict(mine=np.round(gout[key], 6).tolist(), insample=np.round(ref, 6).tolist(), maxdiff=float(np.max(np.abs(gout[key] - ref))))
print(json.dumps(out), flush=True)
