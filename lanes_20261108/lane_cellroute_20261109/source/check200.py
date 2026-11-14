"""Cell-route lane (2026-11-09): do features from the full-data pre-only detectors (sg2pre_s46 deployed, s47/s48 new) track the OOF features
from the fold detectors? 200 random frames, GPU 1: per-feature Pearson r and median relative difference, and the corrected HEX
(deploy params) vs the stored OOF correction."""
import sys, time, json
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/source')
import deploy_hexkonan_image as DI
import deploy_hexkonan as D
ROOT = '/workspace/clear-ec-research/'; L = ROOT + 'experiments/lane_cellroute_20261109/'
sys.path.insert(0, ROOT + 'experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
MM = np.memmap(ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
hk = DI.HexKonan([ROOT + 'experiments/green_track_20261028/deploy/detectors/sg2pre_s46.pt', ROOT + 'experiments/green_track_20261028/deploy/detectors/sg2pre_s47.pt', ROOT + 'experiments/green_track_20261028/deploy/detectors/sg2pre_s48.pt'], L + 'out/deploy_hexkonan_params.json', device='cuda')
zs = np.load(L + 'feats/seg_centroids.npz'); _p, _o, _i = zs['pts'], zs['offsets'], zs['indices']; pos = {int(d): k for k, d in enumerate(_i)}
zf = np.load(L + 'feats/cellroute_feats.npz'); NM = list(zf['names']); FF = zf['F'][:, [NM.index(c) for c in D.NAMES]]
base = np.load(ROOT + 'experiments/lane_hex_20261108/preds/chain_cvw3.npz')['final_hex']; oofc = np.load(L + 'out/corr_hex_a100.npz')['P'][:, 2]
brow = {int(d): k for k, d in enumerate(BP.idx.astype(int))}
rows = np.random.default_rng(1).choice([d for d in brow], 200, replace=False); Fn, Fo, hn, ho, hb = [], [], [], [], []; t0 = time.time()
for r in rows:
    k = pos[int(r)]; hc, F = hk(MM[r], _p[_o[k]:_o[k + 1]], base[brow[int(r)]]); Fn.append(F); Fo.append(FF[r]); hn.append(hc); ho.append(oofc[brow[int(r)]]); hb.append(base[brow[int(r)]])
Fn, Fo = np.array(Fn), np.array(Fo); out = {}
for j, nm in enumerate(D.NAMES):
    m = np.isfinite(Fn[:, j]) & np.isfinite(Fo[:, j]); out[nm] = dict(r=round(float(np.corrcoef(Fn[m, j], Fo[m, j])[0, 1]), 3), med_rel_diff=round(float(np.median(np.abs(Fn[m, j] / Fo[m, j] - 1))), 4))
hn, ho, hb = np.array(hn), np.array(ho), np.array(hb)
out['corrected_hex'] = dict(median_abs_log_diff_vs_oof_correction=round(float(np.median(np.abs(np.log(hn / ho)))), 5), median_abs_log_change_vs_base_deploy=round(float(np.median(np.abs(np.log(hn / hb)))), 5),
                            median_abs_log_change_vs_base_oof=round(float(np.median(np.abs(np.log(ho / hb)))), 5), seconds_per_frame=round((time.time() - t0) / len(rows), 3))
print(json.dumps(out, indent=1)); json.dump(out, open(L + 'out/check200.json', 'w'), indent=1)
