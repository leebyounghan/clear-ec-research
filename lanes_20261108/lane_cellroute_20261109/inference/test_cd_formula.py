"""CD assembly check, no models (cell-route lane, 2026-11-11, after the CD lane's seed audit: D = two-seed mean, freeze_cd_params_20261111.json; CPU).
(1) fold 0: recipe_infer.Recipe('fold0').cd_column on every fold-0 row of lane_cd reads/freeze_cd_oof_20261111.npz (stored P1, A, D, V, mask) vs stored oof_cd;
(2) every fold: cd_combine_20261110.combine with each fold's params vs oof_cd (the CD lane's check, here per fold);
(3) the stored D column vs exp(0.5 log D46 + 0.5 log D47) from reads/seedmean_d_oof_20261111.npz (what the entry's D must reproduce)."""
import sys, json
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference')
import recipe_infer as RI
E = 'experiments/'; Z = np.load(E + 'lane_cd_20261108/reads/freeze_cd_oof_20261111.npz'); P = json.load(open(E + 'lane_cd_20261108/reads/freeze_cd_params_20261111.json'))
R0 = RI.Recipe('manifests/fold0.json', 'params/fold0.json'); out = dict(fold0_params_equal_lane=R0.p['cd'] == P['per_fold']['0'], fold0_dino_cd_seeds=len(R0.m['dino_cd']))
def parts_of(i):
    s = np.full(17, np.nan); s[5] = Z['mask_cd'][i]; s[3] = Z['mask_n_int'][i]
    return dict(P1=[float(Z['P1'][i]), np.nan, np.nan], A_cd=float(Z['A'][i]), D=float(Z['D'][i]), V=float(Z['V'][i]), seg17=s)
r0 = np.where(Z['fold'] == 0)[0]; mine = np.array([R0.cd_column(parts_of(i)) for i in r0])
out['fold0'] = dict(rows=int(len(r0)), max_rel_vs_oof_cd=float(np.max(np.abs(mine / Z['oof_cd'][r0] - 1))))
out['per_fold_combine_vs_oof_cd'] = {}
for f in range(5):
    m = Z['fold'] == f; c = RI.CDC.combine(Z['P1'][m], Z['A'][m], Z['D'][m], Z['V'][m], Z['mask_cd'][m], Z['mask_n_int'][m], P['per_fold'][str(f)])
    out['per_fold_combine_vs_oof_cd'][f] = float(np.max(np.abs(c / Z['oof_cd'][m] - 1)))
S = np.load(E + 'lane_cd_20261108/reads/seedmean_d_oof_20261111.npz'); assert np.array_equal(S['idx'], Z['idx'])
out['D_column_vs_geomean_D46_D47'] = float(np.nanmax(np.abs(Z['D'] / np.sqrt(S['D46'] * S['D47']) - 1)))
y = Z['y_cd']; ok = y > 0; out['five_fold_cd_from_oof_cd'] = round(float(np.mean(np.abs(Z['oof_cd'][ok] - y[ok]) / y[ok]) * 100), 4)
print('SUMMARY', json.dumps(out), flush=True); json.dump(out, open(E + 'lane_cellroute_20261109/out/cd_formula_check_d2.json', 'w'), indent=1)
