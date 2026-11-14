"""Level of each deployed full-data file against its own fold family (coordinator 11:12 EST 11-14). On 500 evenly spaced training rows
(every other row of the 1,000 used in cd_fulldata_check), each row gets the mean log prediction of the four fold models that trained on it
(in-sample for them, as the full-data model is), in the deployed form, read with the deploy package's members.py (preprocess + four-flip
TTA; DINOv3-B as the mean log over epochs 16-20). Reported per family and metric: mean log ratio full-data / family (%), SD, SE (rows iid)
and SE clustered by donor, slope of full on family. usage: cd_family_level_20261114.py FAMILY [FAMILY ...]; writes reads/family_level_<tag>_20261114.json."""
import os, sys, json, importlib.util, numpy as np, torch
os.chdir('/workspace/clear-ec-research')
PK = 'experiments/lane_cellroute_20261109/package_20261114/p3/context/code/experiments/lane_cellroute_20261109/inference/members.py'
spec = importlib.util.spec_from_file_location('members_p3', PK); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
os.chdir('/workspace/clear-ec-research'); sys.path.insert(0, 'experiments/lane_cd_20261108/source'); os.environ['A2SNAP'] = '1'
from cd_common import fold, idx, donors, C, X
D = 'datasets/clear_ec_fullframe9000_20261113'; img = np.memmap(f'{D}/img_full_raw9000.dat', np.uint8, 'r', shape=(9000, 972, 1296))
fold9 = np.full(9000, -1); fold9[np.asarray(idx)] = fold; don9 = np.full(9000, '', dtype=object); don9[np.asarray(idx)] = np.asarray(donors).astype(str)
TRS = np.unique(np.linspace(0, 8953, 1000).round().astype(int)); TRS = TRS[~np.isin(TRS, [4569, 4570])]; ROWS = TRS[::int(os.environ.get('ROWSTEP', '2'))]   # ROWSTEP=4 -> 250 rows (DINOv3 rerun on the idle GPU 1, 12:45 EST 11-14)
L = 'experiments/lane_cd_20261108/'; LB = 'experiments/lane_backbone_20261108/runs/'
def vl(p): return [p]
FAM = {
 'vitl_s46_plain': ('dinov3_l16t', vl(L + 'arms/vitl16_f99_s46/raw/runs/fold99/dinov3_l16t/final.pt'),
                    {f: vl((LB + f'vitl16_avg_f{f}_s46' if f in (0, 4) else L + f'arms/vitl16_f{f}_s46') + f'/raw/runs/fold{f}/dinov3_l16t/final.pt') for f in range(5)}),
 'vitl_s47_plain': ('dinov3_l16t', vl(L + 'arms/vitl16_f99_s47/raw/runs/fold99/dinov3_l16t/final.pt'),
                    {f: vl(L + f'arms/vitl16_f{f}_s47/raw/runs/fold{f}/dinov3_l16t/final.pt') for f in range(5)}),
 'vitl_s46_2e5': ('dinov3_l16t', vl(L + 'arms/vitlfu_elr2e5_f99_s46/raw/runs/fold-1/dinov3_l16t/final.pt'),
                  {f: vl(L + (f'arms/sweep_vitl_elr2e5_f{f}_s46' if f in (0, 4) else f'arms/vitlfu_elr2e5_f{f}_s46') + f'/raw/runs/fold{f}/dinov3_l16t/final.pt') for f in range(5)}),
 'vitl_s47_2e5': ('dinov3_l16t', vl(L + 'arms/vitl47fu_elr2e5_f99_s47/raw/runs/fold-1/dinov3_l16t/final.pt'),
                  {f: vl(L + f'arms/vitl47fu_elr2e5_f{f}_s47/raw/runs/fold{f}/dinov3_l16t/final.pt') for f in range(5)}),
 'dino_s46b': ('dinov3_b16', [L + f'arms/dinov3_b16_f99_s46b/runs/fold99/dinov3_b16/epoch_{e}.pt' for e in range(16, 21)],
               {f: [('experiments/lane_gpu3_20261106/runs' if f in (0, 4) else L + 'arms') + f'/dinov3_b16_f{f}_s46/runs/fold{f}/dinov3_b16/epoch_{e}.pt' for e in range(16, 21)] for f in range(5)}),
}
for s in (46, 47):
    FAM[f'recipeA_s{s}'] = ('cpdino_recipeA', [f'experiments/fulldata_recipeA_20261028/regression_std_s{s}/runs/fold99/cpdino_vitb/epoch_20.pt'],
                            {f: [str(X / C.RECIPE[s][f] / 'runs' / f'fold{f}' / 'cpdino_vitb' / 'epoch_20.pt')] for f in range(5)})
keys = sys.argv[1:]; XS = {int(r): M.preprocess(np.asarray(img[r])) for r in ROWS}
def logs(paths, fam, rows):
    out = np.zeros((len(paths), len(rows), 3))
    for i, p in enumerate(paths):
        m = M.Member(p, fam)
        for k, r in enumerate(rows): out[i, k] = m.logpred(XS[int(r)], True)
        del m; torch.cuda.empty_cache()
    return out.mean(0)
res = {'rows': len(ROWS)}
for key in keys:
    fam, full, folds = FAM[key]
    if not all(os.path.exists(p) for p in full + sum(folds.values(), [])): res[key] = 'missing files'; continue
    lf = logs(full, fam, ROWS); S_ = np.zeros((len(ROWS), 3)); N_ = np.zeros(len(ROWS))
    for f, paths in folds.items():
        m = fold9[ROWS] != f; S_[m] += logs(paths, fam, ROWS[m]); N_[m] += 1
    assert (N_ == 4).all(); fa = S_ / 4; d = lf - fa
    dn = don9[ROWS]; u, inv = np.unique(dn, return_inverse=True); dm = np.array([np.bincount(inv, d[:, j]) / np.bincount(inv) for j in range(3)]).T
    res[key] = dict(mean_log_ratio_pct=[round(float(v) * 100, 3) for v in d.mean(0)], sd_pct=[round(float(v) * 100, 3) for v in d.std(0, ddof=1)],
                    se_iid_pct=[round(float(v) * 100, 3) for v in d.std(0, ddof=1) / np.sqrt(len(d))], se_donor_pct=[round(float(v) * 100, 3) for v in dm.std(0, ddof=1) / np.sqrt(len(u))],
                    slope=[round(float(np.polyfit(fa[:, j], lf[:, j], 1)[0]), 4) for j in range(3)], donors=int(len(u)),
                    full=full[0] if len(full) == 1 else full[0].rsplit('/', 1)[0], deployed_form=('mean log over epochs 16-20, four-flip TTA' if len(full) > 1 else 'four-flip TTA'))
    print(key, json.dumps(res[key]), flush=True)
json.dump(res, open(L + f'reads/family_level_{"_".join(keys)}_rows{len(ROWS)}_20261114.json', 'w'), indent=1)
