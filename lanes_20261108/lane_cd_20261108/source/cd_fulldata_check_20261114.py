"""Full-data build checks (coordinator 10:52 EST 11-14), inference only, one GPU. Members are read with the deploy package's own code
(experiments/lane_cellroute_20261109/package_20261114/p3/.../inference/members.py: preprocess = CLAHE + 1-99 % scaling of the raw frame,
Member.logpred = four-flip TTA in standardized log space, as deployed). Frames from datasets/clear_ec_fullframe9000_20261113/img_full_raw9000.dat.
(1) The 46 frames the ID match dropped (rows 8954-8999, never seen by the 8,952-row full-data fits): recipe A s46 / s47 epoch_20 TTA, DINOv3-B
s46b epochs 16-20 TTA, plain ViT-L s46 / s47 final.pt TTA; MAPE per metric and mean log residual (SE) next to the member's five-fold OOF.
(2) Level of the ViT-L s46 2e-5 full-data build (vitlfu_elr2e5_f99_s46, 9,000 rows) against the plain full-data s46 on 1,000 evenly spaced
training rows plus the 46: mean log ratio, its SD, slope of log(new) on log(old), per metric; and the same statistics for the fold pairs on OOF
rows (2e-5 fold arm vs plain fold run, final TTA files). Writes reads/fulldata_check_20261114.json."""
import os, sys, json, importlib.util, numpy as np, torch
os.chdir('/workspace/clear-ec-research')
PK = 'experiments/lane_cellroute_20261109/package_20261114/p3/context/code/experiments/lane_cellroute_20261109/inference/members.py'
spec = importlib.util.spec_from_file_location('members_p3', PK); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
os.chdir('/workspace/clear-ec-research')
D = 'datasets/clear_ec_fullframe9000_20261113'; meta = np.load(f'{D}/meta9000.npz', allow_pickle=True); Y = meta['Y'].astype(np.float64); ids = meta['ids'].astype(str)
img = np.memmap(f'{D}/img_full_raw9000.dat', np.uint8, 'r', shape=(9000, 972, 1296))
NEW = np.arange(8954, 9000); TRS = np.unique(np.linspace(0, 8953, 1000).round().astype(int)); TRS = TRS[~np.isin(TRS, [4569, 4570])]
L = 'experiments/lane_cd_20261108/'; RA = 'experiments/fulldata_recipeA_20261028/regression_std_s{}/runs/fold99/cpdino_vitb/epoch_20.pt'
DB = [L + f'arms/dinov3_b16_f99_s46b/runs/fold99/dinov3_b16/epoch_{e}.pt' for e in range(16, 21)]
VP = {s: L + f'arms/vitl16_f99_s{s}/raw/runs/fold99/dinov3_l16t/final.pt' for s in (46, 47)}; VN = L + 'arms/vitlfu_elr2e5_f99_s46/raw/runs/fold-1/dinov3_l16t/final.pt'
def logs(path, fam, rows):
    m = M.Member(path, fam); out = np.zeros((len(rows), 3))
    for k, r in enumerate(rows): out[k] = m.logpred(M.preprocess(np.asarray(img[r])), True)
    del m; torch.cuda.empty_cache(); return out
res = {'rows_new': [int(r) for r in NEW], 'ids_new': list(ids[NEW]), 'note': 'n = 46: only a gross gap (about 2 MAPE points, about 2.5 % in level) would show'}
def stat(lp, rows):
    y = Y[rows]; r = lp - np.log(y); ape = np.abs(np.exp(lp) / y - 1) * 100
    return dict(mape=[round(float(v), 3) for v in ape.mean(0)], mean_log_resid_pct=[round(float(v) * 100, 3) for v in r.mean(0)], se_pct=[round(float(v) * 100, 3) for v in r.std(0, ddof=1) / np.sqrt(len(rows))])
# (1) deployed members on the 46 frames
A46, A47 = logs(RA.format(46), 'cpdino_recipeA', NEW), logs(RA.format(47), 'cpdino_recipeA', NEW)
Dl = np.mean([logs(p, 'dinov3_b16', NEW) for p in DB], 0)
V46, V47 = logs(VP[46], 'dinov3_l16t', NEW), logs(VP[47], 'dinov3_l16t', NEW)
mem = {'recipeA_s46_e20tta': A46, 'recipeA_s47_e20tta': A47, 'recipeA_seedmean_arith': np.log(0.5 * (np.exp(A46) + np.exp(A47))), 'dinov3_s46b_e16_20tta': Dl,
       'vitl_s46_final_tta': V46, 'vitl_s47_final_tta': V47, 'vitl_seed_geomean': 0.5 * (V46 + V47)}
res['new46'] = {k: stat(v, NEW) for k, v in mem.items()}
# five-fold OOF of the same members (all labelled rows of the old set), same statistics
sys.path.insert(0, L + 'source'); os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
from cd_common import yc, fold, LANE, X
y3 = R.BP.y.astype(np.float64); okr = (y3 > 0).all(1)
def tta3(name):
    o = np.full((len(yc), 3), np.nan)
    for f in range(5):
        z = np.load(LANE / 'tta' / f'{name}_f{f}_tta4_e16_20.npz'); p = np.full((len(yc), 3), np.nan); p[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, :3]; o[fold == f] = p[fold == f]
    return o
def vfin(sd, arm=False):
    o = np.full((len(yc), 3), np.nan)
    for f in range(5):
        if arm: r = LANE / (f'arms/sweep_vitl_elr2e5_f{f}_s46' if f in (0, 4) else f'arms/vitlfu_elr2e5_f{f}_s46') / f'raw/runs/fold{f}/dinov3_l16t'
        else: r = (X / f'lane_backbone_20261108/runs/vitl16_avg_f{f}_s46' if (f in (0, 4) and sd == 's46') else LANE / f'arms/vitl16_f{f}_{sd}') / f'raw/runs/fold{f}/dinov3_l16t'
        z = np.load(r / 'predictions_final_tta.npz'); p = np.full((len(yc), 3), np.nan); p[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, :3]; o[fold == f] = p[fold == f]
    return o
oof = {'recipeA_s46_e20tta': np.asarray(R.seeds[46], float)[:, :3], 'recipeA_s47_e20tta': np.asarray(R.seeds[47], float)[:, :3]}
oof['recipeA_seedmean_arith'] = 0.5 * (oof['recipeA_s46_e20tta'] + oof['recipeA_s47_e20tta']); oof['dinov3_s46b_e16_20tta'] = tta3('dinov3_s46')
oof['vitl_s46_final_tta'] = vfin('s46'); oof['vitl_s47_final_tta'] = vfin('s47'); oof['vitl_seed_geomean'] = np.sqrt(oof['vitl_s46_final_tta'] * oof['vitl_s47_final_tta'])
def ostat(p):
    m = okr & np.isfinite(p).all(1); r = np.log(p[m] / y3[m]); ape = np.abs(p[m] / y3[m] - 1) * 100
    return dict(mape=[round(float(v), 3) for v in ape.mean(0)], mean_log_resid_pct=[round(float(v) * 100, 3) for v in r.mean(0)], rows=int(m.sum()))
res['oof_fivefold'] = {k: ostat(v) for k, v in oof.items()}
# (2) level: new 2e-5 full-data ViT-L s46 vs plain full-data s46
rows2 = np.concatenate([TRS, NEW]); Lold, Lnew = logs(VP[46], 'dinov3_l16t', rows2), logs(VN, 'dinov3_l16t', rows2)
def lvl(a, b):   # a = new / arm, b = old / plain, log predictions (n, 3)
    d = a - b; sl = [float(np.polyfit(b[:, j], a[:, j], 1)[0]) for j in range(3)]
    return dict(mean_log_ratio_pct=[round(float(v) * 100, 3) for v in d.mean(0)], sd_log_ratio_pct=[round(float(v) * 100, 3) for v in d.std(0, ddof=1)], slope=[round(v, 4) for v in sl], rows=int(len(a)))
res['fulldata_new_vs_old_s46'] = dict(train_rows=lvl(Lnew[:len(TRS)], Lold[:len(TRS)]), new46=lvl(Lnew[len(TRS):], Lold[len(TRS):]), all=lvl(Lnew, Lold))
ARM, PL = vfin(None, arm=True), vfin('s46')
res['fold_pairs_oof'] = {f'fold{f}': lvl(np.log(ARM[(fold == f) & okr]), np.log(PL[(fold == f) & okr])) for f in range(5)}
res['fold_pairs_oof']['pooled'] = lvl(np.log(ARM[okr]), np.log(PL[okr]))
json.dump(res, open(L + 'reads/fulldata_check_20261114.json', 'w'), indent=1); print(json.dumps({k: v for k, v in res.items() if k not in ('rows_new', 'ids_new')}, indent=1))
