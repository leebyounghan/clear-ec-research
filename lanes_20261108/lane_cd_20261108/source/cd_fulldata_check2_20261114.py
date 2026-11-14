"""Follow-up of cd_fulldata_check_20261114.py (11:00 EST 11-14). (a) Checkpoint target statistics (log_mean / log_std) of the full-data ViT-L
builds: plain s46 (bb_runner_cd, 8,952 rows) vs the 2e-5 build (bb_runner_full9000, 9,000 rows). (b) Like-for-like level: the fold pairs
(2e-5 fold arm vs plain fold run, final.pt TTA, deploy-package reader) on their own TRAINING rows (the same 1,000 evenly spaced rows, minus
the fold's own rows), as the full-data pair was read on rows both had trained on; fold 0 and fold 4; plus the fold's own rows as a check
against the stored OOF numbers. (c) Fold models on the 46 new frames (plain ViT-L s46 and recipe A s46, folds 0 and 4), to see whether the
46-frame HEX / CV residuals are frame-set effects (fold models show them too) or full-data effects. Writes reads/fulldata_check2_20261114.json."""
import os, sys, json, importlib.util, numpy as np, torch
os.chdir('/workspace/clear-ec-research')
PK = 'experiments/lane_cellroute_20261109/package_20261114/p3/context/code/experiments/lane_cellroute_20261109/inference/members.py'
spec = importlib.util.spec_from_file_location('members_p3', PK); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
os.chdir('/workspace/clear-ec-research')
D = 'datasets/clear_ec_fullframe9000_20261113'; meta = np.load(f'{D}/meta9000.npz', allow_pickle=True); Y = meta['Y'].astype(np.float64)
img = np.memmap(f'{D}/img_full_raw9000.dat', np.uint8, 'r', shape=(9000, 972, 1296))
NEW = np.arange(8954, 9000); TRS = np.unique(np.linspace(0, 8953, 1000).round().astype(int)); TRS = TRS[~np.isin(TRS, [4569, 4570])]
L = 'experiments/lane_cd_20261108/'; LB = 'experiments/lane_backbone_20261108/runs/'
sys.path.insert(0, L + 'source'); os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
from cd_common import fold, idx, C, X
fold9 = np.full(9000, -1); fold9[np.asarray(idx)[:len(fold)] if len(idx) == len(fold) else idx] = fold   # memmap row -> fold (rows 0-8953 follow idx)
res = {}
ck = lambda p: torch.load(p, map_location='cpu', weights_only=False)
for nm, p in (('plain_f99_s46', L + 'arms/vitl16_f99_s46/raw/runs/fold99/dinov3_l16t/final.pt'), ('elr2e5_f99_s46_9000', L + 'arms/vitlfu_elr2e5_f99_s46/raw/runs/fold-1/dinov3_l16t/final.pt'),
              ('dino_elr2e5_f99_9000', L + 'arms/fu2_elr2e5_f99_s46/raw/runs/fold-1/dinov3_b16/final.pt')):
    c = ck(p); res[f'stats_{nm}'] = dict(log_mean=[float(v) for v in c['log_mean']], log_std=[float(v) for v in c['log_std']])
def logs(path, fam, rows):
    m = M.Member(path, fam); out = np.zeros((len(rows), 3))
    for k, r in enumerate(rows): out[k] = m.logpred(M.preprocess(np.asarray(img[r])), True)
    del m; torch.cuda.empty_cache(); return out
def lvl(a, b):
    d = a - b; ok = np.isfinite(d).all(1); d = d[ok]; a, b = a[ok], b[ok]
    return dict(mean_log_ratio_pct=[round(float(v) * 100, 3) for v in d.mean(0)], sd_log_ratio_pct=[round(float(v) * 100, 3) for v in d.std(0, ddof=1)],
                slope=[round(float(np.polyfit(b[:, j], a[:, j], 1)[0]), 4) for j in range(3)], rows=int(len(d)))
def stat(lp, rows):
    y = Y[rows]; r = lp - np.log(y); ape = np.abs(np.exp(lp) / y - 1) * 100
    return dict(mape=[round(float(v), 3) for v in ape.mean(0)], mean_log_resid_pct=[round(float(v) * 100, 3) for v in r.mean(0)], se_pct=[round(float(v) * 100, 3) for v in r.std(0, ddof=1) / np.sqrt(len(rows))])
for f in (0, 4):
    arm = L + f'arms/sweep_vitl_elr2e5_f{f}_s46/raw/runs/fold{f}/dinov3_l16t/final.pt'; pl = LB + f'vitl16_avg_f{f}_s46/raw/runs/fold{f}/dinov3_l16t/final.pt'
    rows = np.concatenate([TRS, NEW]); la, lp = logs(arm, 'dinov3_l16t', rows), logs(pl, 'dinov3_l16t', rows)
    fr = fold9[TRS]; tr_in = fr != f; own = fr == f; nT = len(TRS)
    res[f'fold{f}_pair'] = dict(training_rows=lvl(la[:nT][tr_in], lp[:nT][tr_in]), own_fold_rows_oof=lvl(la[:nT][own], lp[:nT][own]), new46=lvl(la[nT:], lp[nT:]))
    res[f'fold{f}_plain_vitl_s46_on_new46'] = stat(lp[nT:], NEW); res[f'fold{f}_arm_vitl_s46_on_new46'] = stat(la[nT:], NEW)
    ra = X / C.RECIPE[46][f] / 'runs' / f'fold{f}' / 'cpdino_vitb' / 'epoch_20.pt'
    res[f'fold{f}_recipeA_s46_e20_on_new46'] = stat(logs(str(ra), 'cpdino_recipeA', NEW), NEW)
json.dump(res, open(L + 'reads/fulldata_check2_20261114.json', 'w'), indent=1); print(json.dumps(res, indent=1))
