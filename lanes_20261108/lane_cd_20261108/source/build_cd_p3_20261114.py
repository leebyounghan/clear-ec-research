"""CD deploy constants for p3 = p2 with the ViT-L s46 half replaced by the encoder LR 2e-5 runs, and p3b = p2 with both ViT-L halves replaced
(coordinator 09:16 EST 11-14). Same procedure and format as freeze_cd_params_p2_20261114.json (build_cd_p2_20261114.py): cd_tiers_20261112.py
machinery, strict per-fold fits and the deploy fit with no outer fold (L5). Step 1 writes source/cd_tiers_p3_20261114.py, a copy of
cd_tiers_rerun_20261114.py with the 2e-5 ViT-L members added (OOF: final.pt four-flip TTA, raw track; s46 folds 0 / 4 = sweep_vitl_elr2e5,
1-3 = vitlfu_elr2e5; s47 = vitl47fu_elr2e5) and new output names, and runs it for t1d (must reproduce the stored p2 constants) and the
variant. Step 2 writes reads/freeze_cd_params_<p3|p3b>_20261114.json. usage: VARIANT=p3|p3b python build_cd_p3_20261114.py (from repo root)."""
import os, json, hashlib, subprocess, numpy as np
from pathlib import Path
V = os.environ['VARIANT']; assert V in ('p3', 'p3b'); T = {'p3': 't1d_v46', 'p3b': 't1d_vboth'}[V]
SRC = 'experiments/lane_cd_20261108/source/'; R = 'experiments/lane_cd_20261108/reads/'; LN = 'experiments/lane_cd_20261108/'
code = open(SRC + 'cd_tiers_rerun_20261114.py').read().replace('_rerun_20261114', '_p3_20261114')
anchor = "TIERS = {'inc'"
add = '''def vit_arm(kind):   # ViT-L encoder LR 2e-5 runs, final.pt four-flip TTA, raw track (p3 / p3b, coordinator 09:16 EST 11-14)
    out = np.full(len(yc), np.nan)
    for f in range(5):
        if kind == 's46': r = LANE / (f'arms/sweep_vitl_elr2e5_f{f}_s46' if f in (0, 4) else f'arms/vitlfu_elr2e5_f{f}_s46') / f'raw/runs/fold{f}/dinov3_l16t'
        else: r = LANE / f'arms/vitl47fu_elr2e5_f{f}_s47/raw/runs/fold{f}/dinov3_l16t'
        if not (r / 'predictions_final_tta.npz').exists(): continue
        z = np.load(r / 'predictions_final_tta.npz'); p = np.full(len(yc), np.nan); p[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]
        out[fold == f] = p[fold == f]
    return out


VA46 = vit_arm('s46'); VA47 = vit_arm('s47')
'''
assert code.count(anchor) == 1; code = code.replace(anchor, add + anchor)
old_t = "'t1d': (Afin_raw, D46, Vfin), 't2d': (Afin_raw, D46, None)}"
assert old_t in code
code = code.replace(old_t, "'t1d': (Afin_raw, D46, Vfin), 't2d': (Afin_raw, D46, None),\n         't1d_v46': (Afin_raw, D46, np.exp(0.5 * np.log(VA46) + 0.5 * np.log(vit_final('s47')))), 't1d_vboth': (Afin_raw, D46, np.exp(0.5 * np.log(VA46) + 0.5 * np.log(VA47)))}")
code = code.replace('# copy of cd_tiers_20261112.py writing *_rerun_20261114 files (build_cd_p2_20261114.py)', '# copy of cd_tiers_rerun_20261114.py writing *_p3_20261114 files, with the ViT-L 2e-5 members (build_cd_p3_20261114.py)')
code = code.replace("    Araw, Dk, Vk = TIERS[t]; oof", "    Araw, Dk, Vk = TIERS[t]; assert Vk is None or np.isfinite(Vk[yc > 0]).all(), f'{t}: V missing on some rows'; oof")
open(SRC + 'cd_tiers_p3_20261114.py', 'w').write(code)
env = dict(os.environ, TIER='t1d,' + T, OMP_NUM_THREADS='2'); subprocess.run(['/workspace/venv/bin/python', 'cd_tiers_p3_20261114.py'], cwd=SRC, env=env, check=True, stdout=subprocess.DEVNULL)
p2 = json.load(open(R + 'freeze_cd_params_p2_20261114.json')); chk = json.load(open(R + 'cd_tier_t1d_p3_20261114.json')); new = json.load(open(R + f'cd_tier_{T}_p3_20261114.json'))
keys = ['mask_min_cells', 'mask_a', 'mask_b', 'mask_w', 'stretch_g', 'stretch_c', 'stretch_mu', 'P1_scalar_on_raw', 'p1_cd_multiplier', 'recipeA_cd_scalar']
diff_p2 = max(abs(float(chk['deploy'][k]) - float(p2['deploy'][k])) for k in keys); same_w_p2 = chk['deploy']['weights'] == p2['deploy']['weights']
z = np.load(R + f'cd_tier_{T}_p3_20261114.npz'); zt = np.load(R + 'cd_tier_t1_20261112.npz'); zp2 = np.load(R + 'cd_tier_t1d_rerun_20261114.npz')
yc = z['y_cd']; fold = z['fold']; ok = yc > 0; allr = np.ones(len(yc), bool)
mape = lambda p, m: float(np.mean(np.abs(p[m & ok] - yc[m & ok]) / yc[m & ok]) * 100)
tp, p2s = mape(zt['oof_cd'], allr), mape(zp2['oof_cd'], allr); f13 = ~np.isin(fold, [0, 4])
out = {k: p2[k] for k in p2 if k != 'sha256'}
if V == 'p3b':   # coordinator 10:36 EST 11-14: p3b also against p3 (OOF of t1d_v46 from the p3 build)
    zp3 = np.load(R + 'cd_tier_t1d_v46_p3_20261114.npz'); p3s = mape(zp3['oof_cd'], allr)
    out.update(p3_cd_strict=round(p3s, 5), delta_vs_p3=round(new['strict_five_fold_cd'] - p3s, 5), per_fold_delta_vs_p3={f: round(mape(z['oof_cd'], fold == f) - mape(zp3['oof_cd'], fold == f), 4) for f in range(5)},
               folds_1_3_delta_vs_p3=round(mape(z['oof_cd'], f13) - mape(zp3['oof_cd'], f13), 4))
desc = {'p3': 'p3 (p2 with the ViT-L s46 half replaced by the encoder LR 2e-5 runs; member set t1d_v46)', 'p3b': 'p3b (p2 with both ViT-L halves replaced by the encoder LR 2e-5 runs; member set t1d_vboth)'}[V]
inc = json.load(open(R + 'cd_tier_inc_20261112.json'))['strict_five_fold_cd']
out.update(tier=desc, created='2026-11-14', decision='coordinator 09:16 EST 11-14 (constants written regardless of sign; PI to judge)',
           strict_five_fold_cd=new['strict_five_fold_cd'], tier_p_cd_t1_strict=round(tp, 5), delta_vs_tier_p=round(new['strict_five_fold_cd'] - tp, 5),
           per_fold_delta={f: round(mape(z['oof_cd'], fold == f) - mape(zt['oof_cd'], fold == f), 4) for f in range(5)},
           p2_cd_strict=round(p2s, 5), delta_vs_p2=round(new['strict_five_fold_cd'] - p2s, 5), per_fold_delta_vs_p2={f: round(mape(z['oof_cd'], fold == f) - mape(zp2['oof_cd'], fold == f), 4) for f in range(5)},
           folds_1_3_delta_vs_tier_p=round(mape(z['oof_cd'], f13) - mape(zt['oof_cd'], f13), 4), folds_1_3_delta_vs_p2=round(mape(z['oof_cd'], f13) - mape(zp2['oof_cd'], f13), 4),
           incumbent_D_CD_001_strict_same_run=inc, delta_vs_incumbent=round(new['strict_five_fold_cd'] - inc, 5),
           delta_vs_best_pipeline_as_computed=round(new['strict_five_fold_cd'] - p2['best_pipeline_cd_as_computed'], 4), delta_ci95_donor_bootstrap=None, information_only=False,
           per_fold_delta_note='per_fold_delta is against tier p (t1), per_fold_delta_vs_p2 against p2, fold by fold; folds 0 / 4 chose the ViT-L encoder LR',
           members=['P1', 'A', 'D', 'V', 'mask'], deploy=new['deploy'], per_fold=new['per_fold'],
           L5=dict(procedure='deploy = the strict per-fold fit run with every row as training (no outer fold), same function as the per-fold fits',
                   deploy_params_via_combine_vs_insample_search_max_rel=new['L5_deploy_params_via_combine_vs_insample_search_max_rel'],
                   per_fold_params_via_combine_reproduce_strict_oof_max_rel=new['check_perfold_params_reproduce_oof_max_rel'],
                   machinery_check_t1d_rerun_vs_stored_p2_max_abs_constant_diff=diff_p2, machinery_check_t1d_rerun_vs_stored_p2_weights_equal=same_w_p2,
                   machinery_check_t1d_rerun_strict=chk['strict_five_fold_cd'], weight_on_grid_edge_deploy=new['deploy']['weight_on_grid_edge']),
           oof_file=f'reads/cd_tier_{T}_p3_20261114.npz', code='source/build_cd_p3_20261114.py, source/cd_tiers_p3_20261114.py (copy of cd_tiers_rerun_20261114.py with the 2e-5 ViT-L members)')
hs, hc = Path(R + 'heavy_final_strict_v46_20261114.json'), Path(R + 'heavy_final_cv10_v46_20261114.json')
if V == 'p3b': hs, hc = Path(R + 'heavy_final_strict_vboth_20261114.json'), Path(R + 'heavy_final_cv10_vboth_20261114.json')
hv = {'p3': 'vA46_p2', 'p3b': 'vAboth_p2'}[V]
if hs.exists() and hc.exists():
    s_, c_ = json.load(open(hs))[hv], json.load(open(hc))[hv]
    out['interval_note'] = dict(source=f'{hs.name}, {hc.name} (variant {hv})', strict_vs_tier_p=s_['vs_tierp_all'], strict_vs_p2=s_['vs_p2_all'], cv10_vs_tier_p=c_['vs_tierp_all'], cv10_vs_p2=c_['vs_p2_all'],
                                cv10_folds_1_3_vs_tier_p=c_['vs_tierp_folds1_3'], cv10_folds_1_3_vs_p2=c_['vs_p2_folds1_3'])
    if 'vs_vA46_p2_all' in s_: out['interval_note'].update(strict_vs_p3=s_['vs_vA46_p2_all'], cv10_vs_p3=c_['vs_vA46_p2_all'], cv10_folds_1_3_vs_p3=c_['vs_vA46_p2_folds1_3'])
else:
    out['interval_note'] = 'heavy read not yet available at write time; see the ledger'
vin = {'p3': 'ViT-L full data, raw track, four-flip TTA at final.pt only: exp(0.5 * (log s46 + log s47)), where s46 is the encoder LR 2e-5 full-data build on 9,000 rows (arms/vitlfu_elr2e5_f99_s46, bb_runner_full9000) and s47 the plain full-data s47. No epoch 8-11 single views.',
       'p3b': 'ViT-L full data, raw track, four-flip TTA at final.pt only: exp(0.5 * (log s46 + log s47)), both the encoder LR 2e-5 full-data builds on 9,000 rows (arms/vitlfu_elr2e5_f99_s46, arms/vitl47fu_elr2e5_f99_s47, bb_runner_full9000). No epoch 8-11 single views.'}[V]
out['inputs'] = dict(p2['inputs']); out['inputs']['V'] = vin
rep = {'experiments/lane_cd_20261108/arms/vitl16_f99_s46/raw/runs/fold99/dinov3_l16t/final.pt': LN + 'arms/vitlfu_elr2e5_f99_s46/raw/runs/fold-1/dinov3_l16t/final.pt'}
if V == 'p3b': rep['experiments/lane_cd_20261108/arms/vitl16_f99_s47/raw/runs/fold99/dinov3_l16t/final.pt'] = LN + 'arms/vitl47fu_elr2e5_f99_s47/raw/runs/fold-1/dinov3_l16t/final.pt'
mf = []
for f in p2['model_files']:
    if f['path'] in rep:
        np_ = rep[f['path']]; ex = Path(np_).exists()
        mf.append(dict(path=np_, bytes_fp32=(Path(np_).stat().st_size if ex else None), bytes_fp16_copy=None, shared_with_cv_hex_or_p1=False, exists_at_write_time=ex))
    else: mf.append(f)
out['model_files'] = mf
pk = lambda f: (f['bytes_fp16_copy'] or f['bytes_fp32'] or 0)
out['model_bytes_packaged_all'] = int(sum(pk(f) for f in mf)); out['model_bytes_packaged_cd_only'] = int(sum(pk(f) for f in mf if not f['shared_with_cv_hex_or_p1']))
p = R + f'freeze_cd_params_{V}_20261114.json'; assert not Path(p).exists(), f'{p} exists; not overwritten'
json.dump(out, open(p, 'w'), indent=1)
b = open(p, 'rb').read(); print(json.dumps(dict(file=p, md5=hashlib.md5(b).hexdigest(), sha256=hashlib.sha256(b).hexdigest(), strict=out['strict_five_fold_cd'], tier_p=out['tier_p_cd_t1_strict'],
    delta_tier_p=out['delta_vs_tier_p'], per_fold=out['per_fold_delta'], p2=out['p2_cd_strict'], delta_p2=out['delta_vs_p2'], per_fold_p2=out['per_fold_delta_vs_p2'],
    f13_tier_p=out['folds_1_3_delta_vs_tier_p'], f13_p2=out['folds_1_3_delta_vs_p2'], vs_p3={k: out.get(k) for k in ('p3_cd_strict', 'delta_vs_p3', 'per_fold_delta_vs_p3', 'folds_1_3_delta_vs_p3')}, L5=out['L5'], deploy=out['deploy'], model_files_replaced=[m for m in mf if m['path'] in rep.values()]), indent=1))
