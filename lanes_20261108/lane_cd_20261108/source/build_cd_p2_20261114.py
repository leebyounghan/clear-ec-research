"""CD deploy constants for tier p without the DINOv3 s47 half ('p2', coordinator 02:31 EST 11-14). Same procedure as tier p's CD file
(cd_tiers_20261112.py: strict per-fold fits and the deploy fit with no outer fold), member set t1d = P1, recipe A e20 TTA, DINOv3-B s46b
e16-20 TTA alone, ViT-L s46+s47 final TTA, mask slot, stretch. Step 1 reruns cd_tiers_20261112.py for t1d into new file names (nothing
existing overwritten) and checks that the rerun reproduces the stored 11-12 t1d constants (L5 + reproducibility). Step 2 writes
reads/freeze_cd_params_p2_20261114.json in the format of freeze_cd_params_t1_20261112.json (same keys; D = s46b alone)."""
import os, sys, json, hashlib, subprocess, numpy as np
SRC = 'experiments/lane_cd_20261108/source/'; R = 'experiments/lane_cd_20261108/reads/'
code = open(SRC + 'cd_tiers_20261112.py').read().replace("f'reads/cd_tier_{t}_20261112.json'", "f'reads/cd_tier_{t}_rerun_20261114.json'").replace("f'reads/cd_tier_{t}_20261112.npz'", "f'reads/cd_tier_{t}_rerun_20261114.npz'")
assert 'rerun_20261114' in code
open(SRC + 'cd_tiers_rerun_20261114.py', 'w').write('# copy of cd_tiers_20261112.py writing *_rerun_20261114 files (build_cd_p2_20261114.py)\n' + code)
env = dict(os.environ, TIER='t1d', OMP_NUM_THREADS='2'); subprocess.run(['/workspace/venv/bin/python', 'cd_tiers_rerun_20261114.py'], cwd=SRC, env=env, check=True, stdout=subprocess.DEVNULL)
old = json.load(open(R + 'cd_tier_t1d_20261112.json')); new = json.load(open(R + 'cd_tier_t1d_rerun_20261114.json'))
keys = ['weights', 'mask_min_cells', 'mask_a', 'mask_b', 'mask_w', 'stretch_g', 'stretch_c', 'stretch_mu', 'P1_scalar_on_raw', 'p1_cd_multiplier', 'recipeA_cd_scalar']
diff = max(abs(float(new['deploy'][k]) - float(old['deploy'][k])) for k in keys if k != 'weights'); same_w = new['deploy']['weights'] == old['deploy']['weights']
t1 = json.load(open(R + 'freeze_cd_params_t1_20261112.json'))
out = {k: t1[k] for k in t1 if k not in ('sha256',)}
z = np.load(R + 'cd_tier_t1d_rerun_20261114.npz'); zt = np.load(R + 'cd_tier_t1_20261112.npz'); yc = z['y_cd']; fold = z['fold']; ok = yc > 0
mape = lambda p, m: float(np.mean(np.abs(p[m & ok] - yc[m & ok]) / yc[m & ok]) * 100)
out.update(tier='p2 (tier p CD without the DINOv3 s47 half; member set t1d)', created='2026-11-14', decision='coordinator 02:31 EST 11-14 (deploy speed variant; PI to judge)',
           strict_five_fold_cd=new['strict_five_fold_cd'], tier_p_cd_t1_strict=round(mape(zt['oof_cd'], np.ones(len(yc), bool)), 5),
           delta_vs_tier_p=round(new['strict_five_fold_cd'] - mape(zt['oof_cd'], np.ones(len(yc), bool)), 5),
           per_fold_delta={f: round(mape(z['oof_cd'], fold == f) - mape(zt['oof_cd'], fold == f), 4) for f in range(5)},
           members=['P1', 'A', 'D', 'V', 'mask'], deploy=new['deploy'], per_fold=new['per_fold'],
           L5=dict(procedure='deploy = the strict per-fold fit run with every row as training (no outer fold), same function as the per-fold fits',
                   deploy_params_via_combine_vs_insample_search_max_rel=new['L5_deploy_params_via_combine_vs_insample_search_max_rel'],
                   per_fold_params_via_combine_reproduce_strict_oof_max_rel=new['check_perfold_params_reproduce_oof_max_rel'],
                   rerun_vs_stored_11_12_t1d_max_abs_constant_diff=diff, rerun_vs_stored_weights_equal=same_w, weight_on_grid_edge_deploy=new['deploy']['weight_on_grid_edge']),
           oof_file='reads/cd_tier_t1d_rerun_20261114.npz', code='source/build_cd_p2_20261114.py, source/cd_tiers_rerun_20261114.py (copy of cd_tiers_20261112.py)')
out['inputs'] = dict(t1['inputs']); out['inputs']['D'] = ('DINOv3-B s46b full data ALONE (no s47): four-flip TTA at epoch_16..20.pt, D = exp(mean log over those 20 forwards); '
                                                          'these are the same forwards whose identity flip the CV t1 term already reads')
out['model_files'] = [f for f in t1['model_files'] if 'dinov3_b16_f99_s47' not in f['path']]
pk = lambda f: f['bytes_fp16_copy'] or f['bytes_fp32']
out['model_bytes_packaged_all'] = int(sum(pk(f) for f in out['model_files'])); out['model_bytes_packaged_cd_only'] = int(sum(pk(f) for f in out['model_files'] if not f['shared_with_cv_hex_or_p1']))
p = R + 'freeze_cd_params_p2_20261114.json'; json.dump(out, open(p, 'w'), indent=1)
b = open(p, 'rb').read(); print(json.dumps(dict(file=p, md5=hashlib.md5(b).hexdigest(), sha256=hashlib.sha256(b).hexdigest(), strict=out['strict_five_fold_cd'], tier_p=out['tier_p_cd_t1_strict'],
                                                delta=out['delta_vs_tier_p'], per_fold=out['per_fold_delta'], L5=out['L5'], deploy=out['deploy'], removed_files=len(t1['model_files']) - len(out['model_files']),
                                                cd_only_bytes=out['model_bytes_packaged_cd_only'], keys_t1=sorted(t1.keys()), keys_p2=sorted(out.keys())), indent=1))
