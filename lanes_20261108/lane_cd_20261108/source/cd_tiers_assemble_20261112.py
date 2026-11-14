"""D-PKG-002 assembly: from reads/cd_tier_<t>_20261112.json/.npz (cd_tiers_20261112.py) write reads/freeze_cd_params_t{1,2,3}_20261112.json
(deploy params, per-fold params, strict value vs the incumbent recomputed in the same run, model files with sizes, L5 checks).
Information-only variants t1d/t2d (D = s46 alone) are summarised in reads/cd_tiers_summary_20261112.json. Freeze files are not touched."""
import os, json, hashlib, numpy as np
from cd_common import *
R_ = '/workspace/clear-ec-research/'
F16 = 'experiments/lane_cellroute_20261109/deploy_fp16/'
P1F = [f'experiments/deploy_review_20260905/staged/model/{m}.pt' for m in ('seed42', 'seed77', 'seed2026', 'metric35', 'predbag2_35')] + ['cpsam_models/cpsam_v2']
AF = [f'experiments/fulldata_recipeA_20261028/regression_std_s{s}/runs/fold99/cpdino_vitb/epoch_20.pt' for s in (46, 47)]
DF = [f'experiments/lane_cd_20261108/arms/dinov3_b16_f99_s46b/runs/fold99/dinov3_b16/epoch_{e}.pt' for e in range(16, 21)]
D47F = [f'experiments/lane_cd_20261108/arms/dinov3_b16_f99_s47/runs/fold99/dinov3_b16/epoch_{e}.pt' for e in range(16, 21)]
VF = [f'experiments/lane_cd_20261108/arms/vitl16_f99_s{s}/raw/runs/fold99/dinov3_l16t/final.pt' for s in (46, 47)]
MF = ['cpsam_models/cpsam_v2']
SHARED = set(P1F + AF + DF)    # also needed by the CV / HEX lanes' tiers (P1, recipe A e20 TTA, DINOv3 s46b) or by P1 itself (cpsam_v2)
FILES = {'t1': P1F + AF + DF + D47F + VF, 't2': P1F + AF + DF + D47F, 't3': P1F + AF, 't1d': P1F + AF + DF + VF, 't2d': P1F + AF + DF}
INPUTS = {
 'P1': 'Phase-1 CD as deployed for D-CD-001 (candidate5 members mean x OOF SCALE CD 0.9970446110788349), times deploy.p1_cd_multiplier',
 'A': 'recipe A full-data s46 and s47, four-flip TTA at epoch_20.pt only (the same forward as the CV/HEX lanes\' A): 0.5 * (s46 + s47), times deploy.recipeA_cd_scalar. No epoch 8-19 single views.',
 'D': 'as D-CD-001: DINOv3-B s46b and s47 full data, per seed four-flip TTA at epoch_16..20.pt, mean log over the 20 forwards, then the geometric mean of the two seeds',
 'D46': 'DINOv3-B s46b full data alone: four-flip TTA at epoch_16..20.pt, mean log over the 20 forwards',
 'V': 'ViT-L s46 and s47 full data, raw track, four-flip TTA at final.pt only: exp(0.5 * (log s46 + log s47)). No epoch 8-11 single views.',
 'mask': 'as D-CD-001: Cellpose-SAM cpsam_v2 zero-shot (flow 0.8, cellprob -2, CLAHE), feats() cd_mm2 used only when n_int >= 50'}
USES = {'t1': ['P1', 'A', 'D', 'V', 'mask'], 't2': ['P1', 'A', 'D', 'mask'], 't3': ['P1', 'A', 'mask'], 't1d': ['P1', 'A', 'D46', 'V', 'mask'], 't2d': ['P1', 'A', 'D46', 'mask']}


def sz(p, fp16=False):
    q = R_ + (F16 + p if fp16 else p)
    return os.path.getsize(q) if os.path.exists(q) else None


def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()


T = {t: json.load(open(LANE / f'reads/cd_tier_{t}_20261112.json')) for t in ('inc', 't1', 't2', 't3', 't1d', 't2d')}
Z = {t: np.load(LANE / f'reads/cd_tier_{t}_20261112.npz')['oof_cd'] for t in T}
inc = Z['inc']; frz = json.load(open(LANE / 'reads/freeze_cd_params_20261111.json'))['deploy']
summ = {'incumbent_strict_same_run': T['inc']['strict_five_fold_cd'], 'incumbent_expected': 6.1743,
        'incumbent_deploy_no_outer_fold_vs_frozen_deploy': dict(weights_equal=T['inc']['deploy']['weights'] == frz['weights'], mask_w=(T['inc']['deploy']['mask_w'], frz['mask_w']),
                                                                stretch_g=(T['inc']['deploy']['stretch_g'], frz['stretch_g']), stretch_c=(T['inc']['deploy']['stretch_c'], frz['stretch_c']))}
for t in ('t1', 't2', 't3', 't1d', 't2d'):
    d, ci = boot(Z[t], inc, yc > 0)
    files = []
    for p in FILES[t]:
        files.append(dict(path=p, bytes_fp32=sz(p), bytes_fp16_copy=sz(p, True), shared_with_cv_hex_or_p1=p in SHARED))
    pk = lambda f: f['bytes_fp16_copy'] or f['bytes_fp32']
    out = dict(tier=t, created='2026-11-12', decision='D-PKG-002', metric='CD', information_only=t.endswith('d'),
               strict_five_fold_cd=T[t]['strict_five_fold_cd'], incumbent_D_CD_001_strict_same_run=T['inc']['strict_five_fold_cd'],
               delta_vs_incumbent=round(d, 4), delta_ci95_donor_bootstrap=[round(v, 4) for v in ci],
               per_fold_delta={f: round(cd(Z[t], fold == f) - cd(inc, fold == f), 4) for f in range(5)},
               best_pipeline_cd_as_computed=round(cd(REF), 4), delta_vs_best_pipeline_as_computed=round(cd(Z[t]) - cd(REF), 4),
               members=USES[t], inputs={k: INPUTS[k] for k in USES[t]}, deploy=T[t]['deploy'], per_fold=T[t]['per_fold'],
               combine='cd_combine_20261110.combine(P1_in, A_in, D, V, mask_cd, mask_n_int, deploy) with P1_in = P1 x p1_cd_multiplier, A_in = A x recipeA_cd_scalar; '
                       'a member with weight 0 is not computed: pass 1.0 for it (w * log 1 = 0)',
               L5=dict(procedure='deploy = the strict per-fold fit (scalars, weight grid, mask slot, stretch) run with every row as training, no outer fold; same function as the per-fold fits',
                       deploy_params_via_combine_vs_insample_search_max_rel=T[t]['L5_deploy_params_via_combine_vs_insample_search_max_rel'],
                       per_fold_params_via_combine_reproduce_strict_oof_max_rel=T[t]['check_perfold_params_reproduce_oof_max_rel'],
                       weight_on_grid_edge_deploy=T[t]['deploy']['weight_on_grid_edge']),
               model_files=files, model_bytes_packaged_all=int(sum(pk(f) for f in files)),
               model_bytes_packaged_cd_only=int(sum(pk(f) for f in files if not f['shared_with_cv_hex_or_p1'])),
               oof_file=f'reads/cd_tier_{t}_20261112.npz', code='source/cd_tiers_20261112.py, source/cd_tiers_assemble_20261112.py')
    if not t.endswith('d'):
        pth = LANE / f'reads/freeze_cd_params_{t}_20261112.json'; json.dump(out, open(pth, 'w'), indent=1); out['sha256'] = sha(pth)
    summ[t] = {k: out[k] for k in ('strict_five_fold_cd', 'delta_vs_incumbent', 'delta_ci95_donor_bootstrap', 'per_fold_delta', 'delta_vs_best_pipeline_as_computed',
                                   'model_bytes_packaged_cd_only', 'information_only')}
    summ[t].update(weights=out['deploy']['weights'], mask_w=out['deploy']['mask_w'], grid_edge=out['deploy']['weight_on_grid_edge'], sha256=out.get('sha256'),
                   L5_max_rel=out['L5']['deploy_params_via_combine_vs_insample_search_max_rel'], perfold_check_max_rel=out['L5']['per_fold_params_via_combine_reproduce_strict_oof_max_rel'])
json.dump(summ, open(LANE / 'reads/cd_tiers_summary_20261112.json', 'w'), indent=1); print(json.dumps(summ, indent=1))
