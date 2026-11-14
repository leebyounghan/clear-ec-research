"""Offline parameter and manifest builder for recipe_infer.py (cell-route lane, 2026-11-10). This script may read lane parameter files, stored OOF
predictions and labels, because it FITS or COLLECTS parameters; the inference entry only reads the JSON files written here.
Writes params/{fold0,deploy}.json, manifests/{fold0,deploy}.json and params/cellroute_{fold0,deploy}.json; prints every manifest path that is missing."""
import json, os, sys
import numpy as np
os.chdir('/workspace/clear-ec-research'); E = 'experiments/'; HERE = E + 'lane_cellroute_20261109/inference/'
sys.path[:0] = [E + 'lane_cellroute_20261109/source', E + 'meta_ceiling_20261103/source']
os.makedirs(HERE + 'params', exist_ok=True); os.makedirs(HERE + 'manifests', exist_ok=True)
HXP = json.load(open(E + 'lane_hex_20261108/preds/deploy_params.json'))
# HEX form (11-11): 'joint' = the HEX lane's joint nested ridge (preds/deploy_params_joint.json, hex_apply.hex_any; reads every hex_apply input, g_hex46 included),
# 'chain' = the step chain fallback (preds/deploy_params.json). The lane's whole parameter file is embedded in params/*.json with the fold (0 / None = deploy).
HEX_FORM = os.environ.get('HEX_FORM', 'joint_attnp'); assert HEX_FORM in ('joint_attnp', 'joint', 'chain')   # joint_attnp: 39-column ridge with the attention-pool member attnp5
HXFILE = {'joint_attnp': 'deploy_params_joint_attnp.json', 'joint': 'deploy_params_joint.json'}.get(HEX_FORM)
HXJ = json.load(open(E + 'lane_hex_20261108/preds/' + HXFILE)) if HXFILE else None
ATTNP = HXJ is not None and 'attnp5(1)' in HXJ['columns']
AP_D = E + 'lane_backbone_20261108/runs/attnpool_f99_dinov3b16_s46/raw/runs/fold99/dinov3_b16/'
HEX_EXTRA_D = {'attnp5': dict(read='attnpool_hex', fam='dinov3_b16', col=2, ckpts=[AP_D + f'epoch_{e:02d}.pt' for e in range(8, 12)] + [AP_D + 'final.pt'])} if ATTNP else {}
HSTEPS = [q['step'] for q in HXP['steps']]
HEX46 = HXJ is not None or 'gnn:hex_pre46_s5' in HSTEPS
CVW3M = ('cvw3_mean(1)' in HXJ['columns']) if HXJ is not None else 'cvw3m_fix' in HSTEPS   # cvw3m: dino_cvw3 = exp mean log over s46 + s47 (10 checkpoints)
HEX_F0 = dict(form=HEX_FORM, P=HXJ, fold=0, source='lane_hex_20261108/preds/' + str(HXFILE)) if HXJ is not None else \
    dict(steps=[dict(step=q['step'], pick=q['nested'][0]['pick'], params=q['nested'][0]['params']) for q in HXP['steps']], stretch=HXP['nested_stretch'][0])
HEX_D = dict(form=HEX_FORM, P=HXJ, fold=None, source='lane_hex_20261108/preds/' + str(HXFILE)) if HXJ is not None else \
    dict(steps=[dict(step=q['step'], pick=q['deploy_pick'], params=q['params']) for q in HXP['steps']], stretch=HXP['stretch'])
print('HEX form:', HEX_FORM, '| hex46 graph net:', HEX46, '| cvw3 two seeds:', CVW3M, '| attnp5:', ATTNP, flush=True)
# CD: seed audit 2026-11-11 (CD lane): D = two-seed mean (s46, s47), params freeze_cd_params_20261111.json (only the stretch changed).
# Deploy uses it only once the full-data s47 build exists; until then deploy keeps D = s46 alone with the 20261109 params (coordinator rule:
# if s47 full data is not built by 15:00 EST, deploy that fallback). Fold 0 always uses the two-seed member (fold-0 s47 exists).
CDP = json.load(open(E + 'lane_cd_20261108/reads/freeze_cd_params_20261111.json')); CDP_OLD = json.load(open(E + 'lane_cd_20261108/reads/freeze_cd_params_20261109.json'))
S47_F99 = E + 'lane_cd_20261108/arms/dinov3_b16_f99_s47/runs/fold99/dinov3_b16/'
D2_DEPLOY = os.path.exists(S47_F99 + 'complete.json') and os.environ.get('CD_D_SINGLE') != '1'
CD_DEPLOY = CDP['deploy'] if D2_DEPLOY else CDP_OLD['deploy']
CD_NOTE = ('D = two-seed mean s46b + s47 (full data), freeze_cd_params_20261111.json deploy' if D2_DEPLOY else
           'D = s46b alone, freeze_cd_params_20261109.json deploy (fallback: full-data s47 not built or CD_D_SINGLE=1)')
print('CD deploy:', CD_NOTE, flush=True)
P1_SCALE = [0.9970446110788349, 0.987972130802936, 0.9892234872488541]   # hybrid_oof_20261028.SCALE (Phase-1 five-member scalars)
RECIPE = {46: {0: 'phase2_pooling_replication_20261026/f0_s46_std', 1: 'oof_recipeA_20261027/fold1/regression_std_s46', 2: 'oof_recipeA_20261027/fold2/regression_std_s46_v2',
               3: 'oof_recipeA_20261027/fold3/regression_std_s46', 4: 'phase2_pooling_20261020/arms/std'},
          47: {0: 'phase2_pooling_replication_20261026/f0_s47_std', 4: 'phase2_pooling_replication_20261026/f4_s47_std', **{f: f'oof_recipeA_20261027/fold{f}/regression_std_s47' for f in (1, 2, 3)}}}
HIST = ['followup_20260902/runs/fixed_base_f{f}_seed42_c256k1_mean_multitask_slidesmooth/best.pt', 'submission_20260904/runs/base_seed77_f{f}_seed77_c256k1_mean_multitask_slidesmooth/best.pt',
        'submission_20260904/runs/base_seed2026_f{f}_seed2026_c256k1_mean_multitask_slidesmooth/best.pt', 'methods_20260904/runs/metric35_f{f}_seed42_c256k1_mean_multitask_slidesmooth/best.pt',
        'methods_20260904/runs/predbag2_35_f{f}_seed42_c256k2_mean_multitask_slidesmooth/best.pt']


# --- recipe A scalars (CV/HEX A = mean of seeds 46, 47 TTA@e20, times weighted-median scalars, as hybrid_oof_20261028.fit_scalars) ---
z = np.load(E + 'recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True); idx, fold, y = z['indices'], z['fold'], z['target']; pos = {int(i): k for k, i in enumerate(idx)}
raw = np.full((len(idx), 3), np.nan)
for f in range(5):
    q = [np.load(E + RECIPE[s][f] + f'/runs/fold{f}/cpdino_vitb/predictions_tta4.npz') for s in (46, 47)]
    for k, i in enumerate(q[0]['indices']): raw[pos[int(i)]] = np.mean([q[0]['pred_tta4'][k], q[1]['pred_tta4'][list(q[1]['indices']).index(i)]], 0)
def wmed(v, w):
    o = np.argsort(v); c = np.cumsum(w[o]); return float(v[o][np.searchsorted(c, c[-1] / 2.)])
def fit_scalars(p, t): return [wmed(t[t[:, j] > 0, j] / p[t[:, j] > 0, j], p[t[:, j] > 0, j] / t[t[:, j] > 0, j]) for j in range(3)]
A_fold0 = fit_scalars(raw[fold != 0], y[fold != 0]); A_deploy = fit_scalars(raw, y)
chk = z['A_recipeA_tta_scalar'][fold == 0] / raw[fold == 0]; assert np.allclose(chk, A_fold0, rtol=1e-9), (chk.mean(0), A_fold0)

# --- cell-route ridge parameters (fold 0 = fit on folds != 0; deploy = all folds), same fit as source/fit_deploy_hexkonan.py ---
import deploy_hexkonan as DK
import best_pipeline_20261106 as BP
import ceiling_meta_cpu_20261103 as C
zf = np.load(E + 'lane_cellroute_20261109/feats/cellroute_feats.npz'); NM = list(zf['names']); Fr = zf['F'][:, [NM.index(c) for c in DK.NAMES]][BP.idx.astype(int)]
HEXBASE = 'lane_hex_20261108/preds/' + {'joint_attnp': 'chain_joint_attnp.npz', 'joint': 'chain_joint.npz', 'chain': 'chain_seedaudit10.npz'}[HEX_FORM]   # HEX lane final chain (11-11 seed audit: gnn:hex_pre46_s5 dropped, cvw3 two seeds; was chain_mtcnn5fix)
base = np.load(E + HEXBASE)['final_hex']
def fit_cr(rows, base=base, basefile=HEXBASE):
    X, m = DK.design(Fr, base); ok = BP.y[:, 2] > 0; res = np.log(np.where(ok, BP.y[:, 2], 1.0)) - np.log(base); tr = rows & BP.good & m & ok
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; A = (X[tr] - mu) / sd; w = np.linalg.solve(A.T @ A + 100.0 * np.eye(A.shape[1]), A.T @ (res[tr] - res[tr].mean()))
    q = base * np.exp(((X - mu) / sd) @ w); c = C.wmed(BP.y[tr, 2] / q[tr], q[tr] / BP.y[tr, 2]); c0 = C.wmed(BP.y[tr, 2] / base[tr], base[tr] / BP.y[tr, 2])
    return dict(mu=mu.tolist(), sd=sd.tolist(), w=w.tolist(), c=float(c), c0=float(c0), alpha=100.0, names=DK.NAMES + ['log_base_hex'], train_rows=int(tr.sum()), base=basefile + ' final_hex')
json.dump(fit_cr(BP.fold != 0), open(HERE + 'params/cellroute_fold0.json', 'w'), indent=1)
CRD = fit_cr(np.ones(len(BP.y), bool)); json.dump(CRD, open(HERE + 'params/cellroute_deploy.json', 'w'), indent=1); json.dump(CRD, open(E + 'lane_cellroute_20261109/out/deploy_hexkonan_params.json', 'w'), indent=1)

ZC = np.load(E + 'lane_cv_20261108/deploy/cv_inputs_oof.npz', allow_pickle=True)
CVF0 = dict(weights=dict(zip([str(n) for n in ZC['names']], [float(v) for v in ZC['w0']])), recipeA_scalar_cv=float(ZC['sA'][0]), phase1_scalar_cv=float(ZC['phase1_scalar_cv']),
            stretch_cv=[float(v) for v in ZC['stretch0']], source='lane_cv_20261108/deploy/cv_inputs_oof.npz w0 / sA[0] / stretch0')
assert abs(CVF0['recipeA_scalar_cv'] - A_fold0[1]) < 1e-6, (CVF0['recipeA_scalar_cv'], A_fold0)
G_ = E + 'green_track_20261028/'
DET16_F0 = {'sg2_specialist': ['full', G_ + 'sg2_specialist/fold0/specialist_sg2.pt'], **{f'sg2_specialist_s{s}': ['full', G_ + f'sg2_specialist_s{s}/fold0/specialist_sg2.pt'] for s in range(47, 53)},
            'sg2_sigma3p5': ['full', G_ + 'sg2_sigma3p5/fold0/specialist_sg2.pt'], 'sg2_sigma5': ['full', G_ + 'sg2_sigma5/fold0/specialist_sg2.pt'],
            **{f'frozen_heads/{k}': ['frozen', G_ + f'frozen_heads/{k}/fold0/head.pt'] for k in ('sg2f_s46', 'sg2f_s47', 'sg2f_s48', 'sg2pref_s46')}}
GD_, HD_ = E + 'green_track_20261028/deploy/', E + 'lane_hex_20261108/deploy/'
DET16_D = {'sg2_specialist': ['full', GD_ + 'detectors/sg2_s46.pt'], **{f'sg2_specialist_s{s}': ['full', GD_ + f'detectors/sg2_s{s}.pt'] for s in range(47, 51)},
           'sg2_specialist_s51': ['full', HD_ + 'detectors/sg2_specialist_s51/detectors/sg2_s51.pt'], 'sg2_specialist_s52': ['full', HD_ + 'detectors/sg2_specialist_s52/detectors/sg2_s52.pt'],
           'sg2_sigma3p5': ['full', HD_ + 'detectors/sg2_sigma3p5/detectors/sg2_s46.pt'], 'sg2_sigma5': ['full', HD_ + 'detectors/sg2_sigma5/detectors/sg2_s46.pt'],
           **{f'frozen_heads/{k}': ['frozen', HD_ + f'frozen_heads/{k}/fold0/head.pt'] for k in ('sg2f_s46', 'sg2f_s47', 'sg2f_s48', 'sg2pref_s46')}}   # = det_peaks_image.FULLDATA
P = {'fold0': dict(mode='fold0', cd=CDP['per_fold']['0'], recipeA_cd_scalar=CDP['recipeA_cd_scalar_per_fold']['0'], A_scalars=A_fold0, P1_scale=P1_SCALE,
                   hex=HEX_F0,
                   cellroute_params='params/cellroute_fold0.json', cv=CVF0),
     'deploy': dict(mode='deploy', cd=CD_DEPLOY, cd_note=CD_NOTE, recipeA_cd_scalar=CDP['recipeA_cd_scalar_deploy'], A_scalars=A_deploy, A_scalars_note='fit here on all rows (two-seed CV/HEX scalars are stored by no lane)',
                    P1_scale=P1_SCALE, hex=HEX_D,
                    cellroute_params='params/cellroute_deploy.json', cv=dict(deploy_dir=E + 'lane_cv_20261108/deploy'))}
ra = lambda s, f: E + RECIPE[s][f] + f'/runs/fold{f}/cpdino_vitb/'
FD = E + 'fulldata_recipeA_20261028/regression_std_s{s}/runs/fold99/cpdino_vitb/'
MAN = {'fold0': dict(recipeA_tta=[ra(s, 0) + 'epoch_20.pt' for s in (46, 47)], recipeA_cd=[dict(e20=ra(s, 0) + 'epoch_20.pt', single=[ra(s, 0) + f'epoch_{e:02d}.pt' for e in range(8, 20)]) for s in (46, 47)],
                     dino_cd=[[E + f'{r}/runs/fold0/dinov3_b16/epoch_{e}.pt' for e in range(16, 21)] for r in ('lane_gpu3_20261106/runs/dinov3_b16_f0_s46', 'lane_gpu2_20261106/runs/dinov3_b16_f0_s47')],
                     dino_cv=[[E + f'{r}/runs/fold0/dinov3_b16/epoch_{e}.pt' for e in range(16, 21)] for r in ('lane_gpu3_20261106/runs/dinov3_b16_f0_s46', 'lane_gpu2_20261106/runs/dinov3_b16_f0_s47', 'lane_cv_20261108/arms/dinov3_b16_f0_s49')],
                     vitl_seeds=[dict(single=[], tta=E + 'lane_backbone_20261108/runs/vitl16_avg_f0_s46/raw/runs/fold0/dinov3_l16t/final.pt'), dict(single=[], tta=E + 'lane_cd_20261108/arms/vitl16_f0_s47/raw/runs/fold0/dinov3_l16t/final.pt')],
                     vitl_note='fold runs kept final.pt only: the epoch 8-11 single-view halves are stubs',
                     orderdino_cv=[E + f'lane_cv_20261108/arms/dinov3_order_f0_s46/runs/fold0/dinov3_b16/epoch_{e}.pt' for e in range(16, 21)],
                     detectors_all16=DET16_F0, frozen_encoder=ra(46, 0) + 'epoch_20.pt',
                     cps5=[E + f'lane_hex_20261108/runs/mtcps_nossl_c512_f0/runs/fold0/cpsam_c512/epoch_{e:02d}.pt' for e in range(8, 13)],
                     p1_members=[E + h.format(f=0) for h in HIST], p1_cpsam='cpsam_models/cpsam_v2',
                     preonly_detectors=[E + 'green_track_20261028/external_srep/' + d + '/sg2_pre.pt' for d in ('fold0', 's47/fold0', 's48/fold0')], cpsam_seg='cpsam_models/cpsam_v2'),
       'deploy': dict(recipeA_tta=[FD.format(s=s) + 'epoch_20.pt' for s in (46, 47)], recipeA_cd=[dict(e20=FD.format(s=s) + 'epoch_20.pt', single=[FD.format(s=s) + f'epoch_{e:02d}.pt' for e in range(8, 20)]) for s in (46, 47)],
                      dino_cd=[[E + f'lane_cd_20261108/arms/{r}/runs/fold99/dinov3_b16/epoch_{e}.pt' for e in range(16, 21)] for r in (('dinov3_b16_f99_s46b', 'dinov3_b16_f99_s47') if D2_DEPLOY else ('dinov3_b16_f99_s46b',))],
                      dino_cv=[[E + f'{r}/runs/fold99/dinov3_b16/epoch_{e}.pt' for e in range(16, 21)] for r in ('lane_cd_20261108/arms/dinov3_b16_f99_s46b', 'lane_cv_20261108/arms/dinov3_b16_f99_s47', 'lane_cv_20261108/arms/dinov3_b16_f99_s49')],
                      vitl_seeds=[dict(single=[E + f'lane_cd_20261108/arms/vitl16_f99_{sd}/raw/runs/fold99/dinov3_l16t/epoch_{e:02d}.pt' for e in range(8, 12)], tta=E + f'lane_cd_20261108/arms/vitl16_f99_{sd}/raw/runs/fold99/dinov3_l16t/final.pt') for sd in ('s46', 's47')],
                      vitl_note='not built yet; paths are a guess at the queued builds (CD lane to confirm)',
                      orderdino_cv=[E + f'lane_cv_20261108/arms/dinov3_order_f99_s46/runs/fold99/dinov3_b16/epoch_{e}.pt' for e in range(16, 21)],
                      hex_graph_models=E + 'lane_hex_20261108/deploy/graph_models', hex_graph_nets={'pre47_s10': ['full_d64_l5_pre47_s10', 'pre47'], 'pre48_s10': ['full_d64_l5_pre48_s10', 'pre48'], **({'hex46': ['full_hex_pre46_s5', 'pre46']} if HEX46 else {})},
                      cvw3=[E + f'lane_hex_20261108/deploy/cvw3_f99_{s}/runs/fold99/dinov3_b16/epoch_{e:02d}.pt' for s in (('s46', 's47') if CVW3M else ('s46',)) for e in range(8, 13)],
                      cps5=[E + f'lane_hex_20261108/deploy/mtcps_c512_f99/runs/fold99/cpsam_c512/epoch_{e:02d}.pt' for e in range(8, 13)],
                      detectors_all16=DET16_D, cv_parts=dict(token_encoder=FD.format(s=46) + 'epoch_20.pt', graph_models=E + 'lane_cv_20261108/graph_models', frozen_heads=E + 'lane_cv_20261108/frozen_heads_full'),
                      p1_members=[E + f'deploy_review_20260905/staged/model/{m}.pt' for m in ('seed42', 'seed77', 'seed2026', 'metric35', 'predbag2_35')], p1_cpsam='cpsam_models/cpsam_v2',   # D-PKG-001: byte-identical to staged/model/cpsam_v2 (dup_check.json), one copy ships
                      p1_note='candidate5 package (CD lane table: base_inference candidate5.json members; calibration = OOF SCALE, not candidate5.json calibration)',
                      preonly_detectors=[E + f'green_track_20261028/deploy/detectors/sg2pre_s{s}.pt' for s in (46, 47, 48)], cpsam_seg='cpsam_models/cpsam_v2',
                      hex6_deploy_dir=E + 'green_track_20261028/deploy', **({'hex_extra': HEX_EXTRA_D} if HEX_EXTRA_D else {}))}
# decision ids this build implements (ledger_20261111.jsonl; framework revision 11-11): the deploy follows the highest id per metric
DEC = ['D-CD-001' if D2_DEPLOY else 'D-CD-001 fallback (D = s46b alone)', 'D-CV-001',
       {'joint_attnp': 'D-HEX-004', 'joint': 'D-HEX-004 fallback 1 (38-column ridge)', 'chain': 'D-HEX-004 fallback 2 (10-step chain)'}[HEX_FORM], 'D-HEX-005']
print('decisions:', DEC, flush=True)
for k in ('fold0', 'deploy'):
    P[k]['decisions'] = DEC; MAN[k]['decisions'] = DEC
    json.dump(P[k], open(HERE + f'params/{k}.json', 'w'), indent=1); json.dump(MAN[k], open(HERE + f'manifests/{k}.json', 'w'), indent=1)
    miss = []
    def walk(v):
        if isinstance(v, str) and (v.endswith('.pt') or v.startswith('cpsam')) and not os.path.exists(v): miss.append(v)
        elif isinstance(v, list): [walk(u) for u in v]
        elif isinstance(v, dict): [walk(u) for u in v.values()]
    walk(MAN[k]); print(k, 'missing files:', len(miss), json.dumps(miss[:12]), flush=True)
print('A scalars fold0', np.round(A_fold0, 6).tolist(), 'deploy', np.round(A_deploy, 6).tolist())
# ---- packaging tiers (D-PKG-002, 2026-11-12): params/deploy_{t}.json + manifests/deploy_{t}.json next to the freeze files (which stay as written above).
# CD: lane_cd reads/freeze_cd_params_{t}_20261112.json; CV: lane_cv deploy_{t}/; HEX: lane_hex preds/deploy_params_joint_{t}.json (t1 = the freeze file);
# cell-route: kept version refit with no outer fold on that tier's HEX OOF (L5). Manifest entries a tier never reads are removed.
TIERS = dict(t1=dict(hexp='deploy_params_joint_attnp.json', hexoof='chain_joint_attnp.npz', dino_cv=('s46b', range(16, 21))),
             t2=dict(hexp='deploy_params_joint_t2.json', hexoof='chain_joint_t2.npz', dino_cv=('s46b', range(16, 21))),
             t3=dict(hexp='deploy_params_joint_t3.json', hexoof='chain_joint_t3.npz', dino_cv=('s46b', range(20, 21))),
             p=dict(cd='t1', cv='t1', hexp='deploy_params_joint_t2.json', hexoof='chain_joint_t2.npz', dino_cv=('s46b', range(16, 21))))   # D-PKG-003 primary: CD t1 + CV t1 + HEX t2
for t, spec in TIERS.items():
    tcd, tcv = spec.get('cd', t), spec.get('cv', t)
    files = dict(cd=E + f'lane_cd_20261108/reads/freeze_cd_params_{tcd}_20261112.json', cv=E + f'lane_cv_20261108/deploy_{tcv}/cv_deploy_params.json',
                 hexp=E + 'lane_hex_20261108/preds/' + spec['hexp'], hexoof=E + 'lane_hex_20261108/preds/' + spec['hexoof'])
    if not all(os.path.exists(v) for v in files.values()): print(t, 'skipped: missing', [k for k, v in files.items() if not os.path.exists(v)], flush=True); continue
    cdt = json.load(open(files['cd'])); hxt = json.load(open(files['hexp'])); ob = np.load(files['hexoof'])
    assert (ob['idx'].astype(int) == BP.idx.astype(int)).all()
    crt = fit_cr(np.ones(len(BP.y), bool), base=ob['final_hex'], basefile='lane_hex_20261108/preds/' + spec['hexoof'])
    json.dump(crt, open(HERE + f'params/cellroute_deploy_{t}.json', 'w'), indent=1)
    Pt = json.loads(json.dumps(P['deploy'])); Mt = json.loads(json.dumps(MAN['deploy']))
    Pt.update(cd=cdt['deploy'], cd_note=f"CD tier {t}: {files['cd'][len(E):]} (strict {cdt['strict_five_fold_cd']})", cv=dict(deploy_dir=E + f'lane_cv_20261108/deploy_{tcv}'),
              hex=dict(form=f'joint_{t}', P=hxt, fold=None, source='lane_hex_20261108/preds/' + spec['hexp']), cellroute_params=f'params/cellroute_deploy_{t}.json',
              decisions=[('D-PKG-003 primary tier p' if t == 'p' else f'D-PKG-002 tier {t}'), f'CD {files["cd"][len(E):]}', f'CV lane_cv_20261108/deploy_{tcv}', f'HEX lane_hex_20261108/preds/{spec["hexp"]}', 'cell-route kept version (D-HEX-005) refit on the tier HEX OOF'])
    w = cdt['deploy']['weights']; names = hxt.get('blocks') or [c.split('(')[0] for c in hxt['columns']]
    Mt.pop('recipeA_cd', None)                                                     # every tier: recipe A e20 TTA only for CD
    if w.get('V', 0): Mt['vitl_seeds'] = [dict(single=[], tta=s['tta']) for s in Mt['vitl_seeds']]
    else: Mt.pop('vitl_seeds', None)
    if not w.get('D', 0): Mt.pop('dino_cd', None)
    sd, ep = spec['dino_cv']; Mt['dino_cv'] = [[E + f'lane_cd_20261108/arms/dinov3_b16_f99_{sd}/runs/fold99/dinov3_b16/epoch_{e}.pt' for e in ep]]
    if 'mtcnn5' not in names: Mt.pop('cps5', None)
    if 'cvw3_mean' not in names: Mt.pop('cvw3', None)
    if 'attnp5' not in names: Mt.pop('hex_extra', None)
    if 'ridge_fall16' not in names: Mt.pop('detectors_all16', None)
    if 'gnn46' not in names: Mt['hex_graph_nets'] = {k: v for k, v in Mt['hex_graph_nets'].items() if k != 'hex46'}
    Mt['decisions'] = Pt['decisions']
    json.dump(Pt, open(HERE + f'params/deploy_{t}.json', 'w'), indent=1); json.dump(Mt, open(HERE + f'manifests/deploy_{t}.json', 'w'), indent=1)
    miss = []
    walk(Mt); print(t, 'written; manifest keys', sorted(Mt), '| missing files:', len(miss), flush=True)
