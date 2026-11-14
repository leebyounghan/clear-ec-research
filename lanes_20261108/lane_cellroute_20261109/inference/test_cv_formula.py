"""CV assembly check, no models (cell-route lane, 2026-11-11, after the CV lane's 6-component freeze; CPU).
(1) fold 0: recipe_infer.Recipe('fold0').cv_column on every fold-0 row of lane_cv deploy/cv_inputs_oof.npz with entry-style inputs
    (parts['A'][1] = a_raw_cv x sA[0], parts['P1'][1] = p1_cv x phase-1 scalar, as the entry's members produce them) vs CA.apply_fold(Z, 0) and stored cv_oof;
(2) every fold: CA.apply_fold(Z, f) vs stored cv_oof (the CV lane's reproduction, rechecked);
(3) deploy: Recipe('deploy').cv_column on the in-sample deploy members (deploy/cv_deploy_insample_members.npz) vs CA.apply(load_params(), same members).
Also checks that the entry would skip the order-loss DINOv3 (weight 0) and run plain DINOv3 (weight > 0) under both parameter sets."""
import sys, json
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference')
import recipe_infer as RI
CA = RI.CA; E = 'experiments/'
Z = np.load(E + 'lane_cv_20261108/deploy/cv_inputs_oof.npz', allow_pickle=True); names = [str(n) for n in Z['names']]
out = dict(names=names)
R0 = RI.Recipe('manifests/fold0.json', 'params/fold0.json'); RD = RI.Recipe('manifests/deploy.json', 'params/deploy.json')
out['gating'] = {k: dict(dino=R.cv_weight('dino'), orderdino=R.cv_weight('orderdino')) for k, R in (('fold0', R0), ('deploy', RD))}
nan = float('nan')
def parts_of(src, i, sA, s1):
    p = dict(A=[nan, float(src['a_raw_cv'][i]) * sA, nan], P1=[nan, float(src['p1_cv'][i]) * s1, nan], orderdino_logcv=nan,
             DINO_logcv=float(src['dino_logcv'][i]) if 'dino_logcv' in src else nan)
    for q in ('graph_cv', 'ridge_cv', 'qplain_cv', 'qorder_cv', 'fz_std', 'fz_spectrum', 'fz_cov_sqrt', 'fz_autocorr'): p[q] = float(src[q][i])
    return p
r0 = np.where(Z['fold'] == 0)[0]; s1 = float(Z['phase1_scalar_cv'])
mine = np.array([R0.cv_column(parts_of(Z, i, float(Z['sA'][0]), s1)) for i in r0]); af = CA.apply_fold(Z, 0)
out['fold0'] = dict(rows=int(len(r0)), max_rel_vs_apply_fold=float(np.max(np.abs(mine / af - 1))), max_rel_vs_cv_oof=float(np.max(np.abs(mine / Z['cv_oof'][r0] - 1))),
                    median_rel_vs_cv_oof=float(np.median(np.abs(mine / Z['cv_oof'][r0] - 1))))
out['apply_fold_vs_cv_oof'] = {f: float(np.max(np.abs(CA.apply_fold(Z, f) - Z['cv_oof'][Z['fold'] == f]))) for f in range(5)}
try:
    D = np.load(E + 'lane_cv_20261108/deploy/cv_deploy_insample_members.npz', allow_pickle=True); out['insample_keys'] = list(D.files)
    P = CA.load_params(); n = len(D['a_raw_cv'])
    mineD = np.array([RD.cv_column(parts_of(D, i, P['recipeA_scalar_cv'], P['phase1_scalar_cv'])) for i in range(n)])
    keys = ['p1_cv', 'a_raw_cv', 'graph_cv', 'ridge_cv', 'fz_std'] + [CA.TERMS[c][0] for c in P['weights']]
    ref = CA.apply(P, {k: D[k] for k in keys})
    out['deploy_insample'] = dict(rows=int(n), max_rel_vs_CA_apply=float(np.nanmax(np.abs(mineD / ref - 1))), finite=int(np.isfinite(mineD).sum()))
except Exception as e: out['deploy_insample'] = 'skipped: %r' % (e,)
print('SUMMARY', json.dumps(out), flush=True)
json.dump(out, open(E + 'lane_cellroute_20261109/out/cv_formula_check_cv6.json', 'w'), indent=1)
