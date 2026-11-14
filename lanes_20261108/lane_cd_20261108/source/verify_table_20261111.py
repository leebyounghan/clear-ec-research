"""Verifier table 11-11 (CD lane as verifier, coordinator's framework revision): recompute the three-metric five-fold table from each lane's own
OOF files, as computed and strictly nested, for D-CD-001, D-CV-001, D-HEX-004 + D-HEX-005 and the best pipeline; row-set checks; L5 for CD.
CPU only. Writes reads/verify_table_20261111.json."""
import os, sys, json, itertools, numpy as np
os.environ['A2SNAP'] = '1'
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import stretch_nested_20261103 as N
import cd_read_20261108 as R
import cd_stack_20261109 as K
import cd_combine_20261110 as CB
from cd_common import *
E_ = X
y3 = BP.y.astype(float); f5 = BP.fold; out = {}


def mape(p, j, rows=None):
    ok = y3[:, j] > 0 if rows is None else (rows & (y3[:, j] > 0)); return float(np.mean(np.abs(p[ok] - y3[ok, j]) / y3[ok, j]) * 100)


def folds(p, j, ref):
    return {f: round(mape(p, j, f5 == f) - mape(ref, j, f5 == f), 4) for f in range(5)}


# ---- row order / row sets
chk = {}
zc = np.load(E_ / 'lane_cd_20261108/reads/freeze_cd_oof_20261111.npz'); chk['cd_idx_equal_bp'] = bool((zc['idx'] == BP.idx).all())
zv = np.load(E_ / 'lane_cv_20261108/deploy/cv_final_oof.npz', allow_pickle=True); chk['cv_idx_equal_bp'] = bool((zv['idx'] == BP.idx).all())
zv0 = np.load(E_ / 'lane_cv_20261108/out/cv_final_20261110.npz', allow_pickle=True); chk['cv_out_equals_deploy_oof_P'] = float(np.nanmax(np.abs(zv0['P'] - zv['P'])))
zh = np.load(E_ / 'lane_hex_20261108/preds/chain_joint_attnp.npz'); chk['hex_idx_equal_bp'] = bool((zh['idx'] == BP.idx).all())
zr = np.load(E_ / 'lane_cellroute_20261109/out/corr_hex_on_joint_attnp.npz', allow_pickle=True); zrs = np.load(E_ / 'lane_cellroute_20261109/out/corr_hex_strict_on_joint_attnp.npz', allow_pickle=True)
chk['cellroute_idx_equal_bp'] = bool((zr['idx'] == BP.idx).all() and (zrs['idx'] == BP.idx).all())
chk['fold_equal_everywhere'] = bool((zc['fold'] == f5).all() and (zv['fold'] == f5).all() and (zh['fold'] == f5).all() and (zr['fold'] == f5).all())
chk['n_rows'] = int(len(f5)); chk['n_scored'] = {m: int((y3[:, j] > 0).sum()) for j, m in enumerate(('CD', 'CV', 'HEX'))}
chk['zero_label_rows'] = {m: [int(i) for i in np.flatnonzero(y3[:, j] <= 0)] for j, m in enumerate(('CD', 'CV', 'HEX'))}
chk['cd_label_equal'] = bool(np.allclose(zc['y_cd'], y3[:, 0]))
out['rows'] = chk; print('rows', json.dumps(chk), flush=True)

# ---- as computed and strict values from the lanes' own files
v = {}
v['CD D-CD-001 as computed'] = mape(zc['oof_cd'], 0)
v['CD D-CD-001 strict (scalars refit per outer fold)'] = mape(np.load(E_ / 'lane_cd_20261108/reads/strict_nesting_20261111.npz')['v3_strict'], 0)
v['CV D-CV-001 as computed (cv_final_20261110 P)'] = mape(zv0['P'][:, 1], 1)
v['CV D-CV-001 as computed (deploy/cv_final_oof cv)'] = mape(zv['cv'], 1)
v['CV D-CV-001 strict (search_fixed6, inner stretch excl. outer fold)'] = mape(np.load(E_ / 'lane_cv_20261108/out/search_fixed6_20261111.npz')['P'], 1)
v['HEX D-HEX-004 as computed (chain_joint_attnp final_hex)'] = mape(zh['final_hex'], 2)
v['HEX D-HEX-004+005 as computed (cell-route corr on joint_attnp)'] = mape(zr['P'], 2)
v['HEX D-HEX-004+005 strict (cell-route corr strict)'] = mape(zrs['P'], 2)
for j, m in enumerate(('CD', 'CV', 'HEX')):
    v[f'{m} best pipeline as computed (REF_BEST)'] = mape(BP.REF_BEST[:, j], j)
# other columns of the lanes' files equal REF_BEST?
chk2 = {'cv_file_CD_col_vs_REF': float(np.nanmax(np.abs(zv0['P'][:, 0] / BP.REF_BEST[:, 0] - 1))), 'cv_file_HEX_col_vs_REF': float(np.nanmax(np.abs(zv0['P'][:, 2] / BP.REF_BEST[:, 2] - 1)))}

# ---- best pipeline strict: P1 SCALE and recipe A LOFO scalars refit per outer fold, per column (HEX h6 constants and BEST corrections as shipped)
SCALE = np.array([0.9970446110788349, 0.987972130802936, 0.9892234872488541])
P1raw = R.b['P1'] / SCALE; Araw = 0.5 * (R.seeds[46] + R.seeds[47])
lofo = json.load(open(E_ / 'recipeA_analysis_20261028/hybrid_oof.json'))['recipeA_lofo_scalars']
SC = np.array([lofo[str(f)] for f in f5])
B_rec = 0.5 * Araw * SC + 0.5 * R.b['P1']; chk2['hybrid_B_reconstruction_max_rel'] = float(np.nanmax(np.abs(B_rec / BP.B - 1)))
have = BP._have; f6 = BP.f6


def H_of(B):
    H = B.copy(); H[have, 2] = np.exp((1 - 0.28) * np.log(B[have, 2]) + 0.28 * (-0.2084 + 0.5989 * f6[have])); return H


chk2['H_reconstruction_max_rel'] = float(np.nanmax(np.abs(H_of(B_rec) / BP.H - 1)))
ref_strict = np.full_like(BP.H, np.nan)
for f in range(5):
    tr = f5 != f; Bf = np.empty_like(BP.B)
    for j in range(3):
        ok = tr & (y3[:, j] > 0)
        sP = C.wmed(y3[ok, j] / P1raw[ok, j], P1raw[ok, j] / y3[ok, j]); sA = C.wmed(y3[ok, j] / Araw[ok, j], Araw[ok, j] / y3[ok, j])
        Bf[:, j] = 0.5 * Araw[:, j] * sA + 0.5 * P1raw[:, j] * sP
    Cf = H_of(Bf) * np.exp(BP.BEST[f]); ref_strict[f5 == f] = N.apply_stretch(Cf[f5 == f], N.fit_stretch(Cf, y3, tr & BP.good))
for j, m in enumerate(('CD', 'CV', 'HEX')):
    v[f'{m} best pipeline strict (scalars refit per outer fold)'] = mape(ref_strict[:, j], j)
out['values'] = {k: round(x, 5) for k, x in v.items()}; out['checks'] = chk2
print(json.dumps(out['values'], indent=1), json.dumps(chk2), flush=True)

# ---- integrated means
cdA, cdS = zc['oof_cd'], np.load(E_ / 'lane_cd_20261108/reads/strict_nesting_20261111.npz')['v3_strict']
cvA, cvS = zv0['P'][:, 1], np.load(E_ / 'lane_cv_20261108/out/search_fixed6_20261111.npz')['P']
hxA, hxS = zr['P'], zrs['P']
all3 = (y3 > 0).all(1)


def integ(cdp, cvp, hxp):
    per = [mape(cdp, 0), mape(cvp, 1), mape(hxp, 2)]
    rowwise = float(np.mean([np.mean(np.abs(p[all3] - y3[all3, j]) / y3[all3, j]) * 100 for j, p in enumerate((cdp, cvp, hxp))]))
    return dict(per_metric=[round(x, 5) for x in per], mean_of_metric_means=round(float(np.mean(per)), 5), mean_on_common_rows=round(rowwise, 5), common_rows=int(all3.sum()))


out['integrated'] = {'candidate as computed': integ(cdA, cvA, hxA), 'candidate strict': integ(cdS, cvS, hxS),
                     'best pipeline as computed': integ(BP.REF_BEST[:, 0], BP.REF_BEST[:, 1], BP.REF_BEST[:, 2]),
                     'best pipeline strict': integ(ref_strict[:, 0], ref_strict[:, 1], ref_strict[:, 2])}
print(json.dumps(out['integrated'], indent=1), flush=True)
out['fold_deltas_strict_vs_bp_strict'] = {'CD': folds(cdS, 0, ref_strict[:, 0]), 'CV': folds(cvS, 1, ref_strict[:, 1]), 'HEX': folds(hxS, 2, ref_strict[:, 2])}

# ---- L5 for CD: the strict nested code with no outer fold vs the deploy parameters of D-CD-001
dep = json.load(open(E_ / 'lane_cd_20261108/reads/freeze_cd_params_20261111.json')); D_ = dep['deploy']
Asnap_raw = R.A2 / R.sc; Dm = zc['D']; Vm = zc['V']; allr = np.ones(len(yc), bool)
sP_all = C.wmed(yc[yc > 0] / P1raw[yc > 0, 0], P1raw[yc > 0, 0] / yc[yc > 0]); sS_all = C.wmed(yc[yc > 0] / Asnap_raw[yc > 0], Asnap_raw[yc > 0] / yc[yc > 0])
Lc = [np.log(P1raw[:, 0] * sP_all), np.log(Asnap_raw * sS_all), np.log(Dm), np.log(Vm)]; grids = []
for a in np.round(np.arange(0.25, 0.751, 0.05), 3):
    for mw in itertools.product(np.round(np.arange(0, 0.401, 0.05), 3), repeat=2):
        r = 1 - a - sum(mw)
        if r >= 0.099: grids.append((round(r, 3), a) + tuple(mw))
best = None
for wt in grids:
    q = np.exp(sum(w * l for w, l in zip(wt, Lc))); e = cd(apply1(q, fit_stretch1(q, allr & good)), allr)
    if best is None or e < best[0]: best = (e, wt, q)
q = best[2]; trm = K.have & (yc > 0) & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
for wmk in (0, 0.05, 0.1, 0.15, 0.2):
    qq = np.where(K.have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, allr & good); e = cd(apply1(qq, p2), allr)
    if b2 is None or e < b2[0]: b2 = (e, wmk, qq, p2)
strict_params = dict(weights=dict(zip(('P1', 'A', 'D', 'V'), best[1])), mask_a=float(co[1]), mask_b=float(co[0]), mask_w=b2[1], stretch_g=float(b2[3][0]), stretch_c=float(b2[3][1]), stretch_mu=float(b2[3][2]),
                     P1_scale=float(sP_all), A_scalar=float(sS_all))
# predictions on the OOF inputs: deploy convention (P1 x SCALE, A snapshot x 0.99139) vs strict-no-outer-fold convention
pd = CB.combine(R.P1, Asnap_raw * dep['recipeA_cd_scalar_deploy'], Dm, Vm, zc['mask_cd'], zc['mask_n_int'], D_)
ps = apply1(b2[2], b2[3])
out['L5_CD'] = dict(deploy_params={k: D_[k] for k in ('weights', 'mask_a', 'mask_b', 'mask_w', 'stretch_g', 'stretch_c', 'stretch_mu')},
                    deploy_scalars=dict(P1_SCALE=SCALE[0], A_scalar=dep['recipeA_cd_scalar_deploy']), strict_no_outer_fold=strict_params,
                    pred_max_rel_diff=float(np.max(np.abs(pd / ps - 1))), pred_median_rel_diff=float(np.median(np.abs(pd / ps - 1))),
                    insample_cd_deploy=round(cd(pd), 5), insample_cd_strict=round(cd(ps), 5))
print('L5_CD', json.dumps(out['L5_CD']), flush=True)
json.dump(out, open(E_ / 'lane_cd_20261108/reads/verify_table_20261111.json', 'w'), indent=1); print('saved', flush=True)
