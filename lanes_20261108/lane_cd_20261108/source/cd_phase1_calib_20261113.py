"""PI correction (21:33 EST 11-13): the 21:30 read with the Phase I best model (Sept 14 release, official 8.7010, model_full_quality77_ssl_raw_20260914:
six members equal weight = the five crop CPSAM members + full_quality_seed77 (SSL + raw labels), calibration = the OOF SCALE) in place of submission 1.
Per-row Phase I predictions: five members' historical OOF (all five folds) and the sixth member's development OOF of the release recipe (standard SSL,
encoder LR 1e-5; raw_ssl_next_20260913/ssl_runs/fold{2,3,4}/standard/oof.npz), so the Phase I model exists out-of-fold on folds 2, 3, 4 only.
Its calibration SCALE was fitted on the five members' historical OOF of all folds (not nested). Same items as cd_sub1_calib_20261113.py on those rows.
Writes reads/cd_phase1_calib_20261113.json."""
import json, numpy as np
from cd_common import *
import cd_read_20261108 as R
SC = np.array([0.9970446110788349, 0.987972130802936, 0.9892234872488541]); y3 = BP.y.astype(float); pos = {int(i): k for k, i in enumerate(idx)}
HIST = {'seed42': 'followup_20260902/runs/fixed_base_f{f}_seed42_c256k1_mean_multitask_slidesmooth/best_oof.npz',
        'seed77': 'submission_20260904/runs/base_seed77_f{f}_seed77_c256k1_mean_multitask_slidesmooth/best_oof.npz',
        'seed2026': 'submission_20260904/runs/base_seed2026_f{f}_seed2026_c256k1_mean_multitask_slidesmooth/best_oof.npz',
        'metric35': 'methods_20260904/runs/metric35_f{f}_seed42_c256k1_mean_multitask_slidesmooth/best_oof.npz',
        'predbag2_35': 'methods_20260904/runs/predbag2_35_f{f}_seed42_c256k2_mean_multitask_slidesmooth/best_oof.npz'}
def load(pat, folds):
    v = np.full((len(yc), 3), np.nan)
    for f in folds:
        z = np.load(X / pat.format(f=f)); r = [pos[int(i)] for i in z['indices']]; v[r] = z['pred'].astype(float)
    return v
five = np.mean([load(p, range(5)) for p in HIST.values()], 0)
sixth = load('raw_ssl_next_20260913/ssl_runs/fold{f}/standard/oof.npz', (2, 3, 4))
P1h = np.load(X / 'recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True)['historical_five_calibrated'].astype(float)
out = {'five_member_mean_x_SCALE_vs_candidate_P1_max_rel_diff': float(np.nanmax(np.abs(five * SC / P1h - 1)))}
PH = (5 * five + sixth) / 6 * SC                        # Phase I six-member equal weight x SCALE (folds 2-4)
rows = np.isfinite(PH[:, 0]); out['phase1_rows'] = int(rows.sum()); out['phase1_folds'] = sorted(set(int(f) for f in fold[rows]))
t = np.load(LANE / 'reads/cd_tier_t1_20261112.npz'); CA = t['oof_cd'].astype(float); S1 = np.load(X / 'recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True)['B_hybrid_w0p5'][:, 0].astype(float)
ok = (yc > 0) & rows; Q = PH[:, 0]
def m(p, r): r = r & (yc > 0); return float(np.mean(np.abs(p[r] - yc[r]) / yc[r]) * 100)
def m3(P, r):
    v = []
    for j in range(3):
        rr = r & (y3[:, j] > 0); v.append(float(np.mean(np.abs(P[rr, j] - y3[rr, j]) / y3[rr, j]) * 100))
    return v
ph3 = m3(PH, rows); out['phase1_model_folds234_CD_CV_HEX'] = [round(v, 4) for v in ph3]; out['phase1_model_folds234_mean3'] = round(float(np.mean(ph3)), 4)
out['phase1_six_uncalibrated_mean3_folds234'] = round(float(np.mean(m3((5 * five + sixth) / 6, rows))), 4)
ape = lambda p: np.abs(p - yc) / np.where(yc > 0, yc, 1) * 100; lr = lambda p: np.log(p / np.where(yc > 0, yc, 1))
out['1_side_by_side_folds234'] = dict(n=int(ok.sum()), mape_phase1=round(m(Q, rows), 4), mape_candidate=round(m(CA, rows), 4), mape_sub1=round(m(S1, rows), 4),
    corr_log_pred=round(float(np.corrcoef(np.log(Q[ok]), np.log(CA[ok]))[0, 1]), 4), corr_log_residual=round(float(np.corrcoef(lr(Q)[ok], lr(CA)[ok])[0, 1]), 4),
    share_candidate_closer=round(float(np.mean(ape(CA)[ok] < ape(Q)[ok])), 4),
    per_fold={f: dict(phase1=round(m(Q, fold == f), 4), cand=round(m(CA, fold == f), 4)) for f in (2, 3, 4)})
def deciles(key, p):
    e = np.percentile(key[ok], np.linspace(0, 100, 11)); e[-1] += 1; b = np.digitize(key, e) - 1; r = p / np.where(yc > 0, yc, 1) - 1; res = []
    for k in range(10):
        mm = ok & (b == k); res.append(dict(range=[round(float(e[k])), round(float(e[k + 1]))], median_bias_pct=round(float(np.median(r[mm]) * 100), 2), mean_bias_pct=round(float(np.mean(r[mm]) * 100), 2), mape=round(float(ape(p)[mm].mean()), 3)))
    return res
out['2_calibration_folds234'] = {nm: dict(by_pred_decile=deciles(p, p), by_label_decile=deciles(yc, p), slope=round(float(np.polyfit(np.log(p[ok]), np.log(yc[ok]), 1)[0]), 4),
                                          median_bias_pct=round(float(np.median(p[ok] / yc[ok] - 1) * 100), 3)) for nm, p in (('phase1', Q), ('candidate', CA))}
e = np.percentile(Q[ok], np.linspace(0, 100, 11)); e[-1] += 1; b = np.digitize(Q, e) - 1; rr = []
for k in range(10):
    mm = ok & (b == k); rr.append(dict(phase1_range=[round(float(e[k])), round(float(e[k + 1]))], median_cand_over_phase1=round(float(np.median(CA[mm] / Q[mm])), 4),
                                       median_label_over_phase1=round(float(np.median(yc[mm] / Q[mm])), 4), mape_phase1=round(float(ape(Q)[mm].mean()), 3), mape_cand=round(float(ape(CA)[mm].mean()), 3),
                                       share_moved_toward_label=round(float(np.mean(ape(CA)[mm] < ape(Q)[mm])), 3)))
out['3_candidate_vs_phase1_by_phase1_decile'] = rr
res5 = {}; P = CA.copy(); W = {}
for f in (2, 3, 4):
    tr = ok & (fold != f); best = None
    for w in np.round(np.arange(0, 0.501, 0.05), 2):
        q = np.exp((1 - w) * np.log(CA) + w * np.log(np.where(rows, Q, CA))); e_ = m(q, tr)
        if best is None or e_ < best[0]: best = (e_, w)
    W[f] = best[1]; s = fold == f; P[s] = np.exp((1 - best[1]) * np.log(CA[s]) + best[1] * np.log(Q[s]))
d, ci = boot(P, CA, ok)
out['5_blend_phase1_into_candidate_folds234'] = dict(weights_per_fold=W, cd_folds234=round(m(P, rows), 4), cand_folds234=round(m(CA, rows), 4), delta=round(m(P, rows) - m(CA, rows), 4),
                                                     ci95=[round(v, 4) for v in ci], per_fold_delta={f: round(m(P, fold == f) - m(CA, fold == f), 4) for f in (2, 3, 4)},
                                                     note='weight for fold f fitted on the other two of folds 2-4 (the only folds with a Phase I OOF)')
# sixth member alone added to the candidate (the only part not already a member)
P6 = CA.copy(); W6 = {}; s6 = sixth[:, 0] * SC[0]
for f in (2, 3, 4):
    tr = ok & (fold != f); best = None
    for w in np.round(np.arange(0, 0.401, 0.05), 2):
        q = np.exp((1 - w) * np.log(CA) + w * np.log(np.where(rows, s6, CA))); e_ = m(q, tr)
        if best is None or e_ < best[0]: best = (e_, w)
    W6[f] = best[1]; s = fold == f; P6[s] = np.exp((1 - best[1]) * np.log(CA[s]) + best[1] * np.log(s6[s]))
d6, ci6 = boot(P6, CA, ok)
out['5b_sixth_member_alone_blended_folds234'] = dict(weights_per_fold=W6, cd_folds234=round(m(P6, rows), 4), delta=round(m(P6, rows) - m(CA, rows), 4), ci95=[round(v, 4) for v in ci6],
                                                     per_fold_delta={f: round(m(P6, fold == f) - m(CA, fold == f), 4) for f in (2, 3, 4)}, sixth_alone_cd=round(m(s6, rows), 4))
print(json.dumps(out, indent=1)); json.dump(out, open(LANE / 'reads/cd_phase1_calib_20261113.json', 'w'), indent=1)
