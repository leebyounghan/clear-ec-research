"""PI request via coordinator (21:30 EST 11-13): submission 1's CD OOF (hybrid_oof.npz B_hybrid_w0p5, 6.312) vs the candidate's strict CD OOF
(reads/cd_tier_t1_20261112.npz, 6.1737), CPU only, existing arrays. 1 side by side; 2 calibration of each vs labels by decile of prediction and of
label, log-log slope; 3 candidate vs sub-1 by decile of sub-1; 4 recalibration of the candidate fitted out-of-fold on the other four folds' strict
OOF (a global MAPE-optimal factor, b log-affine (stretch form), c1 per-predicted-quintile factor, c2 isotonic in log + factor); 5 blend of sub-1 (and
of the best pipeline's stretched sub-1, REF) into the candidate with one log weight fitted out-of-fold. Fits use (fold != f) & good & label > 0.
Writes reads/cd_sub1_calib_20261113.json."""
import json, numpy as np
from cd_common import *
ok = yc > 0
z = np.load(X / 'recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True); assert (z['indices'] == idx).all() and (z['fold'] == fold).all()
S1 = z['B_hybrid_w0p5'][:, 0].astype(float); A1 = z['A_recipeA_tta_scalar'][:, 0].astype(float); P1h = z['historical_five_calibrated'][:, 0].astype(float)
t = np.load(LANE / 'reads/cd_tier_t1_20261112.npz'); assert (t['idx'] == idx).all(); CA = t['oof_cd'].astype(float)
out = {'baseline_candidate_t1_strict': round(cd(CA), 4), 'sub1_oof': round(cd(S1), 4), 'best_pipeline_REF_sub1_with_nested_stretch': round(cd(REF), 4)}
out['sub1_is_0.5A_plus_0.5P1_max_rel_diff'] = float(np.max(np.abs((0.5 * A1 + 0.5 * P1h) / S1 - 1)))
out['candidate_P1_input_vs_sub1_P1_corr_log'] = float(np.corrcoef(np.log(t['P1_in'][ok]), np.log(P1h[ok]))[0, 1])
out['candidate_A_input_vs_sub1_A_ratio_median'] = float(np.median(t['A_in'][ok] / A1[ok]))
ape = lambda p: np.abs(p - yc) / np.where(ok, yc, 1) * 100
# 1
lr = lambda p: np.log(p / np.where(ok, yc, 1))
out['1_side_by_side'] = dict(mape_sub1=round(cd(S1), 4), mape_candidate=round(cd(CA), 4),
                             corr_log_pred=round(float(np.corrcoef(np.log(S1[ok]), np.log(CA[ok]))[0, 1]), 4),
                             corr_log_residual=round(float(np.corrcoef(lr(S1)[ok], lr(CA)[ok])[0, 1]), 4),
                             share_candidate_closer=round(float(np.mean(ape(CA)[ok] < ape(S1)[ok])), 4), share_sub1_closer=round(float(np.mean(ape(S1)[ok] < ape(CA)[ok])), 4),
                             per_fold_mape={f: dict(sub1=round(cd(S1, fold == f), 4), cand=round(cd(CA, fold == f), 4)) for f in range(5)})
# 2
def deciles(key, p):
    e = np.percentile(key[ok], np.linspace(0, 100, 11)); e[-1] += 1; b = np.digitize(key, e) - 1; r = p / np.where(ok, yc, 1) - 1; res = []
    for k in range(10):
        m = ok & (b == k); res.append(dict(range=[round(float(e[k])), round(float(e[k + 1]))], n=int(m.sum()), median_bias_pct=round(float(np.median(r[m]) * 100), 2),
                                           mean_bias_pct=round(float(np.mean(r[m]) * 100), 2), mape=round(float(ape(p)[m].mean()), 3)))
    return res
def slope(p): return round(float(np.polyfit(np.log(p[ok]), np.log(yc[ok]), 1)[0]), 4)
out['2_calibration'] = {nm: dict(by_pred_decile=deciles(p, p), by_label_decile=deciles(yc, p), slope_loglabel_on_logpred=slope(p),
                                 overall_median_bias_pct=round(float(np.median(p[ok] / yc[ok] - 1) * 100), 3)) for nm, p in (('sub1', S1), ('candidate', CA))}
# 3
e = np.percentile(S1[ok], np.linspace(0, 100, 11)); e[-1] += 1; b = np.digitize(S1, e) - 1; rows = []
for k in range(10):
    m = ok & (b == k); rows.append(dict(sub1_range=[round(float(e[k])), round(float(e[k + 1]))], median_cand_over_sub1=round(float(np.median(CA[m] / S1[m])), 4),
                                        median_label_over_sub1=round(float(np.median(yc[m] / S1[m])), 4), mape_sub1=round(float(ape(S1)[m].mean()), 3), mape_cand=round(float(ape(CA)[m].mean()), 3),
                                        share_moved_toward_label=round(float(np.mean(ape(CA)[m] < ape(S1)[m])), 3)))
out['3_candidate_vs_sub1_by_sub1_decile'] = rows
# 4
def iso_fit(x, yv):   # pool-adjacent-violators, increasing, unit weights; returns knots (x sorted, fitted)
    o = np.argsort(x); xs, ys = x[o], yv[o]; vals, wts, cnt = [], [], []
    for v in ys:
        vals.append(v); wts.append(1.0); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop(); vals[-1] = (vals[-1] * wts[-1] + v2 * w2) / (wts[-1] + w2); wts[-1] += w2; cnt[-1] += c2
    fit = np.repeat(vals, cnt); return xs, fit
def recal(kind, f):
    tr = (fold != f) & good & ok; te = fold == f; p = CA
    if kind == 'a_global':
        s = C.wmed(yc[tr] / p[tr], p[tr] / yc[tr]); return p[te] * s
    if kind == 'b_logaffine':
        prm = fit_stretch1(p, tr); return apply1(p[te], prm)
    if kind == 'c1_quintile_factor':
        q = np.percentile(p[tr], [20, 40, 60, 80]); bt = np.digitize(p, q); outp = p[te].copy()
        for k in range(5):
            m = tr & (bt == k); s = C.wmed(yc[m] / p[m], p[m] / yc[m]); outp[bt[te] == k] *= s
        return outp
    if kind == 'c2_isotonic_log':
        xs, fit = iso_fit(np.log(p[tr]), np.log(yc[tr])); g = lambda v: np.interp(v, xs, fit)
        qq = np.exp(g(np.log(p))); s = C.wmed(yc[tr] / qq[tr], qq[tr] / yc[tr]); return np.exp(g(np.log(p[te]))) * s
res4 = {}
for kind in ('a_global', 'b_logaffine', 'c1_quintile_factor', 'c2_isotonic_log'):
    P = np.full(len(yc), np.nan)
    for f in range(5): P[fold == f] = recal(kind, f)
    d, ci = boot(P, CA, ok)
    res4[kind] = dict(strict_cd=round(cd(P), 4), delta_vs_6p1737=round(cd(P) - cd(CA), 4), ci95=[round(v, 4) for v in ci], per_fold_delta={f: round(cd(P, fold == f) - cd(CA, fold == f), 4) for f in range(5)})
out['4_recalibration_of_candidate'] = res4
# 5
res5 = {}
for nm, other in (('sub1_B', S1), ('sub1_with_nested_stretch_REF', REF)):
    P = np.full(len(yc), np.nan); W = {}
    for f in range(5):
        tr = (fold != f) & ok; best = None
        for w in np.round(np.arange(0, 0.501, 0.05), 2):
            q = np.exp((1 - w) * np.log(CA) + w * np.log(other)); e_ = cd(q, tr)
            if best is None or e_ < best[0]: best = (e_, w)
        W[f] = best[1]; P[fold == f] = np.exp((1 - best[1]) * np.log(CA[fold == f]) + best[1] * np.log(other[fold == f]))
    d, ci = boot(P, CA, ok)
    res5[nm] = dict(weights_per_fold=W, strict_cd=round(cd(P), 4), delta_vs_6p1737=round(cd(P) - cd(CA), 4), ci95=[round(v, 4) for v in ci],
                    per_fold_delta={f: round(cd(P, fold == f) - cd(CA, fold == f), 4) for f in range(5)})
out['5_blend_sub1_into_candidate'] = res5
print(json.dumps(out, indent=1)); json.dump(out, open(LANE / 'reads/cd_sub1_calib_20261113.json', 'w'), indent=1)
