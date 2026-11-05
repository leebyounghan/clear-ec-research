"""D1b (pre-registered 2026-11-01 20:40 UTC): quality-gated LOFO blending of full-frame detector CD / hex6 estimators into B.
Strata = terciles (or quintiles) of a label-free quality covariate computed from the detector outputs themselves:
  agree_nn  : std over the three pre-only seeds of log nn_cd (low = agreement)
  n_interior: mean interior cell count (high = good coverage)
  win_score : mean max-window detection score
  framed x agree_nn (6 strata)
Gate (as D1): overall <= -0.03 pp mean with >= 4/5 folds improving AND framed <= -0.05 pp mean, OR framed <= -0.08 with unframed not worse than +0.01; fold 4 negative.
"""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, 'experiments/eval_harness_20261031/source'); sys.path.insert(0, 'experiments/green_track_20261028/source'); sys.path.insert(0, 'experiments/green_day2_20261101/source')
import eval_oof_20261031 as H
import cd_estimators_20261028 as P
import green_day2_cd_features_20261101 as D

OUT = Path('experiments/green_day2_20261101/d1b_quality_gate'); OUT.mkdir(parents=True, exist_ok=True)
base = H.load_base(); y, B, fold, ids = base['y'], base['B'], base['fold'], base['ids']
z = np.load('experiments/green_day2_20261101/d1_cd_features/features.npz', allow_pickle=True); assert np.array_equal(z['idx'], base['idx'])
ok = ~z['seen']; fams = ['pre_s46', 'pre_s47', 'pre_s48']
st = H.image_stats(base); framed = (st['black_frac'] > 0.01).astype(int)
def logf(name):
    v = z[name].astype(float); return np.where(v > 0, np.log(np.where(v > 0, v, 1)), np.nan)
def mean3(suf, th=35):
    A = np.stack([logf(f'{f}_t{th}_{suf}') for f in fams])
    with np.errstate(all='ignore'):
        return np.exp(np.nanmean(A, 0)), np.nanstd(A, 0)
nn, agree_nn = mean3('nn_cd'); vor, agree_vor = mean3('vor_cd'); vorh, _ = mean3('vor_cd_hex')
n_int = np.nanmean(np.stack([z[f'{f}_t35_n_interior'].astype(float) for f in fams]), 0)
wscore = np.nanmean(np.stack([z[f'{f}_t35_win_quality_score'].astype(float) for f in fams]), 0)
hex_all, agree_hex = mean3('hex6_all')
def terciles(v, k=3):
    q = np.nanquantile(v[ok], np.linspace(0, 1, k+1)[1:-1]); return np.digitize(np.where(np.isfinite(v), v, np.nanmedian(v)), q)
covs = {'agree_nn_t3': terciles(agree_nn), 'agree_nn_q5': terciles(agree_nn, 5), 'n_interior_t3': terciles(n_int), 'win_score_t3': terciles(wscore), 'framed': framed,
        'framed_x_agree_nn_t3': framed*3+terciles(agree_nn), 'agree_vor_t3': terciles(agree_vor)}
D.STRATA_NAMES = {i: f's{i}' for i in range(10)}
res = dict(created_utc=D.time.strftime('%Y-%m-%dT%H:%M:%SZ', D.time.gmtime()), cd={}, hex={}, covariate_summary={})
for cn, cv in covs.items():
    res['covariate_summary'][cn] = {int(s): dict(n=int((ok & (cv == s)).sum()), framed_frac=float(framed[ok & (cv == s)].mean()), B_cd_mape=float(P.mape(B[ok & (cv == s) & (y[:, 0] > 0), 0], y[ok & (cv == s) & (y[:, 0] > 0), 0]))) for s in np.unique(cv[ok])}
print('B CD MAPE by agree_nn tercile (0 = best agreement):', {k: round(v['B_cd_mape'], 2) for k, v in res['covariate_summary']['agree_nn_t3'].items()}, flush=True)
for fname, f in [('mean3_nn_cd', nn), ('mean3_vor_cd', vor), ('mean3_vor_cd_hex', vorh)]:
    for cn, cv in covs.items():
        s, pred = D.lofo_strat(y, B, fold, ok, f, cv, t=0)
        o = s['overall']
        # per-stratum estimator MAPE (fold-clean calibrated) for the record
        res['cd'][f'{fname}|{cn}'] = dict(delta_mean_pp=o['delta_mean_pp'], folds_improving=o['folds_improving'], dev_pp=o['dev_folds0to3_pp'], fold4_pp=o['fold4_pp'], per_stratum_pp=o['per_stratum_pp'],
                                        w=[s[str(h)]['w'] for h in range(5)], framed_pp=float(P.mape(pred[ok & (framed == 1) & (y[:, 0] > 0)], y[ok & (framed == 1) & (y[:, 0] > 0), 0])-P.mape(B[ok & (framed == 1) & (y[:, 0] > 0), 0], y[ok & (framed == 1) & (y[:, 0] > 0), 0])),
                                        unframed_pp=float(P.mape(pred[ok & (framed == 0) & (y[:, 0] > 0)], y[ok & (framed == 0) & (y[:, 0] > 0), 0])-P.mape(B[ok & (framed == 0) & (y[:, 0] > 0), 0], y[ok & (framed == 0) & (y[:, 0] > 0), 0])))
        print('%-18s %-22s mean %.4f pp (impr %d) dev %.4f f4 %.4f | framed %.4f unframed %.4f CD pp | per-stratum %s | w f0 %s' % (fname, cn, o['delta_mean_pp'], o['folds_improving'], o['dev_folds0to3_pp'], o['fold4_pp'], res['cd'][f'{fname}|{cn}']['framed_pp'], res['cd'][f'{fname}|{cn}']['unframed_pp'], {k: round(v, 3) for k, v in o['per_stratum_pp'].items()}, s['0']['w']), flush=True)
# estimator MAPE by agreement tercile (calibrated LOFO) for interpretation
lf = np.log(nn); ly = np.log(y[:, 0]); est = np.full(len(nn), np.nan)
for h in range(5):
    fit = ok & (fold != h) & np.isfinite(lf) & (y[:, 0] > 0); c = np.polyfit(lf[fit], ly[fit], 1); held = ok & (fold == h) & np.isfinite(lf); est[held] = np.exp(np.polyval(c, lf[held]))
res['nn_cd_mape_by_agree_tercile'] = {int(s): dict(est=float(P.mape(est[ok & (covs['agree_nn_t3'] == s) & np.isfinite(est) & (y[:, 0] > 0)], y[ok & (covs['agree_nn_t3'] == s) & np.isfinite(est) & (y[:, 0] > 0), 0])), B=res['covariate_summary']['agree_nn_t3'][int(s)]['B_cd_mape']) for s in range(3)}
print('nn_cd vs B CD MAPE by agreement tercile:', res['nn_cd_mape_by_agree_tercile'], flush=True)
# HEX: quality-gated hex6 (three pre-only seeds) vs plain
for cn in ['agree_hex_t3', 'framed', 'n_interior_t3']:
    cv = terciles(agree_hex) if cn == 'agree_hex_t3' else covs[cn]
    s, pred = D.lofo_strat(y, B, fold, ok, hex_all, cv, t=2); o = s['overall']
    res['hex'][f'hex6_all_mean3|{cn}'] = dict(delta_mean_pp=o['delta_mean_pp'], folds_improving=o['folds_improving'], dev_pp=o['dev_folds0to3_pp'], fold4_pp=o['fold4_pp'], per_stratum_pp=o['per_stratum_pp'], w=[s[str(h)]['w'] for h in range(5)])
    print('HEX hex6_all_mean3 | %-14s mean %.4f pp (impr %d) dev %.4f f4 %.4f per-stratum %s w f0 %s' % (cn, o['delta_mean_pp'], o['folds_improving'], o['dev_folds0to3_pp'], o['fold4_pp'], {k: round(v, 3) for k, v in o['per_stratum_pp'].items()}, s['0']['w']), flush=True)
r0 = P.lofo_blend(y, B, fold, ok, hex_all, t=2); print('HEX hex6_all_mean3 plain (control): %.4f mean pp' % r0['overall']['delta_mean_pp'])
(OUT/'result.json').write_text(json.dumps(res, indent=1, default=float)); print('done')
