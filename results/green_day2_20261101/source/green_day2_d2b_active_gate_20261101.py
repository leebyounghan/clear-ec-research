"""D2b (pre-registered 2026-11-01 20:50 UTC): coverage-conditional blend weight for the ACTIVE ten-family hex6.
Per-family hex6 (deployed definition via green_t4_numerics_20261101.hex6_from) and per-family detection counts are
recomputed from the fold-clean bf16 OOF outputs.  Strata = terciles of the summed detection count over the ten
families (label-free, available at inference).  Control = plain LOFO (must reproduce about -0.0311 mean pp on B).
Gate: stratified LOFO beats the plain ten-family LOFO by >= 0.005 pp mean, fold-4 delta negative, >= 4/5 folds improving.
"""
import json, sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, 'experiments/eval_harness_20261031/source'); sys.path.insert(0, 'experiments/green_track_20261028/source'); sys.path.insert(0, 'experiments/green_day2_20261101/source')
import eval_oof_20261031 as H
import cd_estimators_20261028 as P
import green_t4_numerics_20261101 as M
import green_day2_cd_features_20261101 as D
V = M.V
OUT = Path('experiments/green_day2_20261101/d2b_active_gate'); OUT.mkdir(parents=True, exist_ok=True)
y, tta4, baseA, fold, seen, keys, F, aa = P.load(); ok = ~seen; n = len(y)
with np.load(P.E/'recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True) as h:
    order = np.argsort(h['indices']); baseB = h['B_hybrid_w0p5'][order]; idx_sorted = h['indices'][order]
assert np.array_equal(idx_sorted, np.arange(n))
base = H.load_base(); pos = np.argsort(base['idx']); st = H.image_stats(base); black = np.zeros(n); black[base['idx']] = st['black_frac']; framed = (black > 0.01).astype(int)

def hex6_and_count(kind, path):
    Hh = np.full(n, np.nan); C = np.full(n, np.nan)
    with np.load(path) as z:
        if kind == 'maps':
            for r, m in zip(z['indices'], z['probability']):
                pk = V.peaks_sg(m); C[r] = len(pk); Hh[r] = V.voronoi_feats(pk)['hex6_all']
        else:
            rows, pk, off = z['indices'], z['peaks'], z['offsets']
            for i, r in enumerate(rows):
                q = pk[off[i]:off[i+1]]; q = q[q[:, 2] >= 0.35, :2].astype(np.float64); C[r] = len(q); Hh[r] = V.voronoi_feats(q)['hex6_all']
    return Hh, C
t0 = time.time(); hx, cnt = {}, {}
for fam in M.ACTIVE_ORDER:
    kind, src = M.BF16_SOURCES[fam]; Hh = np.full(n, np.nan); C = np.full(n, np.nan)
    for f in range(5):
        a, b = hex6_and_count(kind, src(f)); m = np.isfinite(a); Hh[m] = a[m]; C[m] = b[m]
    hx[fam] = Hh; cnt[fam] = C; print('family', fam, 'rows', int(np.isfinite(Hh).sum()), 'median count', float(np.nanmedian(C)), round(time.time()-t0), flush=True)
np.savez_compressed(OUT/'active_family_features.npz', **{f'hex6_{k}': v for k, v in hx.items()}, **{f'count_{k}': v for k, v in cnt.items()}, seen=seen, fold=fold)
A = np.log(np.stack([hx[k] for k in M.ACTIVE_ORDER]))
with np.errstate(all='ignore'):
    feat = np.exp(np.nanmean(A, 0))
total = np.nansum(np.stack([cnt[k] for k in M.ACTIVE_ORDER]), 0)
pre_cnt = cnt['sg2pre_s46']; spread = np.nanstd(A, 0)
def terc(v, k=3):
    q = np.nanquantile(v[ok], np.linspace(0, 1, k+1)[1:-1]); return np.digitize(np.where(np.isfinite(v), v, np.nanmedian(v)), q)
D.STRATA_NAMES = {i: f's{i}' for i in range(10)}
res = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), control={}, gated={})
r0 = P.lofo_blend(y, baseB, fold, ok, feat, t=2)
res['control'] = dict(delta_mean_pp=r0['overall']['delta_mean_pp'], delta_hex_pp=r0['overall']['delta_cd_pp'], folds_improving=r0['overall']['folds_improving'], w=[r0[str(h)]['w'] for h in range(5)], per_fold_mean_pp=[r0[str(h)]['delta_cd_pp']/3 for h in range(5)])
print('CONTROL plain ten-family LOFO on B: %.4f mean pp (HEX %.4f), impr %d, w %s' % (r0['overall']['delta_mean_pp'], r0['overall']['delta_cd_pp'], r0['overall']['folds_improving'], res['control']['w']), flush=True)
covs = {'total_count_t3': terc(total), 'total_count_q5': terc(total, 5), 'pre_count_t3': terc(pre_cnt), 'spread_t3': terc(spread), 'framed': framed, 'framed_x_total_t3': framed*3+terc(total)}
for cn, cv in covs.items():
    s, pred = D.lofo_strat(y, baseB, fold, ok, feat, cv, t=2); o = s['overall']
    res['gated'][cn] = dict(delta_mean_pp=o['delta_mean_pp'], delta_hex_pp=o['delta_pp'], folds_improving=o['folds_improving'], dev_hex_pp=o['dev_folds0to3_pp'], fold4_hex_pp=o['fold4_pp'], per_stratum_hex_pp=o['per_stratum_pp'], w_per_fold=[s[str(h)]['w'] for h in range(5)], per_fold_mean_pp=[s[str(h)]['delta_pp']/3 for h in range(5)],
                           gain_vs_control_mean_pp=o['delta_mean_pp']-r0['overall']['delta_mean_pp'], strata_B_hex_mape={int(k): float(P.mape(baseB[ok & (cv == k) & (y[:, 2] > 0), 2], y[ok & (cv == k) & (y[:, 2] > 0), 2])) for k in np.unique(cv[ok])})
    print('%-20s mean %.4f pp (impr %d) gain vs control %.4f | dev HEX %.4f f4 %.4f | per-stratum HEX %s | w f0 %s' % (cn, o['delta_mean_pp'], o['folds_improving'], o['delta_mean_pp']-r0['overall']['delta_mean_pp'], o['dev_folds0to3_pp'], o['fold4_pp'], {k: round(v, 3) for k, v in o['per_stratum_pp'].items()}, s['0']['w']), flush=True)
(OUT/'result.json').write_text(json.dumps(res, indent=1, default=float)); print('done', round(time.time()-t0))
