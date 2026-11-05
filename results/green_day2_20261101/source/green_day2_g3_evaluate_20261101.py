"""G3 evaluation (gates pre-registered in protocol.json, 21:10 UTC).
  G3a representation: aux vs control, single-seed s46 TTA4 x LOFO scalars, HEX MAPE on folds 0-3 pooled and fold 4.
  G3b submission: B with its s46 member replaced by the arm (A = mean(arm_s46, s47) x re-fit LOFO scalars; B = 0.5 A + 0.5 P1),
                  plus the re-fit active-hex6 blend, vs B + active hex6.
  G3c distilled hex6: the aux head's hex6 prediction LOFO-blended into B's HEX.
    python experiments/green_day2_20261101/source/green_day2_g3_evaluate_20261101.py [--partial]
"""
import argparse, json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, 'experiments/eval_harness_20261031/source'); sys.path.insert(0, 'experiments/green_track_20261028/source')
import eval_oof_20261031 as H
import cd_estimators_20261028 as P

E = Path('experiments'); OUT = E/'green_day2_20261101/g3_hex6aux'
RECIPE = {46: {0: 'phase2_pooling_replication_20261026/f0_s46_std', 1: 'oof_recipeA_20261027/fold1/regression_std_s46', 2: 'oof_recipeA_20261027/fold2/regression_std_s46_v2', 3: 'oof_recipeA_20261027/fold3/regression_std_s46', 4: 'phase2_pooling_20261020/arms/std'},
          47: {0: 'phase2_pooling_replication_20261026/f0_s47_std', 4: 'phase2_pooling_replication_20261026/f4_s47_std', 1: 'oof_recipeA_20261027/fold1/regression_std_s47', 2: 'oof_recipeA_20261027/fold2/regression_std_s47', 3: 'oof_recipeA_20261027/fold3/regression_std_s47'}}
ACTIVE = ['s8_s46', 's8_s47', 's8_s48', 'sg2_s46', 'sg2_s47', 'sg2_s48', 'sg2_s49', 'sg2_s50', 's8multi_s47', 'sg2pre_s46']
K = ('CD', 'CV', 'HEX')


def weighted_median(v, w):
    o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w); return float(v[np.searchsorted(c, 0.5*c[-1])])


def fit_scalars(p, y):
    return np.array([weighted_median(y[y[:, j] > 0, j]/p[y[:, j] > 0, j], p[y[:, j] > 0, j]/y[y[:, j] > 0, j]) for j in range(3)])


def lofo_scale(raw, y, fold, folds):
    out = np.full_like(raw, np.nan); sc = {}
    for h in folds:
        fit = np.isin(fold, [f for f in folds if f != h]) & np.isfinite(raw).all(1)
        c = fit_scalars(raw[fit], y[fit]); sc[int(h)] = c.tolist(); out[fold == h] = raw[fold == h]*c
    return out, sc


def score(pred, y, m):
    e = H.ape(pred[m], y[m]); return H.official(e)


def main(a):
    y, tta4, baseA, fold, seen, keys, F, aa = P.load(); n = len(y); ok = ~seen
    with np.load(P.E/'recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True) as h:
        o = np.argsort(h['indices']); assert np.array_equal(h['indices'][o], np.arange(n))
        B = h['B_hybrid_w0p5'][o]; P1 = h['historical_five_calibrated'][o]; A_ref = h['A_recipeA_tta_scalar'][o]
    with np.load(E/'green_day2_20261101/d2b_active_gate/active_family_features.npz') as z:
        with np.errstate(all='ignore'):
            feat10 = np.exp(np.nanmean(np.log(np.stack([z[f'hex6_{k}'] for k in ACTIVE])), 0))

    def member(root_map, f):
        with np.load(E/root_map[f]/'runs'/f'fold{f}'/'cpdino_vitb'/'predictions_tta4.npz') as z:
            return z['indices'], z['pred_tta4']
    def arm_member(arm, f):
        p = OUT/arm/f'fold{f}'/'predictions_tta4.npz'
        if not p.exists():
            return None
        with np.load(p) as z:
            return z['indices'], z['pred_tta4'], z['hex6_pred_tta4']
    AUX = a.arm
    avail = {arm: [f for f in range(5) if (OUT/arm/f'fold{f}'/'predictions_tta4.npz').exists()] for arm in (AUX, 'control')}
    folds = sorted(set(avail[AUX]) & set(avail['control']))
    print('folds available for both arms:', folds, flush=True)
    if not folds:
        return
    if len(folds) < 5 and not a.partial:
        print('incomplete; rerun with --partial for per-fold G3a only'); return
    s46 = np.full((n, 3), np.nan); s47 = np.full((n, 3), np.nan)
    for f in range(5):
        i, p = member(RECIPE[46], f); s46[i] = p; i, p = member(RECIPE[47], f); s47[i] = p
    arms = {}
    for arm in (AUX, 'control'):
        pa = np.full((n, 3), np.nan); hx = np.full(n, np.nan)
        for f in folds:
            i, p, hh = arm_member(arm, f); pa[i] = p; hx[i] = hh
        arms[arm] = (pa, hx)
    m_all = ok & np.isin(fold, folds)
    res = dict(folds=folds, n_rows=int(m_all.sum()))
    # sanity: reproduce A and B from the two original members
    A_rec, sc_rec = lofo_scale(0.5*(s46+s47), y, fold, list(range(5)))
    res['sanity'] = dict(max_rel_diff_A=float(np.nanmax(np.abs(A_rec-A_ref)/A_ref)), max_rel_diff_B=float(np.nanmax(np.abs(0.5*A_rec+0.5*P1-B)/B)))
    print('sanity: reconstructed A vs stored A max rel diff %.2e, B %.2e' % (res['sanity']['max_rel_diff_A'], res['sanity']['max_rel_diff_B']), flush=True)
    # G3a: single-seed s46 calibrated, per fold
    g3a = {}
    for name, raw in [('baseline_s46', s46), ('aux', arms[AUX][0]), ('control', arms['control'][0])]:
        cal, _ = lofo_scale(raw, y, fold, folds)
        g3a[name] = dict(per_fold={int(f): score(cal, y, ok & (fold == f)) for f in folds},
                         dev=score(cal, y, ok & np.isin(fold, [f for f in folds if f != 4])) if any(f != 4 for f in folds) else None,
                         fold4=score(cal, y, ok & (fold == 4)) if 4 in folds else None, raw_mape=score(raw, y, m_all))
    res['G3a'] = g3a
    for name in g3a:
        print('G3a %-12s per-fold HEX %s | dev %s | fold4 %s | mean %s' % (name, {f: round(v['HEX'], 3) for f, v in g3a[name]['per_fold'].items()}, round(g3a[name]['dev']['HEX'], 4) if g3a[name]['dev'] else None, round(g3a[name]['fold4']['HEX'], 4) if g3a[name]['fold4'] else None, {f: round(v['mean'], 3) for f, v in g3a[name]['per_fold'].items()}), flush=True)
    if g3a['aux']['dev'] and g3a['control']['dev']:
        d = g3a['aux']['dev']['HEX']-g3a['control']['dev']['HEX']; d4 = (g3a['aux']['fold4']['HEX']-g3a['control']['fold4']['HEX']) if g3a['aux']['fold4'] else None
        res['G3a_gate'] = dict(hex_dev_aux_minus_control=d, hex_fold4_aux_minus_control=d4, passed=bool(d <= -0.10 and (d4 is None or d4 < 0)))
        print('G3a gate: HEX dev aux-control %.4f pp, fold4 %s -> %s' % (d, None if d4 is None else round(d4, 4), res['G3a_gate']['passed']), flush=True)
    if len(folds) == 5:
        # G3b: replace the s46 member in B
        r0 = P.lofo_blend(y, B, fold, ok, feat10, t=2); base_plus = score(B, y, ok)['mean']+r0['overall']['delta_cd_pp']/3
        res['G3b'] = dict(B=score(B, y, ok), B_plus_hex6_mean=base_plus, hex6_on_B_mean_pp=r0['overall']['delta_cd_pp']/3)
        for arm in (AUX, 'control'):
            A_new, sc = lofo_scale(0.5*(arms[arm][0]+s47), y, fold, folds); B_new = 0.5*A_new+0.5*P1
            r = P.lofo_blend(y, B_new, fold, ok, feat10, t=2)
            per_fold = {f: score(B_new, y, ok & (fold == f))['mean']+r[str(f)]['delta_cd_pp']/3-(score(B, y, ok & (fold == f))['mean']+r0[str(f)]['delta_cd_pp']/3) for f in range(5)}
            comb = score(B_new, y, ok)['mean']+r['overall']['delta_cd_pp']/3
            res['G3b'][arm] = dict(B_new=score(B_new, y, ok), A_new=score(A_new, y, ok), hex6_on_B_new_mean_pp=r['overall']['delta_cd_pp']/3, hex6_w=[r[str(f)]['w'] for f in range(5)], combined_mean=comb, delta_vs_B_plus_hex6=comb-base_plus, per_fold_delta=per_fold, folds_improving=int(sum(v < 0 for v in per_fold.values())), scalars=sc)
            print('G3b %-8s B_new %.4f (B %.4f) | hex6 on B_new %.4f | combined %.4f vs B+hex6 %.4f -> delta %.4f, folds improving %d, per fold %s' % (arm, res['G3b'][arm]['B_new']['mean'], res['G3b']['B']['mean'], r['overall']['delta_cd_pp']/3, comb, base_plus, comb-base_plus, res['G3b'][arm]['folds_improving'], {f: round(v, 4) for f, v in per_fold.items()}), flush=True)
        res['G3b_gate'] = dict(passed=bool(res['G3b'][AUX]['delta_vs_B_plus_hex6'] <= -0.03 and res['G3b'][AUX]['folds_improving'] >= 4))
        # G3c: distilled hex6
        hx = arms[AUX][1]; mm = ok & np.isfinite(hx) & np.isfinite(feat10)
        rc = P.lofo_blend(y, B, fold, ok, np.where(np.isfinite(hx) & (hx > 0), hx, np.nan), t=2)
        res['G3c'] = dict(delta_mean_pp=rc['overall']['delta_cd_pp']/3, delta_hex_pp=rc['overall']['delta_cd_pp'], folds_improving=rc['overall']['folds_improving'], w=[rc[str(f)]['w'] for f in range(5)], r_log_distilled_vs_active=float(np.corrcoef(np.log(hx[mm]), np.log(feat10[mm]))[0, 1]), passed=bool(rc['overall']['delta_cd_pp']/3 <= -0.02))
        print('G3c distilled hex6 on B: %.4f mean pp (HEX %.4f), impr %d, w %s, r(log distilled, log active) %.3f -> %s' % (rc['overall']['delta_cd_pp']/3, rc['overall']['delta_cd_pp'], rc['overall']['folds_improving'], res['G3c']['w'], res['G3c']['r_log_distilled_vs_active'], res['G3c']['passed']), flush=True)
    res['aux_arm'] = AUX
    suffix = ('' if AUX == 'aux' else '_'+AUX)+('_partial' if len(folds) < 5 else '')
    (OUT/f'result{suffix}.json').write_text(json.dumps(res, indent=1, default=float)); print('written', f'result{suffix}.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--partial', action='store_true'); ap.add_argument('--arm', default='aux'); main(ap.parse_args())
