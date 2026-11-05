"""D2c (pre-registered 2026-11-01 21:00 UTC): new full-frame detector families as members of the ACTIVE hex6 composition.
For each extra family (TTA peaks dir), hex6 is computed with the deployed definition (voronoi_features.voronoi_feats,
peaks >= 0.35) and the LOFO blend on B is run for: control ten; swap sg2pre_s46 -> extra; ten + extra; ten + all extras.
Gate: >= 0.005 pp mean better than the control ten (about -0.0311), fold-4 delta negative, >= 4/5 folds improving.
    python ... --extra aug_s46=experiments/green_day2_20261101/preonly_tta/aug_s46 [name=dir ...] --tag T
"""
import argparse, json, sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, 'experiments/eval_harness_20261031/source'); sys.path.insert(0, 'experiments/green_track_20261028/source'); sys.path.insert(0, 'experiments/green_day2_20261101/source')
import cd_estimators_20261028 as P
import green_t4_numerics_20261101 as M
V = M.V
OUT = Path('experiments/green_day2_20261101/d2c_members'); OUT.mkdir(parents=True, exist_ok=True)
ACTIVE = M.ACTIVE_ORDER


def hex6_from_peaks_dir(d, n):
    Hh = np.full(n, np.nan)
    for f in range(5):
        with np.load(Path(d)/f'fold{f}'/f'fold{f}_sg2_peaks.npz') as z:
            rows, pk, off = z['indices'], z['peaks'], z['offsets']
            for i, r in enumerate(rows):
                q = pk[off[i]:off[i+1]]; q = q[q[:, 2] >= 0.35, :2].astype(np.float64); Hh[r] = V.voronoi_feats(q)['hex6_all']
    return Hh


def main(a):
    y, tta4, baseA, fold, seen, keys, F, aa = P.load(); ok = ~seen; n = len(y)
    with np.load(P.E/'recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True) as h:
        o = np.argsort(h['indices']); B = h['B_hybrid_w0p5'][o]
    with np.load('experiments/green_day2_20261101/d2b_active_gate/active_family_features.npz') as z:
        hx = {k: z[f'hex6_{k}'] for k in ACTIVE}
    cache = OUT/f'extra_hex6_{a.tag}.npz'
    extras = dict(kv.split('=', 1) for kv in a.extra)
    ex = {}
    if cache.exists():
        with np.load(cache) as z:
            ex = {k: z[k] for k in z.files if k in extras}
    t0 = time.time()
    todo = [(k, d) for k, d in extras.items() if k not in ex]
    if todo:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(min(4, len(todo))) as pool:
            for k, Hh in zip([k for k, _ in todo], pool.starmap(hex6_from_peaks_dir, [(d, n) for _, d in todo])):
                ex[k] = Hh; print('hex6 computed', k, int(np.isfinite(Hh).sum()), round(time.time()-t0), flush=True)
    np.savez_compressed(cache, **ex)
    def blend(fams, feats):
        A = np.log(np.stack([feats[k] for k in fams]))
        with np.errstate(all='ignore'):
            f = np.exp(np.nanmean(A, 0))
        r = P.lofo_blend(y, B, fold, ok, f, t=2); o_ = r['overall']
        return dict(families=fams, delta_mean_pp=o_['delta_cd_pp']/3, delta_hex_pp=o_['delta_cd_pp'], folds_improving=o_['folds_improving'], per_fold_mean_pp=[r[str(h)]['delta_cd_pp']/3 for h in range(5)], w=[r[str(h)]['w'] for h in range(5)], est_hex_mape=o_['est_cd_mape'], err_corr=o_['err_corr_mean'])
    feats = dict(hx); feats.update(ex)
    res = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), compositions={})
    comps = {'control_ten': ACTIVE}
    for k in ex:
        comps[f'swap_pre_for_{k}'] = [k if f == 'sg2pre_s46' else f for f in ACTIVE]
        comps[f'ten_plus_{k}'] = ACTIVE+[k]
        comps[f'single_{k}'] = [k]
    if len(ex) > 1:
        comps['ten_plus_all_extras'] = ACTIVE+list(ex)
        comps['swap_pre_for_all_extras'] = [f for f in ACTIVE if f != 'sg2pre_s46']+list(ex)
    comps['single_sg2pre_s46'] = ['sg2pre_s46']
    for name, fams in comps.items():
        r = blend(fams, feats); res['compositions'][name] = r
        print('%-28s mean %.4f pp (HEX %.4f) impr %d | per fold %s | w %s' % (name, r['delta_mean_pp'], r['delta_hex_pp'], r['folds_improving'], [round(v, 4) for v in r['per_fold_mean_pp']], r['w']), flush=True)
    c = res['compositions']['control_ten']['delta_mean_pp']
    for name, r in res['compositions'].items():
        r['gain_vs_control_mean_pp'] = r['delta_mean_pp']-c
        r['gate_passed'] = bool(r['delta_mean_pp'] <= c-0.005 and r['per_fold_mean_pp'][4] < 0 and r['folds_improving'] >= 4)
    passed = [k for k, r in res['compositions'].items() if r['gate_passed']]
    print('GATE passed:', passed, flush=True)
    (OUT/f'result_{a.tag}.json').write_text(json.dumps(res, indent=1, default=float)); print('written')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--extra', nargs='+', required=True); ap.add_argument('--tag', default='aug'); main(ap.parse_args())
