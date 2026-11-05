"""D2d (pre-registered 2026-11-01 21:10 UTC): deployed-definition hex6 (voronoi_features.voronoi_feats) computed on peaks
restricted to tissue (non-black mask, D.tissue_mask) for the full-frame members sg2pre_s46 (shipped) and aug_s46.
Compositions on B: control ten; ten with sg2pre_s46 replaced by its tissue-filtered version; ten + filtered aug_s46; singles.
Gate: >= 0.005 pp mean better than control, fold-4 negative, >= 4/5 folds (evidence-only under the parent adoption rule).
"""
import json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import multiprocessing as mp
import numpy as np
from pathlib import Path
sys.path.insert(0, 'experiments/eval_harness_20261031/source'); sys.path.insert(0, 'experiments/green_track_20261028/source'); sys.path.insert(0, 'experiments/green_day2_20261101/source')
import cd_estimators_20261028 as P
import green_t4_numerics_20261101 as M
import green_day2_cd_features_20261101 as D
V = M.V
OUT = Path('experiments/green_day2_20261101/d2d_tissue_filter'); OUT.mkdir(parents=True, exist_ok=True)
FAMS = {'sg2pre_s46': 'experiments/green_track_20261028/external_srep/preonly', 'aug_s46': 'experiments/green_day2_20261101/preonly_tta/aug_s46'}
_C = {}


def worker(r):
    if 'mm' not in _C:
        _C['mm'] = np.memmap(D.RAW, dtype=np.uint8, mode='r', shape=(D.N_TOTAL, D.HGT, D.WID))
    tis, dt = D.tissue_mask(np.asarray(_C['mm'][r, ::2, ::2]))
    out = {}
    for k, pk in _C['pk'].items():
        p = pk[r]
        if p is None:
            out[k] = (np.nan, np.nan); continue
        q = p[p[:, 2] >= 0.35, :2].astype(np.float64)
        xi = np.clip((q[:, 0]/2).astype(int), 0, tis.shape[1]-1); yi = np.clip((q[:, 1]/2).astype(int), 0, tis.shape[0]-1)
        qf = q[tis[yi, xi]]
        out[k] = (V.voronoi_feats(q)['hex6_all'], V.voronoi_feats(qf)['hex6_all'] if len(qf) >= 6 else np.nan)
    return r, out


def main():
    y, tta4, baseA, fold, seen, keys, F, aa = P.load(); ok = ~seen; n = len(y)
    with np.load(P.E/'recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True) as h:
        o = np.argsort(h['indices']); B = h['B_hybrid_w0p5'][o]
    with np.load('experiments/green_day2_20261101/d2b_active_gate/active_family_features.npz') as z:
        hx = {k: z[f'hex6_{k}'] for k in M.ACTIVE_ORDER}
    _C['pk'] = {k: D.load_peaks(Path(d))[0] for k, d in FAMS.items()}
    raw = {k: np.full(n, np.nan) for k in FAMS}; filt = {k: np.full(n, np.nan) for k in FAMS}
    t0 = time.time()
    with mp.get_context('fork').Pool(4) as pool:
        for i, (r, out) in enumerate(pool.imap_unordered(worker, range(n), chunksize=16)):
            for k, (a_, b_) in out.items():
                raw[k][r] = a_; filt[k][r] = b_
            if i % 2000 == 0:
                print('rows', i, round(time.time()-t0), flush=True)
    np.savez_compressed(OUT/'hex6_raw_vs_filtered.npz', **{f'raw_{k}': v for k, v in raw.items()}, **{f'filt_{k}': v for k, v in filt.items()})
    for k in FAMS:
        m = ok & np.isfinite(raw[k]) & np.isfinite(filt[k])
        print(k, 'raw vs deployed-cache hex6 agreement r=%.4f' % np.corrcoef(raw[k][m], hx['sg2pre_s46'][m])[0, 1] if k == 'sg2pre_s46' else '', 'raw vs filtered r=%.4f mean abs diff %.4f frac changed %.3f' % (np.corrcoef(raw[k][m], filt[k][m])[0, 1], np.abs(raw[k][m]-filt[k][m]).mean(), (np.abs(raw[k][m]-filt[k][m]) > 1e-9).mean()), flush=True)
    def blend(fams, feats):
        A = np.log(np.stack([feats[k] for k in fams]))
        with np.errstate(all='ignore'):
            f = np.exp(np.nanmean(A, 0))
        r = P.lofo_blend(y, B, fold, ok, f, t=2); o_ = r['overall']
        return dict(families=fams, delta_mean_pp=o_['delta_cd_pp']/3, delta_hex_pp=o_['delta_cd_pp'], folds_improving=o_['folds_improving'], per_fold_mean_pp=[r[str(h)]['delta_cd_pp']/3 for h in range(5)], w=[r[str(h)]['w'] for h in range(5)])
    feats = dict(hx); feats.update({f'filt_{k}': v for k, v in filt.items()}); feats.update({f'raw_{k}': v for k, v in raw.items()})
    comps = {'control_ten': M.ACTIVE_ORDER, 'ten_with_filtered_pre': [('filt_sg2pre_s46' if f == 'sg2pre_s46' else f) for f in M.ACTIVE_ORDER],
             'ten_plus_filtered_aug': M.ACTIVE_ORDER+['filt_aug_s46'], 'ten_filtered_pre_plus_filtered_aug': [('filt_sg2pre_s46' if f == 'sg2pre_s46' else f) for f in M.ACTIVE_ORDER]+['filt_aug_s46'],
             'single_raw_pre': ['raw_sg2pre_s46'], 'single_filt_pre': ['filt_sg2pre_s46'], 'single_raw_aug': ['raw_aug_s46'], 'single_filt_aug': ['filt_aug_s46']}
    res = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), compositions={})
    for name, fams in comps.items():
        r = blend(fams, feats); res['compositions'][name] = r
        print('%-36s mean %.4f pp (HEX %.4f) impr %d | per fold %s | w %s' % (name, r['delta_mean_pp'], r['delta_hex_pp'], r['folds_improving'], [round(v, 4) for v in r['per_fold_mean_pp']], r['w']), flush=True)
    c = res['compositions']['control_ten']['delta_mean_pp']
    for name, r in res['compositions'].items():
        r['gain_vs_control_mean_pp'] = r['delta_mean_pp']-c; r['gate_passed'] = bool(r['delta_mean_pp'] <= c-0.005 and r['per_fold_mean_pp'][4] < 0 and r['folds_improving'] >= 4)
    print('GATE passed:', [k for k, r in res['compositions'].items() if r['gate_passed']], flush=True)
    (OUT/'result.json').write_text(json.dumps(res, indent=1, default=float)); print('written', round(time.time()-t0))


if __name__ == '__main__':
    main()
