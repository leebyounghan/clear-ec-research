"""Green day 3 / E7 (pre-registered 2026-11-03 17:05 UTC): the hex6 operating point.
E4: HEX tolerates false positives (~14 % break-even) far better than misses (~8 %); the stored peaks go down to 0.2,
so a lower threshold trades misses for FPs for free.  Choice rule (no tuning on the 8954): for each threshold in
{0.20, 0.25, 0.30, 0.35, 0.50}, measure (miss, FP, jitter) of the active stride-2 families and the pre-only s46 on the
validation-5 expert sets (hull region, 0.5-spacing match) and read the E4-simulated oracle HEX MAPE at that (miss, FP)
on the miss x FP grid (bilinear interpolation, jitter ignored since it barely moves with threshold); pick the threshold
with the lowest simulated HEX MAPE; then ONE LOFO run of the active ten (hex6 recomputed at that threshold for the
stride-2 and pre-only members; stride-8 maps keep their 0.35 peak rule) on B.  All thresholds' LOFO results are also
listed for the record, but the pre-registered pick is the simulation-chosen one.
"""
import json, sys, time, glob
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree, Delaunay
sys.path.insert(0, 'experiments/green_day3_20261103/source'); sys.path.insert(0, 'experiments/green_track_20261028/source'); sys.path.insert(0, 'experiments/eval_harness_20261031/source'); sys.path.insert(0, 'experiments/green_day2_20261101/source')
import cpu_pool
import green_day3_e4_degradation_20261103 as E4
import cd_estimators_20261028 as P
import green_t4_numerics_20261101 as M
import green_day2_cd_features_20261101 as D
V = M.V
ROOT = Path('experiments/green_day3_20261103'); OUT = ROOT/'e7_threshold'; OUT.mkdir(parents=True, exist_ok=True)
THS = [0.20, 0.25, 0.30, 0.35, 0.50]
SG2 = {'sg2_s46': 'sg2_specialist', 'sg2_s47': 'sg2_specialist_s47', 'sg2_s48': 'sg2_specialist_s48', 'sg2_s49': 'sg2_specialist_s49', 'sg2_s50': 'sg2_specialist_s50', 'sg2pre_s46': 'external_srep/preonly'}
GT = Path('experiments/green_track_20261028')
_C = {}


def measure(name, det):
    gt = E4.PTS[name]; sp = E4.SPACING[name]
    if len(det) == 0:
        return 1.0, 0.0, np.nan
    hull = Delaunay(gt); dmin, _ = cKDTree(gt).query(det); d_ = det[(hull.find_simplex(det) >= 0) | (dmin <= 0.5*sp)]
    if len(d_) == 0:
        return 1.0, 0.0, np.nan
    d, j = cKDTree(d_).query(gt); used = set(); tp = []
    for dd, jj in sorted(zip(d, j)):
        if dd <= 0.5*sp and jj not in used: used.add(jj); tp.append(dd)
    return 1-len(tp)/len(gt), (len(d_)-len(tp))/len(gt), float(np.median(tp)/sp) if tp else np.nan


def sim_hex(table, miss, fp):
    """Bilinear interpolation of the E4 oracle HEX MAPE on the miss x FP grid {0,.1,.2,.3}^2 (clipped)."""
    g = [0, .1, .2, .3]; m = float(np.clip(miss, 0, .3)); f = float(np.clip(fp, 0, .3))
    def val(mm, ff): return table[f'grid|({mm}, {ff})']['oracle_HEX']['mape']
    i = min(max(np.searchsorted(g, m)-1, 0), 2); j = min(max(np.searchsorted(g, f)-1, 0), 2)
    m0, m1, f0, f1 = g[i], g[i+1], g[j], g[j+1]; wm = (m-m0)/(m1-m0); wf = (f-f0)/(f1-f0)
    return (1-wm)*(1-wf)*val(m0, f0)+wm*(1-wf)*val(m1, f0)+(1-wm)*wf*val(m0, f1)+wm*wf*val(m1, f1)


def hex_task(item):
    fam, r, th = item; p = _C['pk'][fam][r]
    if p is None:
        return fam, th, r, np.nan
    q = p[p[:, 2] >= th, :2].astype(np.float64)
    return fam, th, r, V.voronoi_feats(q)['hex6_all'] if len(q) >= 12 else np.nan


def main():
    sim = json.loads((ROOT/'e4_degradation/simulate.json').read_text())['table']
    binding = json.load(open('experiments/green_point_probe_20261023/data_binding.json')); train_ids = set(binding['train_ids']); val5 = [n for n in E4.NAMES if n not in train_ids]
    with np.load('datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True) as z:
        ids = z['ids'].astype(str)
    pos = {n: i for i, n in enumerate(ids)}
    pk = {fam: D.load_peaks(GT/d)[0] for fam, d in SG2.items()}
    # step 1: operating point per threshold on the validation-5 (and all-25 for the record)
    op = {}
    for th in THS:
        rec = {}
        for fam in SG2:
            vals = []
            for n in val5:
                p = pk[fam][pos[n]]
                if p is None: continue
                vals.append(measure(n, p[p[:, 2] >= th, :2].astype(np.float64)))
            miss, fp, jit = np.nanmean(np.array(vals, float), 0)
            rec[fam] = dict(miss=float(miss), fp=float(fp), jitter=float(jit), sim_hex=float(sim_hex(sim, miss, fp)))
        mean_sim = float(np.mean([v['sim_hex'] for v in rec.values()]))
        op[str(th)] = dict(per_family=rec, mean_sim_hex=mean_sim, mean_miss=float(np.mean([v['miss'] for v in rec.values()])), mean_fp=float(np.mean([v['fp'] for v in rec.values()])))
        print('threshold %.2f: val-5 miss %.3f fp %.3f -> simulated HEX MAPE %.2f' % (th, op[str(th)]['mean_miss'], op[str(th)]['mean_fp'], mean_sim), flush=True)
    chosen = min(THS, key=lambda t: op[str(t)]['mean_sim_hex']); print('pre-registered choice:', chosen, flush=True)
    # step 2: hex6 at every threshold for the six stride-2 / pre-only families (pool), stride-8 members fixed
    y, tta4, baseA, fold, seen, keys, F, aa = P.load(); ok = ~seen; n = len(y)
    with np.load(P.E/'recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True) as h:
        o = np.argsort(h['indices']); B = h['B_hybrid_w0p5'][o]
    with np.load('experiments/green_day2_20261101/d2b_active_gate/active_family_features.npz') as z:
        hx = {k.replace('hex6_', ''): z[k] for k in z.files if k.startswith('hex6_')}
    _C['pk'] = pk
    items = [(fam, r, th) for fam in SG2 for th in THS if th != 0.35 for r in range(n)]
    t0 = time.time(); w = cpu_pool.n_workers(); print('hex6 items', len(items), 'workers', w, flush=True)
    feats = {th: {fam: (hx[fam].copy() if th == 0.35 else np.full(n, np.nan)) for fam in SG2} for th in THS}
    for fam, th, r, v in cpu_pool.pmap(hex_task, items, chunksize=64, workers=w, ordered=False):
        feats[th][fam][r] = v
    print('hex6 recomputed', round(time.time()-t0), 's', flush=True)
    def blend(fe):
        A = np.log(np.stack([fe[k] for k in M.ACTIVE_ORDER]))
        with np.errstate(all='ignore'):
            f = np.exp(np.nanmean(A, 0))
        r = P.lofo_blend(y, B, fold, ok, f, t=2); o_ = r['overall']
        return dict(delta_mean_pp=o_['delta_cd_pp']/3, delta_hex_pp=o_['delta_cd_pp'], folds_improving=o_['folds_improving'], per_fold_mean_pp=[r[str(h)]['delta_cd_pp']/3 for h in range(5)], w=[r[str(h)]['w'] for h in range(5)])
    res = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), operating_points=op, chosen_threshold=chosen, lofo={})
    for th in THS:
        fe = dict(hx); fe.update(feats[th]); res['lofo'][str(th)] = blend(fe); v = res['lofo'][str(th)]
        print('threshold %.2f: active ten LOFO on B %.4f mean pp (HEX %.4f) impr %d per fold %s w %s%s' % (th, v['delta_mean_pp'], v['delta_hex_pp'], v['folds_improving'], [round(x, 4) for x in v['per_fold_mean_pp']], v['w'], '   <- pre-registered pick' if th == chosen else ''), flush=True)
    c = res['lofo']['0.35']['delta_mean_pp']; v = res['lofo'][str(chosen)]
    res['gate'] = dict(gain_vs_035=v['delta_mean_pp']-c, evidence=bool(v['delta_mean_pp'] <= c-0.005 and v['folds_improving'] >= 4 and v['per_fold_mean_pp'][4] < 0), adoption=bool(v['delta_mean_pp'] <= c-0.02 and v['folds_improving'] >= 4 and v['per_fold_mean_pp'][4] < 0))
    (OUT/'result.json').write_text(json.dumps(res, indent=1, default=float)); print('GATE', res['gate'], flush=True); print('done', round(time.time()-t0), 's', flush=True)


if __name__ == '__main__':
    main()
