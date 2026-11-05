"""Green day 3 / E8 (pre-registered 2026-11-03 17:40 UTC): localisation-aware stride-2 detector (smaller Gaussian target).
E4: HEX needs misses <= 8 % and localisation <= 0.17 x spacing (about 4 px); the current detectors sit at 0.20-0.25 x
spacing with a target sigma of 8 px (spacing ~24 px).  Variant: the same stride-2 specialist recipe (fold-clean init,
1000 updates, batch 2, four flips, 20 train-green images) with spatial.SIGMA = 5 or 3.5 px.
  python ... SIGMA train --fold F --tag sg2_sigmaX   /   python ... SIGMA infer --fold F --tag sg2_sigmaX
  python ... 0 evaluate --tags sg2_sigma5 sg2_sigma3p5      (CPU: val-5 miss / FP / jitter, hex6 LOFO alone and in the active ten)
Gates: (1) localisation: val-5 jitter <= 0.17 x spacing with miss not worse than +0.02 vs the sigma-8 member (0.221) at
threshold 0.35; (2) HEX: as D2c (>= 0.005 over the control ten = evidence, >= 0.02 = adoption, fold-4 negative, >= 4/5 folds).
"""
import argparse, json, sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, 'experiments/green_track_20261028/source'); sys.path.insert(0, 'experiments/green_day3_20261103/source'); sys.path.insert(0, 'experiments/eval_harness_20261031/source'); sys.path.insert(0, 'experiments/green_day2_20261101/source')
import green_point_spatial_20261023 as spatial
SIGMA = float(sys.argv[1]); sys.argv = [sys.argv[0]]+sys.argv[2:]
if SIGMA > 0:
    spatial.SIGMA = SIGMA
import green_sg2_specialist_20261028 as SG
ROOT = Path('experiments/green_day3_20261103'); OUT = ROOT/'e8_sigma'


def evaluate(a):
    import cpu_pool
    import green_day3_e4_degradation_20261103 as E4
    import cd_estimators_20261028 as P
    import green_t4_numerics_20261101 as M
    import green_day2_cd_features_20261101 as D
    from scipy.spatial import cKDTree, Delaunay
    V = M.V; OUT.mkdir(parents=True, exist_ok=True)
    binding = json.load(open('experiments/green_point_probe_20261023/data_binding.json')); train_ids = set(binding['train_ids']); val5 = [n for n in E4.NAMES if n not in train_ids]
    with np.load('datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True) as z:
        ids = z['ids'].astype(str)
    pos = {n: i for i, n in enumerate(ids)}
    def measure(name, det):
        gt = E4.PTS[name]; sp = E4.SPACING[name]
        if len(det) == 0: return 1.0, 0.0, np.nan
        hull = Delaunay(gt); dmin, _ = cKDTree(gt).query(det); d_ = det[(hull.find_simplex(det) >= 0) | (dmin <= 0.5*sp)]
        if len(d_) == 0: return 1.0, 0.0, np.nan
        d, j = cKDTree(d_).query(gt); used = set(); tp = []
        for dd, jj in sorted(zip(d, j)):
            if dd <= 0.5*sp and jj not in used: used.add(jj); tp.append(dd)
        return 1-len(tp)/len(gt), (len(d_)-len(tp))/len(gt), float(np.median(tp)/sp) if tp else np.nan
    y, tta4, baseA, fold, seen, keys, F, aa = P.load(); ok = ~seen; n = len(y)
    with np.load(P.E/'recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True) as h:
        o = np.argsort(h['indices']); B = h['B_hybrid_w0p5'][o]
    with np.load('experiments/green_day2_20261101/d2b_active_gate/active_family_features.npz') as z:
        hx = {k.replace('hex6_', ''): z[k] for k in z.files if k.startswith('hex6_')}
    tags = ['sg2_specialist']+a.tags
    res = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), operating=dict(), lofo=dict())
    feats = {}
    for tag in tags:
        d = SG.ROOT/tag
        if not (d/'fold4'/'fold4_sg2_peaks.npz').exists():
            print('missing', tag); continue
        pk, _ = D.load_peaks(d)
        vals = [measure(nm, pk[pos[nm]][pk[pos[nm]][:, 2] >= 0.35, :2].astype(np.float64)) for nm in val5 if pk[pos[nm]] is not None]
        miss, fp, jit = np.nanmean(np.array(vals, float), 0)
        f1 = []
        for f in range(5):
            tj = d/f'fold{f}'/'train.json'
            if tj.exists(): f1.append(json.loads(tj.read_text())['validation']['mean_f1'])
        res['operating'][tag] = dict(val5_miss=float(miss), val5_fp=float(fp), val5_jitter=float(jit), val5_f1_mean=float(np.mean(f1)) if f1 else None)
        print('%-14s val-5 miss %.3f fp %.3f jitter %.3f F1 %s' % (tag, miss, fp, jit, round(np.mean(f1), 3) if f1 else None), flush=True)
        if tag == 'sg2_specialist':
            feats[tag] = hx['sg2_s46']; continue
        items = [(tag, r) for r in range(n)]
        def task(it):
            t, r = it; p = pk[r]
            if p is None: return r, np.nan
            q = p[p[:, 2] >= 0.35, :2].astype(np.float64); return r, V.voronoi_feats(q)['hex6_all'] if len(q) >= 12 else np.nan
        hf = np.full(n, np.nan)
        for r, v in map(task, items):
            hf[r] = v
        feats[tag] = hf
    def blend(fams, fe):
        A = np.log(np.stack([fe[k] for k in fams]))
        with np.errstate(all='ignore'):
            f = np.exp(np.nanmean(A, 0))
        r = P.lofo_blend(y, B, fold, ok, f, t=2); o_ = r['overall']
        return dict(delta_mean_pp=o_['delta_cd_pp']/3, folds_improving=o_['folds_improving'], per_fold_mean_pp=[r[str(h)]['delta_cd_pp']/3 for h in range(5)], w=[r[str(h)]['w'] for h in range(5)])
    fe = dict(hx); fe.update({t: v for t, v in feats.items() if t != 'sg2_specialist'})
    res['lofo']['control_ten'] = blend(M.ACTIVE_ORDER, fe); res['lofo']['single_sg2_s46'] = blend(['sg2_s46'], fe)
    for tag in a.tags:
        if tag not in feats: continue
        res['lofo'][f'single_{tag}'] = blend([tag], fe); res['lofo'][f'swap_sg2_s46_for_{tag}'] = blend([tag if k == 'sg2_s46' else k for k in M.ACTIVE_ORDER], fe); res['lofo'][f'ten_plus_{tag}'] = blend(M.ACTIVE_ORDER+[tag], fe)
    c = res['lofo']['control_ten']['delta_mean_pp']
    for k, v in res['lofo'].items():
        v['gain_vs_control'] = v['delta_mean_pp']-c; v['evidence'] = bool(v['delta_mean_pp'] <= c-0.005 and v['per_fold_mean_pp'][4] < 0 and v['folds_improving'] >= 4); v['adoption'] = bool(v['delta_mean_pp'] <= c-0.02 and v['per_fold_mean_pp'][4] < 0 and v['folds_improving'] >= 4)
        print('%-28s %.4f mean pp impr %d per fold %s w %s %s' % (k, v['delta_mean_pp'], v['folds_improving'], [round(x, 4) for x in v['per_fold_mean_pp']], v['w'], 'EVIDENCE' if v['evidence'] else ''), flush=True)
    res['localisation_gate'] = {t: dict(passed=bool(res['operating'][t]['val5_jitter'] <= 0.17 and res['operating'][t]['val5_miss'] <= res['operating']['sg2_specialist']['val5_miss']+0.02)) for t in a.tags if t in res['operating']}
    (OUT/'result.json').write_text(json.dumps(res, indent=1, default=float)); print('localisation gate', res['localisation_gate'], flush=True); print('done', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['train', 'infer', 'evaluate']); ap.add_argument('--fold', type=int, default=0); ap.add_argument('--tag', default='sg2_sigma5'); ap.add_argument('--seed', type=int, default=46); ap.add_argument('--flips', type=int, default=1); ap.add_argument('--tags', nargs='*', default=['sg2_sigma5', 'sg2_sigma3p5'])
    a = ap.parse_args()
    if a.command == 'evaluate':
        evaluate(a)
    else:
        print(json.dumps(dict(msg='sigma override', sigma=spatial.SIGMA, tag=a.tag, fold=a.fold)), flush=True)
        {'train': SG.cmd_train, 'infer': SG.cmd_infer}[a.command](a)
