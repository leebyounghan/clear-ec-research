"""Green day 3 / E5 (pre-registered 2026-11-03 16:25 UTC): split-aware robust area estimator for CV.
  tune  : on the E4 degradation grid (realistic FP 0-40 %, misses 0-40 %, jitter, miss x FP grid; 10 repeats) choose the
          estimator (merge radius, sliver cut, complete-cell filter, dispersion statistic) by the mean LOO-scaled CV MAPE
          over the FP axis + the miss x FP grid + zero degradation, subject to zero-degradation MAPE <= 10; report the new
          break-even per axis for the chosen estimator.  Parameters are frozen here, never on GT of the 8954.
  apply : the frozen estimator on real fold-clean points (active ten families, pre-only s46, TTA s49, CPSAM centroids)
          and on CPSAM mask areas; per-image CV in the cluster window (full-frame sources) or over all cells (compact);
          LOFO on B for CV; gates evidence <= -0.005 mean pp, adoption <= -0.02 with >= 4/5 folds and fold-4 negative.
CPU only; cpu_pool (free-quota workers, fork, chunked imap); Voronoi computed once per point set and shared by all variants.
"""
import argparse, glob, itertools, json, sys, time
import numpy as np
from pathlib import Path
from scipy.spatial import Voronoi, Delaunay, cKDTree
sys.path.insert(0, 'experiments/green_day3_20261103/source'); sys.path.insert(0, 'experiments/green_track_20261028/source'); sys.path.insert(0, 'experiments/eval_harness_20261031/source'); sys.path.insert(0, 'experiments/green_day2_20261101/source')
import cpu_pool
import green_day3_e4_degradation_20261103 as E4
ROOT = Path('experiments/green_day3_20261103'); OUT = ROOT/'e5_robust_cv'; OUT.mkdir(parents=True, exist_ok=True)
MERGE = [None, 0.4, 0.5, 0.6]; SLIVER = [None, 0.3, 0.4, 0.5]; COMPLETE = [False, True]; DISP = ['cv', 'trim5', 'trim10', 'mad', 'mix']
TUNE_LEVELS = dict(miss=E4.LEVELS['miss'], fp=E4.LEVELS['fp'], jitter=E4.LEVELS['jitter']); TUNE_REPEATS = 10


def poly_area(v):
    x, y = v[:, 0], v[:, 1]; return 0.5*abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def merge_close(pts, frac):
    if frac is None or len(pts) < 3:
        return pts
    sp = float(np.median(cKDTree(pts).query(pts, k=2)[0][:, 1])); pairs = cKDTree(pts).query_pairs(frac*sp)
    if not pairs:
        return pts
    parent = np.arange(len(pts))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j in pairs:
        a, b = find(i), find(j)
        if a != b: parent[a] = b
    roots = np.array([find(i) for i in range(len(pts))]); out = []
    for r in np.unique(roots):
        out.append(pts[roots == r].mean(0))
    return np.array(out)


def cells(pts):
    """Bounded Voronoi cells inside the hull: areas, sides, complete flag, Delaunay-neighbour area medians."""
    if len(pts) < 20:
        return None
    try:
        vor = Voronoi(pts); tri = Delaunay(pts)
    except Exception:
        return None
    d, _ = cKDTree(pts).query(pts, k=2); med = np.median(d[:, 1]); indptr, indices = tri.vertex_neighbor_vertices
    area = np.full(len(pts), np.nan); sides = np.zeros(len(pts), int); complete = np.zeros(len(pts), bool)
    for i in range(len(pts)):
        r = vor.regions[vor.point_region[i]]
        if -1 in r or len(r) == 0:
            continue
        v = vor.vertices[r]
        if (tri.find_simplex(v) < 0).any():
            continue
        area[i] = poly_area(v); sides[i] = len(r)
        nb = indices[indptr[i]:indptr[i+1]]; dn = np.linalg.norm(pts[nb]-pts[i], axis=1); complete[i] = len(nb) >= 4 and not (dn > 2.2*med).any()
    ok = np.isfinite(area)
    local = np.full(len(pts), np.nan)
    for i in np.where(ok)[0]:
        nb = indices[indptr[i]:indptr[i+1]]; a = area[nb]; a = a[np.isfinite(a)]
        local[i] = np.median(a) if len(a) >= 3 else np.nanmedian(area[ok])
    return dict(area=area, sides=sides, complete=complete, local=local, ok=ok)


def disp(a, kind):
    a = np.asarray(a, float); a = a[np.isfinite(a) & (a > 0)]
    if len(a) < 12:
        return np.nan
    if kind == 'cv':
        return a.std()/a.mean()
    if kind.startswith('trim'):
        t = int(kind[4:])/100; lo, hi = np.quantile(a, (t, 1-t)); b = a[(a >= lo) & (a <= hi)]; return b.std()/b.mean() if len(b) >= 8 else np.nan
    if kind == 'mad':
        m = np.median(a); return 1.4826*np.median(np.abs(a-m))/m
    if kind == 'mix':
        x = np.log(a); mu = np.median(x); sig = max(1.4826*np.median(np.abs(x-mu)), 0.05); pi = np.array([0.8, 0.1, 0.1]); off = np.array([0.0, -np.log(2), np.log(2)])
        for _ in range(60):
            ll = -0.5*((x[:, None]-mu-off[None, :])/sig)**2+np.log(pi)[None, :]; ll -= ll.max(1, keepdims=True); r = np.exp(ll); r /= r.sum(1, keepdims=True)
            pi = np.maximum(r.mean(0), 1e-3); pi /= pi.sum(); mu = float((r*(x[:, None]-off[None, :])).sum()/len(x)); sig = float(np.sqrt(max((r*(x[:, None]-mu-off[None, :])**2).sum()/len(x), 1e-4)))
        w = r[:, 0]
        if w.sum() < 6:
            return np.nan
        m = (w*a).sum()/w.sum(); v = (w*(a-m)**2).sum()/w.sum(); return float(np.sqrt(v)/m)
    raise ValueError(kind)


def estimate_all(pts):
    """All estimator variants from one point set -> dict variant -> CV."""
    out = {}
    for mg in MERGE:
        p = merge_close(pts, mg); c = cells(p)
        for sl in SLIVER:
            for comp in COMPLETE:
                if c is None:
                    for dk in DISP: out[(mg, sl, comp, dk)] = np.nan
                    continue
                keep = c['ok'].copy()
                if comp: keep &= c['complete']
                if sl is not None: keep &= ~(c['area'] < sl*c['local'])
                a = c['area'][keep]
                for dk in DISP:
                    out[(mg, sl, comp, dk)] = disp(a, dk)
    return out


def tune_task(args):
    axis, level, rep = args
    rng = np.random.default_rng([rep, {'miss': 1, 'fp': 2, 'jitter': 3, 'grid': 5}[axis], int(round((level if not isinstance(level, tuple) else level[0]*10+level[1])*1000))])
    out = {}
    for n in E4.NAMES:
        if axis == 'grid':
            p = E4.degrade(n, 'miss', level[0], rng); save = E4.PTS[n]; E4.PTS[n] = p; p = E4.degrade(n, 'fp', level[1], rng); E4.PTS[n] = save
        else:
            p = E4.degrade(n, axis, level, rng)
        out[n] = estimate_all(p)
    return (axis, level, rep), out


def tune(a):
    tasks = [(axis, lv, rep) for axis, lvs in TUNE_LEVELS.items() for lv in lvs for rep in range(a.repeats)]+[('grid', g, rep) for g in E4.GRID for rep in range(a.repeats)]
    t0 = time.time(); w = cpu_pool.n_workers(); print('tune tasks', len(tasks), 'workers', w, flush=True)
    results = dict(cpu_pool.pmap(tune_task, tasks, chunksize=4, workers=w))
    variants = list(itertools.product(MERGE, SLIVER, COMPLETE, DISP)); gt = [E4.GT[n]['CV'] for n in E4.NAMES]
    by = {}
    for (axis, level, rep), out in results.items():
        by.setdefault((axis, level), []).append(out)
    table = {}      # (axis, level) -> variant -> mean LOO MAPE over repeats
    for key, reps in by.items():
        table[key] = {}
        for v in variants:
            ms = [E4.loo_scaled([out[n][v] for n in E4.NAMES], gt)[0] for out in reps]
            table[key][v] = float(np.nanmean(ms))
    sel_keys = [('fp', lv) for lv in TUNE_LEVELS['fp'][1:]]+[('grid', g) for g in E4.GRID]+[('miss', 0)]
    score = {v: float(np.mean([table[k][v] for k in sel_keys])) for v in variants}
    zero = {v: table[('miss', 0)][v] for v in variants}
    ranked = sorted([v for v in variants if zero[v] <= 10.0], key=lambda v: score[v])
    best = ranked[0]; plain = (None, None, False, 'cv')
    def be(v, axis):
        levels = TUNE_LEVELS[axis]; ys = [table[(axis, lv)][v] for lv in levels]; b = E4.B25['CV']
        for k in range(1, len(levels)):
            if ys[k-1] < b <= ys[k]:
                return levels[k-1]+(b-ys[k-1])*(levels[k]-levels[k-1])/(ys[k]-ys[k-1])
        return None if ys[-1] < b else levels[0]
    res = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), repeats=a.repeats, workers=w, seconds=round(time.time()-t0, 1), selection_keys=[str(k) for k in sel_keys],
               chosen=dict(merge=best[0], sliver=best[1], complete=best[2], disp=best[3], score=score[best], zero_mape=zero[best]),
               top10=[dict(variant=str(v), score=score[v], zero=zero[v]) for v in ranked[:10]], plain=dict(score=score[plain], zero=zero[plain]),
               break_even_chosen={axis: be(best, axis) for axis in TUNE_LEVELS}, break_even_plain={axis: be(plain, axis) for axis in TUNE_LEVELS},
               curves={axis: {str(lv): dict(chosen=table[(axis, lv)][best], plain=table[(axis, lv)][plain], complete_cv=table[(axis, lv)][(None, None, True, 'cv')]) for lv in TUNE_LEVELS[axis]} for axis in TUNE_LEVELS},
               grid={str(g): dict(chosen=table[('grid', g)][best], plain=table[('grid', g)][plain]) for g in E4.GRID})
    (OUT/'tune.json').write_text(json.dumps(res, indent=1, default=float))
    print('chosen', res['chosen'], flush=True); print('plain score %.2f zero %.2f' % (score[plain], zero[plain]), flush=True)
    for d in res['top10']: print('  ', d, flush=True)
    for axis in TUNE_LEVELS:
        print(axis, 'chosen', {lv: round(table[(axis, lv)][best], 2) for lv in TUNE_LEVELS[axis]}, 'break-even chosen', res['break_even_chosen'][axis], 'plain', res['break_even_plain'][axis], flush=True)
    print('grid chosen', {str(g): round(table[('grid', g)][best], 2) for g in E4.GRID}, flush=True); print('done', round(time.time()-t0), 's', flush=True)


# ---------------- apply ----------------
def frozen():
    c = json.loads((OUT/'tune.json').read_text())['chosen']; return (c['merge'], c['sliver'], c['complete'], c['disp'])


def estimate_one(pts, params):
    mg, sl, comp, dk = params
    p = merge_close(pts, mg); c = cells(p)
    if c is None:
        return np.nan
    keep = c['ok'].copy()
    if comp: keep &= c['complete']
    if sl is not None: keep &= ~(c['area'] < sl*c['local'])
    return disp(c['area'][keep], dk)


_A = {}


def apply_task(item):
    fam, r, pts = item; params = _A['params']; box = _A['box'][r]
    if pts is None or len(pts) < 20:
        return fam, r, np.nan, np.nan
    full = estimate_one(pts, params)
    inb = (pts[:, 0] >= box[0]) & (pts[:, 0] < box[2]) & (pts[:, 1] >= box[1]) & (pts[:, 1] < box[3])
    win = estimate_one(pts[inb], params) if inb.sum() >= 20 else np.nan
    return fam, r, full, win


def apply(a):
    import cd_estimators_20261028 as P
    import eval_oof_20261031 as H
    import green_day2_cd_features_20261101 as D
    import green_t4_numerics_20261101 as M
    V = M.V; params = frozen(); print('frozen estimator', params, flush=True)
    y, tta4, baseA, fold, seen, keys, F, aa = P.load(); ok = ~seen; n = len(y)
    with np.load(P.E/'recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True) as h:
        o = np.argsort(h['indices']); B = h['B_hybrid_w0p5'][o]
    base = H.load_base(); st = H.image_stats(base); framed = np.zeros(n, int); framed[base['idx']] = (st['black_frac'] > 0.01)
    box = np.load(ROOT/'e3_boxpool/cluster_boxes.npz')['box']
    GT_ = Path('experiments/green_track_20261028'); D2 = Path('experiments/green_day2_20261101')
    sources = {}
    for fam in M.ACTIVE_ORDER:
        kind, src = M.BF16_SOURCES[fam]; pk = [None]*n
        for f in range(5):
            with np.load(src(f)) as z:
                if kind == 'maps':
                    for r, m in zip(z['indices'], z['probability']): pk[int(r)] = V.peaks_sg(m)[:, :2].astype(np.float64)
                else:
                    rows, peaks, off = z['indices'], z['peaks'], z['offsets']
                    for i, r in enumerate(rows):
                        q = peaks[off[i]:off[i+1]]; pk[int(r)] = q[q[:, 2] >= 0.35, :2].astype(np.float64)
        sources[fam] = pk
    for fam, d in {'tta_s49': D2/'preonly_tta/s49', 'tta_s46': D2/'preonly_tta/s46'}.items():
        if (d/'fold4'/'fold4_sg2_peaks.npz').exists():
            pk, _ = D.load_peaks(d); sources[fam] = [None if p is None else p[p[:, 2] >= 0.35, :2].astype(np.float64) for p in pk]
    cp_cent = [None]*n; cp_area = [None]*n
    for p in glob.glob(str(ROOT/'e1_cpsam/v2/rows_*.npz')):
        with np.load(p, allow_pickle=True) as z:
            for r, stt in zip(z['rows'], z['stats']):
                if stt.get('n', 0) >= 20:
                    cp_cent[int(r)] = stt['centroid'].astype(np.float64); cp_area[int(r)] = (stt['centroid'].astype(np.float64), stt['area'].astype(np.float64), stt['border'])
    if any(c is not None for c in cp_cent):
        sources['cpsam_centroids'] = cp_cent
    _A['params'] = params; _A['box'] = box
    items = [(fam, r, pk[r]) for fam, pk in sources.items() for r in range(n)]
    t0 = time.time(); w = cpu_pool.n_workers(); print('apply items', len(items), 'workers', w, flush=True)
    feats = {fam: dict(full=np.full(n, np.nan), win=np.full(n, np.nan)) for fam in sources}
    for fam, r, full, win in cpu_pool.pmap(apply_task, items, chunksize=64, workers=w, ordered=False):
        feats[fam]['full'][r] = full; feats[fam]['win'][r] = win
    print('point sources done', round(time.time()-t0), 's', flush=True)
    # CPSAM mask areas directly (no Voronoi): sliver cut vs global median, chosen dispersion, window / all interior cells
    if any(c is not None for c in cp_area):
        mg, sl, comp, dk = params; fa = np.full(n, np.nan); fw = np.full(n, np.nan)
        for r in range(n):
            if cp_area[r] is None: continue
            c, ar, bd = cp_area[r]; inter = ~bd; keep = inter.copy()
            if sl is not None: keep &= ~(ar < sl*np.median(ar[inter]))
            fa[r] = disp(ar[keep], dk)
            inb = (c[:, 0] >= box[r][0]) & (c[:, 0] < box[r][2]) & (c[:, 1] >= box[r][1]) & (c[:, 1] < box[r][3])
            fw[r] = disp(ar[keep & inb], dk) if (keep & inb).sum() >= 12 else np.nan
        feats['cpsam_mask_area'] = dict(full=fa, win=fw)
    np.savez_compressed(OUT/'features.npz', **{f'{fam}_{k}': v for fam, d in feats.items() for k, v in d.items()})
    D.STRATA_NAMES = {0: 'unframed', 1: 'framed'}
    res = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), params=dict(merge=params[0], sliver=params[1], complete=params[2], disp=params[3]), cv={})
    def run(name, f):
        r = P.lofo_blend(y, B, fold, ok, np.where(np.isfinite(f) & (f > 0), f, np.nan), t=1); s, _ = D.lofo_strat(y, B, fold, ok, f, framed, t=1)
        d = dict(delta_mean_pp=r['overall']['delta_cd_pp']/3, delta_cv_pp=r['overall']['delta_cd_pp'], folds_improving=r['overall']['folds_improving'], w=[r[str(h)]['w'] for h in range(5)], est_mape=r['overall']['est_cd_mape'], err_corr=r['overall']['err_corr_mean'],
                 per_fold_mean_pp=[r[str(h)]['delta_cd_pp']/3 for h in range(5)], strat_mean_pp=s['overall']['delta_mean_pp'], per_stratum_pp=s['overall']['per_stratum_pp'], n_with_estimate=int(np.isfinite(f[ok]).sum()))
        d['evidence'] = bool(d['delta_mean_pp'] <= -0.005 and d['folds_improving'] >= 4); d['adoption'] = bool(d['delta_mean_pp'] <= -0.02 and d['folds_improving'] >= 4 and d['per_fold_mean_pp'][4] < 0)
        print('%-26s CV plain %.4f mean pp (CV %.4f, impr %d, est %.2f, r_err %.2f, w %s, n %d) | strat %.4f framed %.4f unframed %.4f %s' % (name, d['delta_mean_pp'], d['delta_cv_pp'], d['folds_improving'], d['est_mape'], d['err_corr'], d['w'], d['n_with_estimate'], d['strat_mean_pp'], d['per_stratum_pp'].get('framed', np.nan), d['per_stratum_pp'].get('unframed', np.nan), 'EVIDENCE' if d['evidence'] else ''), flush=True)
        return d
    for fam, d in feats.items():
        for k in ('win', 'full'):
            res['cv'][f'{fam}_{k}'] = run(f'{fam}_{k}', d[k])
    for k in ('win', 'full'):
        A = np.log(np.stack([feats[f][k] for f in M.ACTIVE_ORDER]))
        with np.errstate(all='ignore'):
            res['cv'][f'mean_ten_{k}'] = run(f'mean_ten_{k}', np.exp(np.nanmean(A, 0)))
        fams = [f for f in ('sg2pre_s46', 'tta_s46', 'tta_s49', 'cpsam_centroids') if f in feats]
        A = np.log(np.stack([feats[f][k] for f in fams]))
        with np.errstate(all='ignore'):
            res['cv'][f'mean_fullframe_{k}'] = run(f'mean_fullframe_{k}', np.exp(np.nanmean(A, 0)))
    res['gates_passed'] = {k: dict(evidence=v['evidence'], adoption=v['adoption']) for k, v in res['cv'].items() if v['evidence'] or v['adoption']}
    (OUT/'apply.json').write_text(json.dumps(res, indent=1, default=float)); print('GATES', json.dumps(res['gates_passed']), flush=True); print('done', round(time.time()-t0), 's', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['tune', 'apply']); ap.add_argument('--repeats', type=int, default=TUNE_REPEATS)
    a = ap.parse_args(); {'tune': tune, 'apply': apply}[a.command](a)
