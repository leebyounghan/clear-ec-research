"""Green day 3 / E4 (pre-registered 2026-11-03 16:10 UTC, PI-approved theory step): error decomposition on the 25 expert
marker sets.  Which detection error drives each metric's loss vs the oracle, and what accuracy beats B there?
  simulate : degrade the expert points one axis at a time (misses; realistic false positives; localisation jitter;
             window shift) at 8 levels x 30 repeats, plus a miss x FP grid; CD / CV / HEX (hex6) with the p1 oracle
             definition (bounded cells) and LOO scaling; MAPE and r vs GT; break-even per axis vs B on the 25
             (CD 7.66 / CV 17.93 / HEX 11.45).  Secondary estimator: complete-cell filter (degradation_sim).
  sources  : the real point sources measured against the expert points on the same axes (miss, FP, jitter, window IoU).
CPU only; pool sized by cpu_pool (free quota), fork after loading, chunked imap; Voronoi cached per task.
"""
import argparse, json, sys, time
import numpy as np
from pathlib import Path
from scipy.spatial import Voronoi, Delaunay, cKDTree
sys.path.insert(0, 'experiments/green_day3_20261103/source'); sys.path.insert(0, 'experiments/green_track_20261028/source'); sys.path.insert(0, 'experiments/green_day2_20261101/source')
import cpu_pool
ROOT = Path('experiments/green_day3_20261103'); OUT = ROOT/'e4_degradation'; OUT.mkdir(parents=True, exist_ok=True)
ANN = 'datasets/clear_ec_official/green_eval/annotations.json'; UM = 1000/1296
B25 = dict(CD=7.66, CV=17.93, HEX=11.45)
LEVELS = dict(miss=[0, .05, .1, .15, .2, .25, .3, .4], fp=[0, .05, .1, .15, .2, .25, .3, .4], jitter=[0, .05, .1, .15, .2, .25, .3, .4], window=[1.0, .9, .8, .7, .6, .5])
GRID = [(m, f) for m in (0, .1, .2, .3) for f in (0, .1, .2, .3)]
REPEATS = 30
ann = json.load(open(ANN)); NAMES = list(ann); PTS = {n: np.asarray(ann[n]['markers_mha'], float) for n in NAMES}; GT = {n: ann[n]['gt'] for n in NAMES}; BOX = {n: ann[n]['roi_box_mha'] for n in NAMES}
SPACING = {n: float(np.median(cKDTree(PTS[n]).query(PTS[n], k=2)[0][:, 1])) for n in NAMES}


def poly_area(v):
    x, y = v[:, 0], v[:, 1]; return 0.5*abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def estimates(pts, complete=False):
    """(CD, CV, hex6, n_bounded) from a point set with the p1 oracle definition; complete=True adds the degradation_sim
    neighbour-distance filter (all Delaunay neighbours within 2.2x median NN distance, >= 4 neighbours)."""
    if len(pts) < 20:
        return (np.nan, np.nan, np.nan, 0)
    try:
        vor = Voronoi(pts); tri = Delaunay(pts)
    except Exception:
        return (np.nan, np.nan, np.nan, 0)
    if complete:
        d, _ = cKDTree(pts).query(pts, k=2); med = np.median(d[:, 1]); indptr, indices = tri.vertex_neighbor_vertices
    areas, sides = [], []
    for i in range(len(pts)):
        r = vor.regions[vor.point_region[i]]
        if -1 in r or len(r) == 0:
            continue
        v = vor.vertices[r]
        if (tri.find_simplex(v) < 0).any():
            continue
        if complete:
            nb = indices[indptr[i]:indptr[i+1]]; dn = np.linalg.norm(pts[nb]-pts[i], axis=1)
            if len(nb) < 4 or (dn > 2.2*med).any():
                continue
        areas.append(poly_area(v)*UM*UM); sides.append(len(r))
    if len(areas) < 12:
        return (np.nan, np.nan, np.nan, len(areas))
    areas = np.array(areas); sides = np.array(sides)
    return (1e6/areas.mean(), areas.std()/areas.mean(), (sides == 6).mean(), len(areas))


def degrade(n, axis, level, rng):
    pts = PTS[n]; sp = SPACING[n]
    if axis == 'miss':
        keep = rng.random(len(pts)) >= level; return pts[keep]
    if axis == 'jitter':
        return pts+rng.normal(0, level*sp, pts.shape)
    if axis == 'fp':
        nfp = int(round(level*len(pts)))
        if nfp == 0:
            return pts
        vor = Voronoi(pts); fps = []
        for _ in range(nfp):
            i = int(rng.integers(len(pts)))
            if rng.random() < 0.5:                                   # on the boundary of cell i (Voronoi edge midpoint)
                r = vor.regions[vor.point_region[i]]
                if -1 in r or len(r) < 3:
                    fps.append(pts[i]+rng.normal(0, 0.35*sp, 2)); continue
                v = vor.vertices[r]; k = int(rng.integers(len(v))); fps.append(0.5*(v[k]+v[(k+1) % len(v)]))
            else:                                                    # near-cell duplicate at ~0.35 spacing
                th = rng.uniform(0, 2*np.pi); fps.append(pts[i]+0.35*sp*np.array([np.cos(th), np.sin(th)]))
        return np.vstack([pts, np.array(fps)])
    if axis == 'window':
        x0, y0 = pts.min(0); x1, y1 = pts.max(0); w, h = x1-x0, y1-y0
        if level >= 1.0:
            return pts
        # shift along a random direction so that IoU(window, shifted) = level: IoU = (w-dx)(h-dy)/(2wh-(w-dx)(h-dy))
        th = rng.uniform(0, 2*np.pi); target = level/(1+level)      # intersection fraction of one window
        lo, hi = 0.0, max(w, h)
        for _ in range(40):
            s = 0.5*(lo+hi); dx, dy = abs(s*np.cos(th)), abs(s*np.sin(th))
            frac = max(0, w-dx)*max(0, h-dy)/(w*h)
            if frac > target: lo = s
            else: hi = s
        dx, dy = s*np.cos(th), s*np.sin(th)
        inside = (pts[:, 0] >= x0+dx) & (pts[:, 0] <= x1+dx) & (pts[:, 1] >= y0+dy) & (pts[:, 1] <= y1+dy)
        return pts[inside]
    raise ValueError(axis)


def task(args):
    axis, level, rep = args
    rng = np.random.default_rng(1000*rep+hash(axis) % 1000+int(level*1000) if isinstance(level, float) else rep)
    rng = np.random.default_rng([rep, {'miss': 1, 'fp': 2, 'jitter': 3, 'window': 4, 'grid': 5}[axis], int(round((level if not isinstance(level, tuple) else level[0]*10+level[1])*1000))])
    out = {}
    for n in NAMES:
        if axis == 'grid':
            p = degrade(n, 'miss', level[0], rng); PTS_save = PTS[n]; PTS[n] = p; p = degrade(n, 'fp', level[1], rng); PTS[n] = PTS_save
        else:
            p = degrade(n, axis, level, rng)
        out[n] = dict(zip(('p', 'c'), (estimates(p, False), estimates(p, True))))
    return (axis, level, rep), out


def loo_scaled(est, gt):
    est = np.asarray(est, float); gt = np.asarray(gt, float); ok = np.isfinite(est) & (est > 0); e, g = est[ok], gt[ok]
    if ok.sum() < 5:
        return np.nan, np.nan, int(ok.sum())
    ap = []
    for i in range(len(e)):
        m = np.arange(len(e)) != i; s = np.median(g[m]/e[m]); ap.append(abs(e[i]*s-g[i])/g[i]*100)
    return float(np.mean(ap)), float(np.corrcoef(e, g)[0, 1]) if len(e) > 2 else np.nan, int(ok.sum())


def summarise(results):
    """results: {(axis, level, rep): {name: {'p': (cd,cv,hex,n), 'c': ...}}} -> per (axis, level): mean/sd over repeats of LOO MAPE and r."""
    by = {}
    for (axis, level, rep), out in results.items():
        by.setdefault((axis, level), []).append(out)
    table = {}
    for (axis, level), reps in sorted(by.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        rec = {}
        for est_key, est_name in (('p', 'oracle'), ('c', 'complete')):
            for j, metric in enumerate(('CD', 'CV', 'HEX')):
                mapes, rs, ns = [], [], []
                for out in reps:
                    est = [out[n][est_key][j] for n in NAMES]; gt = [GT[n][metric] for n in NAMES]
                    m, r, k = loo_scaled(est, gt); mapes.append(m); rs.append(r); ns.append(k)
                rec[f'{est_name}_{metric}'] = dict(mape=float(np.nanmean(mapes)), mape_sd=float(np.nanstd(mapes)), r=float(np.nanmean(rs)), n_images=float(np.mean(ns)))
            rec[f'{est_name}_cells'] = float(np.mean([np.mean([out[n][est_key][3] for n in NAMES]) for out in reps]))
        table[f'{axis}|{level}'] = rec
    return table


def break_even(table, axis):
    """Per metric: the lowest level at which the oracle-definition MAPE reaches B on the 25 (linear interpolation)."""
    levels = LEVELS[axis]; out = {}
    for metric in ('CD', 'CV', 'HEX'):
        ys = [table[f'{axis}|{lv}'][f'oracle_{metric}']['mape'] for lv in levels]; b = B25[metric]
        xs = list(levels) if axis != 'window' else [1-lv for lv in levels]          # window: use 1-IoU as the 'error'
        be = None
        for k in range(1, len(xs)):
            if (ys[k-1] < b) != (ys[k] < b) or (k == 1 and ys[0] >= b):
                if ys[0] >= b: be = xs[0]; break
                be = xs[k-1]+(b-ys[k-1])*(xs[k]-xs[k-1])/(ys[k]-ys[k-1]) if ys[k] != ys[k-1] else xs[k]; break
        out[metric] = dict(break_even=be, never_reached=bool(be is None and ys[-1] < b), already_above_at_zero=bool(ys[0] >= b), mape_at_zero=ys[0], mape_at_max=ys[-1])
    return out


def simulate(a):
    tasks = [(axis, lv, rep) for axis, lvs in LEVELS.items() for lv in lvs for rep in range(a.repeats)]+[('grid', g, rep) for g in GRID for rep in range(a.repeats)]
    t0 = time.time(); w = cpu_pool.n_workers(); print('tasks', len(tasks), 'workers', w, flush=True)
    results = dict(cpu_pool.pmap(task, tasks, chunksize=8, workers=w))
    table = summarise(results)
    res = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), repeats=a.repeats, levels=LEVELS, grid=GRID, B25=B25, workers=w, seconds=round(time.time()-t0, 1), table=table,
               break_even={axis: break_even(table, axis) for axis in LEVELS}, spacing_px=dict(median=float(np.median(list(SPACING.values()))), min=float(min(SPACING.values())), max=float(max(SPACING.values()))))
    (OUT/'simulate.json').write_text(json.dumps(res, indent=1, default=float))
    for axis in LEVELS:
        print('axis', axis)
        for lv in LEVELS[axis]:
            rec = table[f'{axis}|{lv}']; print('  %-5s oracle CD %.2f CV %.2f HEX %.2f (r %.2f/%.2f/%.2f, cells %.0f) | complete CV %.2f HEX %.2f' % (lv, rec['oracle_CD']['mape'], rec['oracle_CV']['mape'], rec['oracle_HEX']['mape'], rec['oracle_CD']['r'], rec['oracle_CV']['r'], rec['oracle_HEX']['r'], rec['oracle_cells'], rec['complete_CV']['mape'], rec['complete_HEX']['mape']), flush=True)
        print('  break-even vs B:', {m: (round(v['break_even'], 3) if v['break_even'] is not None else ('never' if v['never_reached'] else 'at 0')) for m, v in res['break_even'][axis].items()}, flush=True)
    print('grid (miss, fp): oracle CV / HEX MAPE'); print({str(g): (round(table[f'grid|{g}']['oracle_CV']['mape'], 2), round(table[f'grid|{g}']['oracle_HEX']['mape'], 2)) for g in GRID}, flush=True)
    print('done', round(time.time()-t0), 's', flush=True)


def sources(a):
    """Real point sources vs expert points inside the expert ROI box: miss rate, FP rate, jitter / spacing, window IoU."""
    import green_day2_cd_features_20261101 as D
    GT_ = Path('experiments/green_track_20261028'); binding = json.load(open('experiments/green_point_probe_20261023/data_binding.json')); train_ids = set(binding['train_ids'])
    with np.load('datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True) as z:
        ids = z['ids'].astype(str)
    pos = {n: i for i, n in enumerate(ids)}
    fams = {'sg2_s46': GT_/'sg2_specialist', 'sg2_s47': GT_/'sg2_specialist_s47', 'sg2_s48': GT_/'sg2_specialist_s48', 'sg2_s49': GT_/'sg2_specialist_s49', 'sg2_s50': GT_/'sg2_specialist_s50',
            'sg2pre_s46': GT_/'external_srep/preonly', 'tta_s46': ROOT.parent/'green_day2_20261101/preonly_tta/s46', 'tta_s49': ROOT.parent/'green_day2_20261101/preonly_tta/s49'}
    box_tab = np.load(ROOT/'e3_boxpool/cluster_boxes.npz')['box']
    cp = {}
    import glob
    for p in glob.glob(str(ROOT/'e1_cpsam/v2/rows_*.npz')):
        with np.load(p, allow_pickle=True) as z:
            for r, st in zip(z['rows'], z['stats']):
                if int(r) in {pos[n] for n in NAMES}: cp[int(r)] = st
    out = {}
    def measure(name, det, r, is_val):
        gt = PTS[name]; bx = BOX[name]; sp = SPACING[name]
        # detections inside the convex hull of the expert markers (the annotation covers only that part of the ROI box),
        # dilated by half a spacing so boundary cells count
        hull = Delaunay(gt); d_ = det[hull.find_simplex(det) >= 0] if len(det) else det
        if len(det):
            dmin, _ = cKDTree(gt).query(det); d_ = det[(hull.find_simplex(det) >= 0) | (dmin <= 0.5*sp)]
        if len(d_) == 0:
            return dict(miss=1.0, fp=0.0, jitter=np.nan, n_det=0, n_gt=int(len(gt)), val=is_val)
        d, j = cKDTree(d_).query(gt); used = set(); tp = []; 
        for dd, jj, gi in sorted(zip(d, j, range(len(gt)))):
            if dd <= 0.5*sp and jj not in used: used.add(jj); tp.append(dd)
        miss = 1-len(tp)/len(gt); fp = (len(d_)-len(tp))/len(gt); jit = float(np.median(tp)/sp) if tp else np.nan
        x0, y0, x1, y1 = box_tab[r]; ix = max(0, min(x1, bx[2])-max(x0, bx[0])); iy = max(0, min(y1, bx[3])-max(y0, bx[1])); inter = ix*iy
        iou = inter/((x1-x0)*(y1-y0)+(bx[2]-bx[0])*(bx[3]-bx[1])-inter)
        return dict(miss=miss, fp=fp, jitter=jit, window_iou=float(iou), n_det=int(len(d_)), n_gt=int(len(gt)), val=is_val)
    for fam, d in fams.items():
        if not (d/'fold4'/'fold4_sg2_peaks.npz').exists():
            continue
        pk, seen = D.load_peaks(d); rec = {}
        for n in NAMES:
            r = pos[n]; p = pk[r]
            if p is None: continue
            rec[n] = measure(n, p[p[:, 2] >= 0.35, :2], r, n not in train_ids)
        out[fam] = rec
    if cp:
        out['cpsam_v2'] = {n: measure(n, cp[pos[n]]['centroid'].astype(float), pos[n], n not in train_ids) for n in NAMES if pos[n] in cp and cp[pos[n]].get('n', 0) > 0}
    summary = {}
    for fam, rec in out.items():
        for subset, sel in (('val5', [n for n in rec if rec[n]['val']]), ('all25', list(rec))):
            if not sel: continue
            summary[f'{fam}|{subset}'] = {k: float(np.nanmean([rec[n][k] for n in sel])) for k in ('miss', 'fp', 'jitter', 'window_iou') if any(k in rec[n] for n in sel)}
            summary[f'{fam}|{subset}']['n_images'] = len(sel)
    (OUT/'sources.json').write_text(json.dumps(dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), match_radius='0.5 x median NN spacing', region='convex hull of the expert markers dilated by 0.5 spacing (not the full ROI box)', summary=summary, per_image=out), indent=1, default=float))
    for k, v in summary.items():
        print('%-22s' % k, {kk: round(vv, 3) for kk, vv in v.items()}, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['simulate', 'sources']); ap.add_argument('--repeats', type=int, default=REPEATS)
    a = ap.parse_args(); {'simulate': simulate, 'sources': sources}[a.command](a)
