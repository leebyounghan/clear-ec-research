"""Green day 2 / D1 + D2: CD and HEX features from full-frame (pre-trained-only) detector peaks, tissue-aware and
ROI-window based, evaluated as stratified LOFO blends into hybrid B.  CPU only.

    python experiments/green_day2_20261101/source/green_day2_cd_features_20261101.py features [--limit N]
    python experiments/green_day2_20261101/source/green_day2_cd_features_20261101.py evaluate
"""
import argparse, json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import multiprocessing as mp
from pathlib import Path
import numpy as np
import cv2
cv2.setNumThreads(1)
from scipy.spatial import Voronoi, cKDTree
from scipy.ndimage import binary_fill_holes

sys.path.insert(0, 'experiments/eval_harness_20261031/source')
sys.path.insert(0, 'experiments/green_track_20261028/source')
import eval_oof_20261031 as H
import cd_estimators_20261028 as P

ROOT = Path('experiments/green_day2_20261101'); OUT = ROOT/'d1_cd_features'
RAW = Path('datasets/clear_ec_fullframe_20260904/img_full_raw.dat')
GT = Path('experiments/green_track_20261028')
N_TOTAL, HGT, WID = 8954, 972, 1296
UM = 1000/1296; PX2MM2 = (UM/1000)**2
ROI_W, ROI_H = 538, 408; ROI_MM2 = ROI_W*ROI_H*PX2MM2
BIN = 8; WB, HB = ROI_W//BIN, ROI_H//BIN            # 67 x 51 bins
FAMS = {'pre_s46': GT/'external_srep/preonly', 'pre_s47': GT/'external_srep/s47_preonly', 'pre_s48': GT/'external_srep/s48_preonly'}
COMPACT = GT/'sg2_specialist'
THRESHOLDS = (0.35, 0.5)
INTERIOR_PX = 15.0
STRATA_NAMES = {0: 'unframed', 1: 'framed'}


def log(msg, **kw):
    print(json.dumps(dict(utc=H.datetime.datetime.now(H.datetime.timezone.utc).isoformat() if hasattr(H, 'datetime') else time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), msg=msg, **kw)), flush=True)


def load_peaks(d):
    pk = [None]*N_TOTAL; seen = np.zeros(N_TOTAL, bool)
    for f in range(5):
        with np.load(d/f'fold{f}'/f'fold{f}_sg2_peaks.npz') as z:
            rows, peaks, off, flag = z['indices'], z['peaks'], z['offsets'], z['point_labels_seen']
            for i, r in enumerate(rows):
                pk[int(r)] = np.asarray(peaks[off[i]:off[i+1]], dtype=np.float64)
            seen[rows[flag]] = True
    return pk, seen


def tissue_mask(a_half):
    """Half-resolution tissue mask (non-black) and distance (full-res px) to black or to the image border."""
    m = (a_half > 5).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = binary_fill_holes(m.astype(bool))
    pad = np.pad(m.astype(np.uint8), 1)
    dt = cv2.distanceTransform(pad, cv2.DIST_L2, 3)[1:-1, 1:-1]*2.0
    return m, dt


def poly_area(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5*abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def voronoi_stats(pts, dist_px):
    """pts: (n,2) full-res px; dist_px: distance of each point to black/border.  Returns dict of estimators."""
    out = dict(vor_cd=np.nan, vor_cd_hex=np.nan, hex6_all=np.nan, hex6_interior=np.nan, n_interior=0, nn_cd=np.nan)
    if len(pts) < 12:
        return out
    try:
        vor = Voronoi(pts)
    except Exception:
        return out
    areas, sides, interior, bounded = [], [], [], []
    for i, ri in enumerate(vor.point_region):
        reg = vor.regions[ri]
        if len(reg) == 0 or -1 in reg:
            bounded.append(False); areas.append(np.nan); sides.append(0); interior.append(False); continue
        v = vor.vertices[reg]
        inside = (v[:, 0] >= 0).all() and (v[:, 1] >= 0).all() and (v[:, 0] <= WID-1).all() and (v[:, 1] <= HGT-1).all()
        bounded.append(bool(inside)); areas.append(poly_area(v)); sides.append(len(reg)); interior.append(bool(inside and dist_px[i] >= INTERIOR_PX))
    areas, sides, interior, bounded = np.array(areas), np.array(sides), np.array(interior), np.array(bounded)
    if bounded.sum() >= 6:
        out['hex6_all'] = float((sides[bounded] == 6).mean())
    if interior.sum() >= 6:
        a = areas[interior]
        med = np.median(a)
        keep = a <= 3*med
        if keep.sum() >= 6:
            out['vor_cd'] = float(1.0/(np.median(a[keep])*PX2MM2))
        h6 = interior & (sides == 6)
        out['hex6_interior'] = float((sides[interior] == 6).mean())
        if h6.sum() >= 6:
            out['vor_cd_hex'] = float(1.0/(np.median(areas[h6])*PX2MM2))
        out['n_interior'] = int(interior.sum())
        q = pts[interior]
        if len(q) >= 6:
            d, _ = cKDTree(q).query(q, k=2)
            s = np.median(d[:, 1])
            out['nn_cd'] = float(2.0/(np.sqrt(3)*s*s*PX2MM2))
    return out


def window_stats(pts, scores, tissue_half, centre):
    """538x408 windows on an 8-px bin grid; returns counts for quality / dense / typical / cluster windows and the
    cluster-window hex6 generators mask."""
    ny, nx = (HGT+BIN-1)//BIN, WID//BIN               # 122 x 162
    cnt = np.zeros((ny, nx)); ssum = np.zeros((ny, nx))
    if len(pts):
        bx = np.clip((pts[:, 0]//BIN).astype(int), 0, nx-1); by = np.clip((pts[:, 1]//BIN).astype(int), 0, ny-1)
        np.add.at(cnt, (by, bx), 1); np.add.at(ssum, (by, bx), scores)
    t = tissue_half.astype(np.float32)
    th = t[:(ny*4), :].copy() if t.shape[0] >= ny*4 else np.pad(t, ((0, ny*4-t.shape[0]), (0, 0)))
    tb = th.reshape(ny, 4, nx, 4).mean((1, 3))
    k = (HB, WB)
    C = cv2.boxFilter(cnt, -1, (WB, HB), normalize=False, borderType=cv2.BORDER_CONSTANT)
    S = cv2.boxFilter(ssum, -1, (WB, HB), normalize=False, borderType=cv2.BORDER_CONSTANT)
    T = cv2.boxFilter(tb, -1, (WB, HB), normalize=True, borderType=cv2.BORDER_CONSTANT)
    y0, y1 = HB//2, ny-(HB-HB//2); x0, x1 = WB//2, nx-(WB-WB//2)
    Cv, Sv, Tv = C[y0:y1, x0:x1], S[y0:y1, x0:x1], T[y0:y1, x0:x1]
    valid = (Tv >= 0.98) & (Cv >= 20)
    out = dict(win_quality_cd=np.nan, win_dense_cd=np.nan, win_typical_cd=np.nan, win_cluster_cd=np.nan, win_quality_score=np.nan, n_valid_windows=int(valid.sum()))
    if valid.sum() == 0:
        return out, None
    ms = np.where(valid, Sv/np.maximum(Cv, 1), -1)
    iq = np.unravel_index(np.argmax(ms), ms.shape)
    out['win_quality_cd'] = float(Cv[iq]/ROI_MM2); out['win_quality_score'] = float(ms[iq])
    out['win_dense_cd'] = float(Cv[valid].max()/ROI_MM2)
    out['win_typical_cd'] = float(np.median(Cv[valid])/ROI_MM2)
    cwin = None
    if centre is not None:
        cy = int(np.clip(round(centre[1]/BIN)-y0, 0, Cv.shape[0]-1)); cx = int(np.clip(round(centre[0]/BIN)-x0, 0, Cv.shape[1]-1))
        out['win_cluster_cd'] = float(Cv[cy, cx]/ROI_MM2)
        cwin = ((cx+x0-WB//2)*BIN, (cy+y0-HB//2)*BIN, (cx+x0-WB//2+WB)*BIN, (cy+y0-HB//2+HB)*BIN)
    return out, cwin


_CTX = {}


def row_features(r, mm, fam_pk, comp_pk):
    a_half = np.asarray(mm[r, ::2, ::2])
    tis, dt = tissue_mask(a_half)
    tissue_mm2 = float(tis.sum()*4*PX2MM2)
    cp = comp_pk[r]
    centre = None
    if cp is not None:
        q = cp[cp[:, 2] >= 0.35]
        if len(q) >= 10:
            centre = (float(np.median(q[:, 0])), float(np.median(q[:, 1])))
    rec = dict(tissue_mm2=tissue_mm2, black_frac_half=float(1-tis.mean()))
    for fam, pk in fam_pk.items():
        p = pk[r]
        for th in THRESHOLDS:
            tag = f'{fam}_t{int(th*100)}'
            if p is None:
                continue
            q = p[p[:, 2] >= th]
            xi = np.clip((q[:, 0]/2).astype(int), 0, tis.shape[1]-1); yi = np.clip((q[:, 1]/2).astype(int), 0, tis.shape[0]-1)
            intis = tis[yi, xi]
            q = q[intis]; d = dt[yi[intis], xi[intis]]
            rec[f'{tag}_n_tissue'] = float(len(q))
            rec[f'{tag}_dens_tissue'] = float(len(q)/tissue_mm2) if tissue_mm2 > 0.05 else np.nan
            vs = voronoi_stats(q[:, :2], d)
            for k, v in vs.items():
                rec[f'{tag}_{k}'] = v
            ws, cwin = window_stats(q[:, :2], q[:, 2], tis, centre)
            for k, v in ws.items():
                rec[f'{tag}_{k}'] = v
            if cwin is not None and th == 0.35:
                inw = (q[:, 0] >= cwin[0]) & (q[:, 0] < cwin[2]) & (q[:, 1] >= cwin[1]) & (q[:, 1] < cwin[3])
                vw = voronoi_stats(q[inw, :2], d[inw]) if inw.sum() >= 12 else dict(hex6_all=np.nan)
                rec[f'{tag}_hex6_cluster_window'] = vw['hex6_all']
                rec[f'{tag}_cwin'] = float(cwin[0])
                rec[f'{tag}_cwin_top'] = float(cwin[1])
    return rec


def _worker(item):
    n, r = item
    if 'mm' not in _CTX:
        _CTX['mm'] = np.memmap(RAW, dtype=np.uint8, mode='r', shape=(N_TOTAL, HGT, WID))
    return n, row_features(int(r), _CTX['mm'], _CTX['fam_pk'], _CTX['comp_pk'])


def features(a):
    """Per-row features with a bounded fork pool (PI rule: about 4 workers; never one long single process)."""
    OUT.mkdir(parents=True, exist_ok=True)
    base = H.load_base()
    fams_map = dict(kv.split('=', 1) for kv in a.fams) if a.fams else {k: str(v) for k, v in FAMS.items()}
    fam_pk = {k: load_peaks(Path(v))[0] for k, v in fams_map.items()}
    seen = np.zeros(N_TOTAL, bool)
    for k, v in fams_map.items():
        seen |= load_peaks(Path(v))[1]
    comp_pk, comp_seen = load_peaks(COMPACT); seen |= comp_seen
    rows = base['idx'] if not a.limit else base['idx'][:a.limit]
    _CTX.update(fam_pk=fam_pk, comp_pk=comp_pk)
    names = None; F = {}
    t0 = time.time(); done = 0
    with mp.get_context('fork').Pool(a.workers) as pool:
        for n, rec in pool.imap_unordered(_worker, list(enumerate(rows)), chunksize=16):
            if names is None:
                names = sorted(rec.keys()); F = {k: np.full(len(rows), np.nan) for k in names}
            for k in names:
                F[k][n] = rec.get(k, np.nan)
            done += 1
            if done % 500 == 0:
                log('features', n=done, of=len(rows), seconds=round(time.time()-t0, 1), workers=a.workers)
    tag = ('_'+a.tag if a.tag else '')+(f'_limit{a.limit}' if a.limit else '')
    np.savez_compressed(OUT/f'features{tag}.npz', idx=rows, seen=seen[rows], names=np.array(names), fams=np.array(list(fams_map)), **{k: F[k] for k in names})
    log('features done', n=len(rows), seconds=round(time.time()-t0, 1), workers=a.workers)
    if a.limit:
        for k in names:
            v = F[k]; print(k, 'finite %d/%d' % (np.isfinite(v).sum(), len(v)), 'median %.4g' % np.nanmedian(v) if np.isfinite(v).any() else '')


def lofo_strat(y, base, fold, ok, f, strata, t=0):
    """LOFO log-linear calibration (a, b) on the fitting folds; blend weight w chosen per stratum on the fitting
    folds; held fold scored per stratum.  Returns per-fold and per-stratum deltas (CD pp) plus mean pp."""
    ok = ok & (y[:, t] > 0)
    lf = np.where(np.isfinite(f) & (f > 0), np.log(np.where(f > 0, f, 1.0)), np.nan)
    ly = np.log(y[:, t]); lb = np.log(base[:, t]); have = np.isfinite(lf)
    pred = base[:, t].copy()
    out = {}
    for h in range(5):
        fit = ok & (fold != h) & have; held = ok & (fold == h)
        X = np.c_[np.ones(fit.sum()), lf[fit]]
        coef, *_ = np.linalg.lstsq(X, ly[fit], rcond=None)
        est = coef[0]+coef[1]*lf
        ws = {}
        for s in np.unique(strata[ok]):
            fs = fit & (strata == s)
            if fs.sum() < 50:
                ws[int(s)] = 0.0; continue
            scores = [P.mape(np.exp((1-w)*lb[fs]+w*est[fs]), y[fs, t]) for w in P.W_GRID]
            ws[int(s)] = float(P.W_GRID[int(np.argmin(scores))])
            hs = held & have & (strata == s)
            pred[hs] = np.exp((1-ws[int(s)])*lb[hs]+ws[int(s)]*est[hs])
        e_est = est[held & have]-ly[held & have]; e_b = lb[held & have]-ly[held & have]
        out[str(h)] = dict(a=float(coef[0]), b=float(coef[1]), w=ws, n=int(held.sum()),
                           delta_pp=float(P.mape(pred[held], y[held, t])-P.mape(base[held, t], y[held, t])),
                           est_mape=float(P.mape(np.exp(est[held & have]), y[held & have, t])),
                           err_corr=float(np.corrcoef(e_est, e_b)[0, 1]) if (held & have).sum() > 10 else np.nan,
                           per_stratum={STRATA_NAMES[int(s)]: float(P.mape(pred[held & (strata == s)], y[held & (strata == s), t])-P.mape(base[held & (strata == s), t], y[held & (strata == s), t])) for s in np.unique(strata[held])})
    d = np.array([out[str(h)]['delta_pp'] for h in range(5)]); n = np.array([out[str(h)]['n'] for h in range(5)])
    dev = ok & (fold != 4)
    out['overall'] = dict(delta_pp=float((d*n).sum()/n.sum()), delta_mean_pp=float((d*n).sum()/n.sum()/3), folds_improving=int((d < 0).sum()),
                          dev_folds0to3_pp=float(P.mape(pred[dev], y[dev, t])-P.mape(base[dev, t], y[dev, t])),
                          fold4_pp=out['4']['delta_pp'],
                          per_stratum_pp={STRATA_NAMES[int(s)]: float(P.mape(pred[ok & (strata == s)], y[ok & (strata == s), t])-P.mape(base[ok & (strata == s), t], y[ok & (strata == s), t])) for s in np.unique(strata[ok])},
                          err_corr_mean=float(np.nanmean([out[str(h)]['err_corr'] for h in range(5)])),
                          est_mape=float(np.mean([out[str(h)]['est_mape'] for h in range(5)])))
    return out, pred


def evaluate(a):
    base = H.load_base(); y, fold, B, idx, ids = base['y'], base['fold'], base['B'], base['idx'], base['ids']
    tag = ('_'+a.tag if a.tag else '')
    with np.load(OUT/f'features{tag}.npz', allow_pickle=True) as z:
        assert np.array_equal(z['idx'], idx)
        names = list(z['names']); F = {k: z[k] for k in names}; seen = z['seen']
        fams = [str(f) for f in z['fams']] if 'fams' in z.files else list(FAMS)
    ok = ~seen
    st = H.image_stats(base); framed = (st['black_frac'] > 0.01).astype(int)
    f0 = fams[0]; mname = f'mean{len(fams)}'

    def fam_mean(suffix, th):
        A = np.stack([F[f'{f}_t{th}_{suffix}'] for f in fams])
        A = np.where(A > 0, np.log(np.where(A > 0, A, 1.0)), np.nan)
        with np.errstate(all='ignore'):
            return np.exp(np.nanmean(A, 0))
    cand = {}
    for th in (35, 50):
        for suf in ('dens_tissue', 'vor_cd', 'vor_cd_hex', 'nn_cd', 'win_quality_cd', 'win_dense_cd', 'win_typical_cd', 'win_cluster_cd'):
            cand[f'{mname}_t{th}_{suf}'] = fam_mean(suf, th)
            cand[f'{f0}_t{th}_{suf}'] = F[f'{f0}_t{th}_{suf}']
    res = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), n_ok=int(ok.sum()), n_framed=int((framed[ok] == 1).sum()), base_B=H.official(H.ape(B, y), ok), cd={}, hex={})
    for name, f in cand.items():
        r = P.lofo_blend(y, B, fold, ok, np.where(np.isfinite(f) & (f > 0), f, np.nan), t=0)
        s, pred = lofo_strat(y, B, fold, ok, f, framed, t=0)
        res['cd'][name] = dict(plain=dict(delta_mean_pp=r['overall']['delta_mean_pp'], folds_improving=r['overall']['folds_improving'], est_cd_mape=r['overall']['est_cd_mape'], err_corr=r['overall']['err_corr_mean'], w=[r[str(h)]['w'] for h in range(5)]),
                               strat=s['overall'], strat_w=[s[str(h)]['w'] for h in range(5)], per_fold_pp=[s[str(h)]['delta_pp'] for h in range(5)])
        print(name, 'plain %.4f mean pp (impr %d, est %.2f, r_err %.2f) | strat %.4f mean pp, framed %.4f unframed %.4f CD pp, dev %.4f f4 %.4f' % (
            r['overall']['delta_mean_pp'], r['overall']['folds_improving'], r['overall']['est_cd_mape'], r['overall']['err_corr_mean'], s['overall']['delta_mean_pp'],
            s['overall']['per_stratum_pp'].get('framed', np.nan), s['overall']['per_stratum_pp'].get('unframed', np.nan), s['overall']['dev_folds0to3_pp'], s['overall']['fold4_pp']), flush=True)
    # HEX variants (D2)
    hexc = {f'hex6_all_{mname}': fam_mean('hex6_all', 35), f'hex6_interior_{mname}': fam_mean('hex6_interior', 35), f'hex6_cluster_window_{mname}': fam_mean('hex6_cluster_window', 35),
            f'hex6_all_{f0}': F[f'{f0}_t35_hex6_all'], f'hex6_interior_{f0}': F[f'{f0}_t35_hex6_interior']}
    for name, f in hexc.items():
        r = P.lofo_blend(y, B, fold, ok, np.where(np.isfinite(f) & (f > 0), f, np.nan), t=2)
        s, pred = lofo_strat(y, B, fold, ok, f, framed, t=2)
        res['hex'][name] = dict(plain=dict(delta_mean_pp=r['overall']['delta_mean_pp'], folds_improving=r['overall']['folds_improving'], w=[r[str(h)]['w'] for h in range(5)]), strat=s['overall'], per_fold_pp=[s[str(h)]['delta_pp'] for h in range(5)])
        print(name, 'HEX plain %.4f mean pp (impr %d) | strat %.4f, framed %.4f unframed %.4f HEX pp, dev %.4f f4 %.4f' % (
            r['overall']['delta_mean_pp'], r['overall']['folds_improving'], s['overall']['delta_mean_pp'], s['overall']['per_stratum_pp'].get('framed', np.nan), s['overall']['per_stratum_pp'].get('unframed', np.nan), s['overall']['dev_folds0to3_pp'], s['overall']['fold4_pp']), flush=True)
    # bootstrap of the top-3 CD candidates (stratified)
    top = sorted(res['cd'], key=lambda k: res['cd'][k]['strat']['delta_mean_pp'])[:3]
    g = H.donors(ids); e0 = H.ape(B, y)
    for name in top:
        s, pred = lofo_strat(y, B, fold, ok, cand[name], framed, t=0)
        pb = B.copy(); pb[:, 0] = pred; e1 = H.ape(pb, y)
        bs = dict(all=H.paired_bootstrap(e1[ok], e0[ok], g[ok]))
        for sv, sn in STRATA_NAMES.items():
            m = ok & (framed == sv); bs[sn] = H.paired_bootstrap(e1[m], e0[m], g[m])
        res['cd'][name]['bootstrap'] = bs
        print('bootstrap', name, json.dumps(bs), flush=True)
    # window-selection check on the 25 annotated images (overlap of the cluster window with the expert ROI box)
    try:
        ann = json.loads((Path('datasets/clear_ec_official/green_eval/annotations.json')).read_text())
        pos = {str(i): n for n, i in enumerate(ids)}
        ious = {}
        for k, e in ann.items():
            if k in pos and np.isfinite(F[f'{f0}_t35_cwin'][pos[k]]):
                n = pos[k]; x0, y0 = F[f'{f0}_t35_cwin'][n], F[f'{f0}_t35_cwin_top'][n]
                bx = e['roi_box_mha']
                ix = max(0, min(x0+ROI_W, bx[2])-max(x0, bx[0])); iy = max(0, min(y0+ROI_H, bx[3])-max(y0, bx[1]))
                inter = ix*iy; ious[k] = float(inter/(2*ROI_W*ROI_H-inter))
        res['cluster_window_vs_expert_roi_iou_25'] = dict(median=float(np.median(list(ious.values()))) if ious else None, values=ious, note='20 of the 25 trained the compact detector that sets the window centre; indicative only')
    except Exception as ex:
        res['cluster_window_vs_expert_roi_iou_25'] = str(ex)
    res['fams'] = fams
    (OUT/f'result{tag}.json').write_text(json.dumps(res, indent=1, default=float))
    log('evaluate done')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['features', 'evaluate']); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--fams', nargs='*', default=None, help='name=dir overrides of the detector families'); ap.add_argument('--tag', default='')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args(); {'features': features, 'evaluate': evaluate}[a.command](a)
