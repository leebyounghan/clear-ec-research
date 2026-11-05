"""E1 evaluation (CPU): per-image features from CPSAM cells, LOFO on B (HEX, CV), member test in the active ten,
centroid F1 vs expert points on the 25 annotated images."""
import json, sys, time, glob
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, 'experiments/eval_harness_20261031/source'); sys.path.insert(0, 'experiments/green_track_20261028/source'); sys.path.insert(0, 'experiments/green_day2_20261101/source'); sys.path.insert(0, 'experiments/green_day3_20261103/source')
import eval_oof_20261031 as H
import cd_estimators_20261028 as P
import green_t4_numerics_20261101 as M
import green_day2_cd_features_20261101 as D
V = M.V
ROOT = Path('experiments/green_day3_20261103'); OUT = ROOT/'e1_cpsam'; N = 8954
box = np.load(ROOT/'e3_boxpool/cluster_boxes.npz')['box']
cells = {}
for p in sorted(glob.glob(str(OUT/'v2'/'rows_*.npz'))):
    with np.load(p, allow_pickle=True) as z:
        for r, st in zip(z['rows'], z['stats']):
            cells[int(r)] = st
print('images with CPSAM cells:', len(cells), flush=True)
KEYS = ('n_cells', 'hex6_cpsam', 'cv_seg_win', 'cv_seg_all', 'hexadj_win', 'hexadj_all', 'hexorg_win', 'hexorg_all', 'sec')
import cpu_pool


def _cv(v):
    if len(v) < 8: return np.nan
    lo, hi = np.percentile(v, (5, 95)); t = v[(v >= lo) & (v <= hi)]; return t.std()/t.mean() if len(t) >= 8 else np.nan


def _row_feats(r):
    st = cells[r]
    if st.get('n', 0) < 12:
        return r, None
    c = st['centroid']; a = st['area'].astype(float); s_ = st['sides']; b = st['border']; h = st['hexagonality'].astype(float)
    x0, y0, x1, y1 = box[r]; inb = (c[:, 0] >= x0) & (c[:, 0] < x1) & (c[:, 1] >= y0) & (c[:, 1] < y1)
    interior = ~b
    win = inb & interior if (inb & interior).sum() >= 12 else interior
    out = dict(n_cells=st['n'], sec=st.get('seconds', np.nan), hex6_cpsam=V.voronoi_feats(c.astype(np.float64))['hex6_all'],
               cv_seg_win=_cv(a[win]), cv_seg_all=_cv(a[interior]), hexadj_win=(s_[win] == 6).mean(), hexadj_all=(s_[interior] == 6).mean() if interior.sum() else np.nan,
               hexorg_win=h[win].mean(), hexorg_all=h[interior].mean() if interior.sum() else np.nan)
    return r, out


F = {k: np.full(N, np.nan) for k in KEYS}
t0 = time.time(); w = cpu_pool.n_workers()
for r, out in cpu_pool.pmap(_row_feats, sorted(cells), chunksize=32, workers=w):
    if out is not None:
        for k, v in out.items():
            F[k][r] = v
print('features from CPSAM cells:', int(np.isfinite(F['hex6_cpsam']).sum()), 'rows, workers', w, round(time.time()-t0), 's', flush=True)
np.savez_compressed(OUT/'features.npz', **F)
base = H.load_base(); y, fold, B, idx = base['y'], base['fold'], base['B'], base['idx']
yr, Br, fr = np.zeros((N, 3)), np.zeros((N, 3)), np.zeros(N, int); yr[idx] = y; Br[idx] = B; fr[idx] = fold
st_ = H.image_stats(base); framed = np.zeros(N, int); framed[idx] = (st_['black_frac'] > 0.01)
with np.load('experiments/green_day2_20261101/d2b_active_gate/active_family_features.npz') as z:
    seen = z['seen']; hx = {k.replace('hex6_', ''): z[k] for k in z.files if k.startswith('hex6_')}
ok = ~seen
D.STRATA_NAMES = {0: 'unframed', 1: 'framed'}
res = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), n_images=len(cells), cells_per_image=dict(q10=float(np.nanquantile(F['n_cells'], .1)), q50=float(np.nanquantile(F['n_cells'], .5)), q90=float(np.nanquantile(F['n_cells'], .9))), hex={}, cv={}, member={})
def run(name, f, t):
    r = P.lofo_blend(yr, Br, fr, ok, np.where(np.isfinite(f) & (f > 0), f, np.nan), t=t); s, _ = D.lofo_strat(yr, Br, fr, ok, f, framed, t=t)
    d = dict(delta_mean_pp=r['overall']['delta_cd_pp']/3, folds_improving=r['overall']['folds_improving'], w=[r[str(h)]['w'] for h in range(5)], est_mape=r['overall']['est_cd_mape'], err_corr=r['overall']['err_corr_mean'], strat_mean_pp=s['overall']['delta_mean_pp'], per_stratum_pp=s['overall']['per_stratum_pp'], dev_pp=s['overall']['dev_folds0to3_pp'], fold4_pp=s['overall']['fold4_pp'])
    print('%-14s %s plain %.4f mean pp (impr %d, est %.2f, r_err %.2f, w %s) | strat %.4f framed %.4f unframed %.4f | dev %.4f f4 %.4f' % (name, 'HEX' if t == 2 else 'CV', d['delta_mean_pp'], d['folds_improving'], d['est_mape'], d['err_corr'], d['w'], d['strat_mean_pp'], d['per_stratum_pp'].get('framed', np.nan), d['per_stratum_pp'].get('unframed', np.nan), d['dev_pp'], d['fold4_pp']), flush=True)
    return d
for k in ('hex6_cpsam', 'hexadj_win', 'hexadj_all', 'hexorg_win', 'hexorg_all'):
    res['hex'][k] = run(k, F[k], 2)
for k in ('cv_seg_win', 'cv_seg_all'):
    res['cv'][k] = run(k, F[k], 1)
ACTIVE = M.ACTIVE_ORDER
def blend(fams, fe):
    A = np.log(np.stack([fe[k] for k in fams]))
    with np.errstate(all='ignore'):
        f = np.exp(np.nanmean(A, 0))
    r = P.lofo_blend(yr, Br, fr, ok, f, t=2); o = r['overall']; return dict(delta_mean_pp=o['delta_cd_pp']/3, folds_improving=o['folds_improving'], per_fold_mean_pp=[r[str(h)]['delta_cd_pp']/3 for h in range(5)], w=[r[str(h)]['w'] for h in range(5)])
fe = dict(hx); fe['hex6_cpsam'] = F['hex6_cpsam']; fe['hexadj_win'] = F['hexadj_win']
for name, fams in {'control_ten': ACTIVE, 'ten_plus_hex6_cpsam': ACTIVE+['hex6_cpsam'], 'swap_pre_for_hex6_cpsam': [('hex6_cpsam' if f == 'sg2pre_s46' else f) for f in ACTIVE], 'ten_plus_hexadj_win': ACTIVE+['hexadj_win'], 'single_hex6_cpsam': ['hex6_cpsam']}.items():
    res['member'][name] = blend(fams, fe); v = res['member'][name]
    print('member %-26s %.4f mean pp impr %d per fold %s w %s' % (name, v['delta_mean_pp'], v['folds_improving'], [round(x, 4) for x in v['per_fold_mean_pp']], v['w']), flush=True)
c = res['member']['control_ten']['delta_mean_pp']
res['gates'] = {k: dict(gain=v['delta_mean_pp']-c, evidence=bool(v['delta_mean_pp'] <= c-0.005 and v['per_fold_mean_pp'][4] < 0 and v['folds_improving'] >= 4), adoption=bool(v['delta_mean_pp'] <= c-0.02 and v['per_fold_mean_pp'][4] < 0 and v['folds_improving'] >= 4)) for k, v in res['member'].items() if k != 'control_ten'}
res['gates'].update({f'cv_{k}': dict(evidence=bool(v['delta_mean_pp'] <= -0.005 and v['folds_improving'] >= 4), adoption=bool(v['delta_mean_pp'] <= -0.02 and v['folds_improving'] >= 4)) for k, v in res['cv'].items()})
# centroid F1 vs expert points (25 annotated; 12-px match, inside the expert ROI box)
ann = json.loads(Path('datasets/clear_ec_official/green_eval/annotations.json').read_text()); binding = json.loads(Path('experiments/green_point_probe_20261023/data_binding.json').read_text())
pos = {str(i): n for n, i in enumerate(base['ids'])}; rowof = {str(i): int(r) for i, r in zip(base['ids'], base['idx'])}
f1s = {}
for name, e in ann.items():
    r = rowof.get(name)
    if r is None or r not in cells or cells[r].get('n', 0) == 0:
        continue
    gt = np.asarray(e['markers_mha'], float); bx = e['roi_box_mha']; c = cells[r]['centroid'].astype(float)
    inb = (c[:, 0] >= bx[0]) & (c[:, 0] <= bx[2]) & (c[:, 1] >= bx[1]) & (c[:, 1] <= bx[3]); c = c[inb]
    hull_in = np.ones(len(gt), bool)
    if len(c) == 0:
        f1s[name] = dict(f1=0.0, n_pred=0, n_gt=int(len(gt))); continue
    d, j = cKDTree(c).query(gt); used = set(); tp = 0
    for dd, jj in sorted(zip(d, j)):
        if dd <= 12 and jj not in used: used.add(jj); tp += 1
    prec = tp/len(c); rec = tp/len(gt); f1s[name] = dict(f1=2*prec*rec/(prec+rec) if prec+rec else 0.0, precision=prec, recall=rec, n_pred=int(len(c)), n_gt=int(len(gt)), val=name not in set(binding['train_ids']))
res['centroid_f1_25'] = dict(mean_all=float(np.mean([v['f1'] for v in f1s.values()])), mean_val5=float(np.mean([v['f1'] for v in f1s.values() if v.get('val')])), per_image=f1s)
res['seconds_per_image_median'] = float(np.nanmedian(np.diff(np.sort(F['sec'][np.isfinite(F['sec'])])))) if np.isfinite(F['sec']).sum() > 10 else None
print('centroid F1 vs expert points (12 px, in ROI): all-25 %.3f, val-5 %.3f' % (res['centroid_f1_25']['mean_all'], res['centroid_f1_25']['mean_val5']), flush=True)
print('GATES', json.dumps({k: v for k, v in res['gates'].items() if v.get('evidence') or v.get('adoption')}), flush=True)
(OUT/'result.json').write_text(json.dumps(res, indent=1, default=float)); print('written')
