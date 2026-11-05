"""Green day 3 / E6 (pre-registered 2026-11-03 16:45 UTC): CV from E2-predicted per-cell areas at precision-oriented points.
  points   (CPU) per image, inside the cluster window: (a) CPSAM v2 centroids (non-border); (b) consensus = CPSAM centroids
           that have a detector point (pre-only s46 or compact sg2_s46, fold-clean OOF, >= 0.35) within 0.5 x the CPSAM
           NN spacing.  Parameters inherited from E4/E5 (match radius, window, 10 % trim); nothing tuned on the 8954.
  infer    (GPU) the E2 CNN on 96x96 patches at those points -> predicted log area per point.
  evaluate (CPU) CV of predicted areas (10 % trimmed, and plain) per image; LOFO on B for CV, stratified; E5 gates.
"""
import argparse, glob, json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, 'experiments/green_day3_20261103/source'); sys.path.insert(0, 'experiments/green_track_20261028/source'); sys.path.insert(0, 'experiments/eval_harness_20261031/source'); sys.path.insert(0, 'experiments/green_day2_20261101/source')
import cpu_pool
ROOT = Path('experiments/green_day3_20261103'); OUT = ROOT/'e6_pred_area_cv'; OUT.mkdir(parents=True, exist_ok=True)
GT = Path('experiments/green_track_20261028'); N = 8954
SETS = ('cpsam_win', 'consensus_win')
_C = {}


def _points_task(r):
    st = _C['cp'].get(r)
    if st is None or st.get('n', 0) < 12:
        return r, {s: np.zeros((0, 3), np.float32) for s in SETS}
    c = st['centroid'].astype(np.float64)[~st['border']]
    x0, y0, x1, y1 = _C['box'][r]; inb = (c[:, 0] >= x0) & (c[:, 0] < x1) & (c[:, 1] >= y0) & (c[:, 1] < y1); cw = c[inb]
    out = {'cpsam_win': np.c_[cw, np.ones(len(cw))].astype(np.float32)}
    if len(cw) >= 4:
        sp = float(np.median(cKDTree(cw).query(cw, k=2)[0][:, 1])); det = [p for p in (_C['pre'][r], _C['comp'][r]) if p is not None and len(p)]
        if det:
            d = np.vstack(det); dist, _ = cKDTree(d).query(cw); keep = dist <= 0.5*sp
            out['consensus_win'] = np.c_[cw[keep], np.ones(int(keep.sum()))].astype(np.float32)
        else:
            out['consensus_win'] = np.zeros((0, 3), np.float32)
    else:
        out['consensus_win'] = np.zeros((0, 3), np.float32)
    return r, out


def points(a):
    import green_day2_cd_features_20261101 as D
    cp = {}
    for p in glob.glob(str(ROOT/'e1_cpsam/v2/rows_*.npz')):
        with np.load(p, allow_pickle=True) as z:
            for r, st in zip(z['rows'], z['stats']): cp[int(r)] = st
    pre, _ = D.load_peaks(GT/'external_srep/preonly'); comp, _ = D.load_peaks(GT/'sg2_specialist')
    _C.update(cp=cp, box=np.load(ROOT/'e3_boxpool/cluster_boxes.npz')['box'], pre=[None if p is None else p[p[:, 2] >= 0.35, :2] for p in pre], comp=[None if p is None else p[p[:, 2] >= 0.35, :2] for p in comp])
    t0 = time.time(); res = dict(cpu_pool.pmap(_points_task, range(N), chunksize=64))
    for s in SETS:
        rows = np.arange(N); pts = [res[r][s] for r in rows]; off = np.cumsum([0]+[len(p) for p in pts])
        np.savez_compressed(OUT/f'points_{s}.npz', rows=rows, offsets=off, points=np.concatenate(pts))
        n = np.array([len(p) for p in pts]); print(s, 'points per image q10/q50/q90', np.percentile(n, [10, 50, 90]), 'images with >= 12:', int((n >= 12).sum()), flush=True)
    print('points done', round(time.time()-t0), 's', flush=True)


def infer(a):
    import torch
    from torch.utils.data import DataLoader
    import green_day3_e2_cellhead_20261103 as E2
    ck = torch.load(E2.OUT/'model.pt', map_location='cpu', weights_only=False); net = E2.build_net().cuda(); net.load_state_dict(ck['model']); net.eval()
    for s in a.sets:
        with np.load(OUT/f'points_{s}.npz') as z:
            rows, off, pts = z['rows'], z['offsets'], z['points']
        sel = [i for i in range(len(rows)) if a.rows_from <= int(rows[i]) < a.rows_to]
        items = [(int(rows[i]), pts[off[i]:off[i+1]]) for i in sel]
        out = OUT/f'pred_{s}_{a.rows_from:05d}.npz'
        if out.exists():
            continue
        dl = DataLoader(E2.PatchSet(items), batch_size=1, shuffle=False, num_workers=3, collate_fn=lambda b: b[0], prefetch_factor=4)
        R_, O_, P_, preds = [], [0], [], []; t0 = time.time()
        with torch.no_grad():
            for k, (r, p, X) in enumerate(dl):
                o = torch.cat([net(torch.from_numpy(X[i:i+2048]).cuda().float().unsqueeze(1)/255) for i in range(0, len(X), 2048)]).cpu().numpy() if len(X) else np.zeros((0, 2), np.float32)
                R_.append(r); P_.append(p); preds.append(o.astype(np.float32)); O_.append(O_[-1]+len(p))
                if k % 1000 == 0:
                    print(json.dumps(dict(msg='infer', set=s, image=k, of=len(items), seconds=round(time.time()-t0, 1))), flush=True)
        np.savez_compressed(out, rows=np.array(R_), offsets=np.array(O_), points=np.concatenate(P_), pred=np.concatenate(preds), m_la=ck['m_la'], s_la=ck['s_la'])
        print(json.dumps(dict(msg='infer done', set=s, images=len(items), seconds=round(time.time()-t0, 1))), flush=True)


def evaluate(a):
    import eval_oof_20261031 as H
    import cd_estimators_20261028 as P
    import green_day2_cd_features_20261101 as D
    y, tta4, baseA, fold, seen, keys, F, aa = P.load(); ok = ~seen
    with np.load(P.E/'recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True) as h:
        o = np.argsort(h['indices']); B = h['B_hybrid_w0p5'][o]
    base = H.load_base(); st = H.image_stats(base); framed = np.zeros(N, int); framed[base['idx']] = (st['black_frac'] > 0.01)
    D.STRATA_NAMES = {0: 'unframed', 1: 'framed'}
    feats = {}
    for s in SETS:
        cv10 = np.full(N, np.nan); cv0 = np.full(N, np.nan); ncell = np.full(N, np.nan)
        for p in sorted(glob.glob(str(OUT/f'pred_{s}_*.npz'))):
            with np.load(p) as z:
                rows, off, pred, m_la, s_la = z['rows'], z['offsets'], z['pred'], float(z['m_la']), float(z['s_la'])
            for i, r in enumerate(rows):
                o_ = pred[off[i]:off[i+1]]
                if len(o_) < 12:
                    continue
                area = np.exp(o_[:, 1]*s_la+m_la); lo, hi = np.quantile(area, (0.1, 0.9)); t = area[(area >= lo) & (area <= hi)]
                cv10[int(r)] = t.std()/t.mean() if len(t) >= 8 else np.nan; cv0[int(r)] = area.std()/area.mean(); ncell[int(r)] = len(o_)
        feats[s] = dict(cv_pred_trim10=cv10, cv_pred_plain=cv0, n=ncell)
    np.savez_compressed(OUT/'features.npz', **{f'{s}_{k}': v for s, d in feats.items() for k, v in d.items()})
    res = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), cv={})
    for s, d in feats.items():
        for k in ('cv_pred_trim10', 'cv_pred_plain'):
            f = d[k]; r = P.lofo_blend(y, B, fold, ok, np.where(np.isfinite(f) & (f > 0), f, np.nan), t=1); sr, _ = D.lofo_strat(y, B, fold, ok, f, framed, t=1)
            rec = dict(delta_mean_pp=r['overall']['delta_cd_pp']/3, delta_cv_pp=r['overall']['delta_cd_pp'], folds_improving=r['overall']['folds_improving'], w=[r[str(h)]['w'] for h in range(5)], est_mape=r['overall']['est_cd_mape'], err_corr=r['overall']['err_corr_mean'],
                       per_fold_mean_pp=[r[str(h)]['delta_cd_pp']/3 for h in range(5)], strat_mean_pp=sr['overall']['delta_mean_pp'], per_stratum_pp=sr['overall']['per_stratum_pp'], n_with_estimate=int(np.isfinite(f[ok]).sum()), cells_median=float(np.nanmedian(d['n'])))
            rec['evidence'] = bool(rec['delta_mean_pp'] <= -0.005 and rec['folds_improving'] >= 4); rec['adoption'] = bool(rec['delta_mean_pp'] <= -0.02 and rec['folds_improving'] >= 4 and rec['per_fold_mean_pp'][4] < 0)
            res['cv'][f'{s}_{k}'] = rec
            print('%-30s CV %.4f mean pp (CV %.4f, impr %d, est %.2f, r_err %.2f, w %s, n %d, cells %.0f) | strat %.4f framed %.4f unframed %.4f %s' % (f'{s}_{k}', rec['delta_mean_pp'], rec['delta_cv_pp'], rec['folds_improving'], rec['est_mape'], rec['err_corr'], rec['w'], rec['n_with_estimate'], rec['cells_median'], rec['strat_mean_pp'], rec['per_stratum_pp'].get('framed', np.nan), rec['per_stratum_pp'].get('unframed', np.nan), 'EVIDENCE' if rec['evidence'] else ''), flush=True)
    tr = json.loads((ROOT/'e2_cellhead'/'train_result.json').read_text())
    res['e2_area_head_sanity'] = dict(srep_val_area_r=tr['history'][-1]['val_area_r'], clear_ec_val5_area_r=tr['val5_mean_area_r'], clear_ec_val5_six_acc=tr['val5_mean_six_acc'], passed_clear_ec=bool(tr['val5_mean_area_r'] >= 0.5))
    (OUT/'result.json').write_text(json.dumps(res, indent=1, default=float)); print('E2 area head sanity', res['e2_area_head_sanity'], flush=True); print('done', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['points', 'infer', 'evaluate']); ap.add_argument('--sets', nargs='*', default=list(SETS)); ap.add_argument('--rows-from', type=int, default=0); ap.add_argument('--rows-to', type=int, default=N)
    a = ap.parse_args(); {'points': points, 'infer': infer, 'evaluate': evaluate}[a.command](a)
