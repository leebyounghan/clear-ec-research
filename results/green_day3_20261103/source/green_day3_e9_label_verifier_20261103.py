"""Green day 3 / E9: label-verifier feasibility (CPU).  See e9_label_verifier/protocol.json."""
import json, sys, time
import numpy as np
from pathlib import Path
from scipy.spatial import Voronoi
sys.path.insert(0, 'experiments/green_day3_20261103/source'); sys.path.insert(0, 'experiments/green_track_20261028/source'); sys.path.insert(0, 'experiments/eval_harness_20261031/source'); sys.path.insert(0, 'experiments/green_day2_20261101/source')
import cpu_pool
import eval_oof_20261031 as H
import cd_estimators_20261028 as P
import green_day2_cd_features_20261101 as D
ROOT = Path('experiments/green_day3_20261103'); OUT = ROOT/'e9_label_verifier'; OUT.mkdir(parents=True, exist_ok=True)
GT = Path('experiments/green_track_20261028'); N, HGT, WID = 8954, 972, 1296
RW, RH, STEP = 538, 408, 16
XS = np.arange(0, WID-RW+1, STEP); YS = np.arange(0, HGT-RH+1, STEP)         # 48 x 36 windows
SOURCES = {'pre_s46': GT/'external_srep/preonly', 'sg2_s49': GT/'sg2_specialist_s49'}
_C = {}


def window_sums(mask_or_weights):
    """Sums over all 538x408 windows at stride 16 from a full-res (972 x 1296) array via an integral image."""
    I = np.zeros((HGT+1, WID+1), np.float64); I[1:, 1:] = np.cumsum(np.cumsum(mask_or_weights, 0), 1)
    y0 = YS[:, None]; x0 = XS[None, :]
    return I[y0+RH, x0+RW]-I[y0, x0+RW]-I[y0+RH, x0]+I[y0, x0]


def per_image(r):
    out = {}
    mm = _C.setdefault('mm', np.memmap(D.RAW, dtype=np.uint8, mode='r', shape=(N, HGT, WID)))
    tis, _ = D.tissue_mask(np.asarray(mm[r, ::2, ::2]))
    tissue_full = np.repeat(np.repeat(tis.astype(np.float32), 2, 0), 2, 1)[:HGT, :WID]
    tfrac = window_sums(tissue_full)/(RW*RH)
    for src, pk in _C['pk'].items():
        p = pk[r]
        if p is None or len(p) < 20:
            out[src] = None; continue
        q = p[p[:, 2] >= 0.35, :2].astype(np.float64)
        if len(q) < 20:
            out[src] = None; continue
        cnt_img = np.zeros((HGT, WID), np.float32); xi = np.clip(q[:, 0].astype(int), 0, WID-1); yi = np.clip(q[:, 1].astype(int), 0, HGT-1); np.add.at(cnt_img, (yi, xi), 1)
        cnt = window_sums(cnt_img)
        bnd = np.zeros((HGT, WID), np.float32); six = np.zeros((HGT, WID), np.float32)
        try:
            vor = Voronoi(q)
            for i, ri in enumerate(vor.point_region):
                reg = vor.regions[ri]
                if len(reg) == 0 or -1 in reg: continue
                v = vor.vertices[reg]
                if (v[:, 0] < 0).any() or (v[:, 1] < 0).any() or (v[:, 0] > WID-1).any() or (v[:, 1] > HGT-1).any(): continue
                bnd[yi[i], xi[i]] += 1
                if len(reg) == 6: six[yi[i], xi[i]] += 1
        except Exception:
            out[src] = None; continue
        nb = window_sums(bnd); ns = window_sums(six)
        with np.errstate(all='ignore'):
            hexw = np.where(nb >= 12, ns/np.maximum(nb, 1), np.nan)
        valid = (tfrac >= 0.98) & (nb >= 12)
        out[src] = dict(cnt=cnt.astype(np.float32), hexw=hexw.astype(np.float32), valid=valid, n_det=int(len(q)))
    return r, out


def main():
    base = H.load_base(); y_b, ids = base['y'], base['ids']
    yr = np.zeros((N, 3)); yr[base['idx']] = y_b
    st = H.image_stats(base); black = np.zeros(N); black[base['idx']] = st['black_frac']; framed = black > 0.01
    year = np.full(N, -1); year[base['idx']] = [H.year_of(i) for i in ids]
    with np.load('experiments/green_day2_20261101/d2b_active_gate/active_family_features.npz') as z:
        seen = z['seen']
    _C['pk'] = {s: D.load_peaks(d)[0] for s, d in SOURCES.items()}
    t0 = time.time(); w = cpu_pool.n_workers(); print('windows per image', len(XS)*len(YS), 'workers', w, flush=True)
    res_rows = dict(cpu_pool.pmap(per_image, range(N), chunksize=16, workers=w, ordered=False)); print('per-image pass', round(time.time()-t0), 's', flush=True)
    area_mm2 = RW*RH*(1000/1296/1000)**2
    ok_lab = (yr[:, 0] > 0) & (yr[:, 2] > 0)
    s_cd = float(np.std(np.log(yr[ok_lab, 0]))); s_hex = float(np.std(np.log(yr[ok_lab, 2])))
    ann = json.loads(Path('datasets/clear_ec_official/green_eval/annotations.json').read_text()); binding = json.loads(Path('experiments/green_point_probe_20261023/data_binding.json').read_text()); train_ids = set(binding['train_ids'])
    rowof = {str(i): int(r) for i, r in zip(ids, base['idx'])}
    report = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), s_cd=s_cd, s_hex=s_hex, windows=int(len(XS)*len(YS)), workers=w, sources={})
    rng = np.random.default_rng(46)
    for src in SOURCES:
        rows = [r for r in range(N) if res_rows[r].get(src) is not None and ok_lab[r]]
        # global k: median over images of CD_label / median valid-window density
        dens = []
        for r in rows:
            d = res_rows[r][src]; v = d['valid']
            if v.sum() >= 1: dens.append(yr[r, 0]/np.median(d['cnt'][v]/area_mm2))
        k = float(np.median(dens)); print(src, 'global k', round(k, 3), 'images', len(rows), flush=True)
        def Dmap(r, cd, hx):
            d = res_rows[r][src]; cd_est = k*d['cnt']/area_mm2
            with np.errstate(all='ignore'):
                Dv = np.abs(np.log(cd_est/cd))/s_cd+np.abs(np.log(d['hexw']/hx))/s_hex
            Dv = np.where(d['valid'] & np.isfinite(Dv), Dv, np.inf); return Dv
        # report 1: the 25 annotated images
        r1 = {}
        for name, e in ann.items():
            r = rowof.get(name)
            if r is None or res_rows[r].get(src) is None: continue
            bx = e['roi_box_mha']; Dv = Dmap(r, yr[r, 0], yr[r, 2])
            if not np.isfinite(Dv).any(): continue
            iy, ix = np.unravel_index(np.argmin(Dv), Dv.shape); wx, wy = XS[ix], YS[iy]
            inter = max(0, min(wx+RW, bx[2])-max(wx, bx[0]))*max(0, min(wy+RH, bx[3])-max(wy, bx[1])); iou = inter/(RW*RH+(bx[2]-bx[0])*(bx[3]-bx[1])-inter)
            cdist = float(np.hypot(wx+RW/2-(bx[0]+bx[2])/2, wy+RH/2-(bx[1]+bx[3])/2))
            jx = int(np.argmin(np.abs(XS-bx[0]))); jy = int(np.argmin(np.abs(YS-bx[1]))); d_true = float(Dv[jy, jx]); finite = np.isfinite(Dv)
            rank = int((Dv[finite] < d_true).sum())+1 if np.isfinite(d_true) else None
            r1[name] = dict(val=name not in train_ids, iou=float(iou), centre_dist_px=cdist, rank_true=rank, n_valid=int(finite.sum()), d_true=d_true, d_best=float(Dv[iy, ix]), n_det=res_rows[r][src]['n_det'])
        def agg(sel):
            if not sel: return None
            return dict(n=len(sel), iou_median=float(np.median([v['iou'] for v in sel])), iou_mean=float(np.mean([v['iou'] for v in sel])), centre_dist_median_px=float(np.median([v['centre_dist_px'] for v in sel])),
                        rank_true_median=float(np.median([v['rank_true'] for v in sel if v['rank_true']])), rank_true_pct_median=float(np.median([v['rank_true']/v['n_valid'] for v in sel if v['rank_true']])), d_true_minus_best_median=float(np.median([v['d_true']-v['d_best'] for v in sel if np.isfinite(v['d_true'])])))
        rep1 = dict(val5=agg([v for v in r1.values() if v['val']]), train20=agg([v for v in r1.values() if not v['val']]), per_image=r1)
        # report 2: own labels vs nulls within year x style
        groups = {}
        for r in rows: groups.setdefault((int(year[r]), bool(framed[r])), []).append(r)
        own = np.full(N, np.nan); beat_all = np.zeros(N, bool); pct = np.full(N, np.nan); ndet = np.full(N, np.nan)
        for r in rows:
            Dv = Dmap(r, yr[r, 0], yr[r, 2]); m = float(Dv.min())
            if not np.isfinite(m): continue
            g = [x for x in groups[(int(year[r]), bool(framed[r]))] if x != r]
            if len(g) < 5: continue
            draws = rng.choice(g, 20, replace=len(g) < 20); nulls = np.array([Dmap(r, yr[x, 0], yr[x, 2]).min() for x in draws])
            own[r] = m; beat_all[r] = bool(m < np.nanmin(nulls)); pct[r] = float((nulls <= m).mean()); ndet[r] = res_rows[r][src]['n_det']
        done = np.isfinite(own) & ~seen
        terc = np.digitize(ndet, np.nanquantile(ndet[done], [1/3, 2/3]))
        rep2 = dict(n=int(done.sum()), beat_all_20_fraction=float(beat_all[done].mean()), chance=1/21, own_percentile_median=float(np.nanmedian(pct[done])),
                    by_style={'framed': float(beat_all[done & framed].mean()), 'unframed': float(beat_all[done & ~framed].mean())},
                    by_n_det_tercile={f't{t}': float(beat_all[done & (terc == t)].mean()) for t in range(3)},
                    by_year={int(yv): float(beat_all[done & (year == yv)].mean()) for yv in np.unique(year[done])})
        report['sources'][src] = dict(k=k, report1=rep1, report2=rep2)
        print(src, 'report1 val5', rep1['val5'], '\n', src, 'report1 train20', {kk: vv for kk, vv in (rep1['train20'] or {}).items()}, '\n', src, 'report2', {kk: vv for kk, vv in rep2.items()}, flush=True)
    (OUT/'result.json').write_text(json.dumps(report, indent=1, default=float)); print('done', round(time.time()-t0), 's', flush=True)


if __name__ == '__main__':
    main()
