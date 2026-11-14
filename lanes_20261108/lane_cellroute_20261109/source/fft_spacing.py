"""Fourier estimate of endothelial cell density, label-free (deploy lane, 2026-11-11, D-CDSIMPLE-001 part a). One fixed form, nothing fitted to labels.
Per frame (raw 972 x 1296 uint8, native resolution): subtract the mean, 2-D Hann window, power spectrum (rfft2), radial mean in bins of 1/N cycles/px.
Background: log10 power vs log10 frequency fitted by one quadratic over 0.006-0.25 cycles/px; pass 2 refits without [f1/1.25, 1.25 f1] around the pass-1
ring peak f1. Ring peak f0 = argmax of the residual inside the plausible band (500-5000 cells/mm2), refined by a parabola on the three bins.
Density: hexagonal lattice first ring |k| = 2 / (sqrt(3) d)  ->  D = (sqrt(3)/2) (f0 / s)^2 * 1e6 cells/mm2, s = 1000/1296 um/px.
Confidence: peak height above the background (log10 power, 'db') and its robust z (MAD of the band residual outside the peak window).
Scales: the full frame; and 256 px tiles at <= 128 px stride (7 x 10 = 70 tiles, >= 50 % overlap), each zero-padded to 512. Tile summaries: median D over
tiles with a positive peak, median D over the tiles whose centre lies in the 3 legible 380 x 300 windows of deploy_hexkonan.windows (stored seeds3_tta4
detector peaks, as deploy_hexkonan.features picks them), median tile z.
usage: fft_spacing.py probe | fft_spacing.py all WORKERS | fft_spacing.py eval"""
import os, sys, json, re, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
R = '/workspace/clear-ec-research/experiments/'; L = R + 'lane_cellroute_20261109/'; OUTNPZ = L + 'feats/fft_spacing_cd.npz'
H, W = 972, 1296; S_UM = 1000.0 / 1296; F_FIT = (0.006, 0.25); D_BAND = (500.0, 5000.0); EXCL = 1.25; TILE, PAD = 256, 512
f_of_D = lambda D: np.sqrt(np.asarray(D, float) * 1e-6 / (np.sqrt(3) / 2)) * S_UM
D_of_f = lambda f: (np.sqrt(3) / 2) * (np.asarray(f, float) / S_UM) ** 2 * 1e6
BAND = (float(f_of_D(D_BAND[0])), float(f_of_D(D_BAND[1])))
_RAD = {}


def radial(a, pad=None):
    """log10 radial mean power of the Hann-windowed, mean-removed array a (optionally zero-padded to pad x pad); returns (f, logp)."""
    a = np.asarray(a, np.float32); a = a - a.mean(); h, w = a.shape
    a = a * np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    shp = (pad, pad) if pad else (h, w); P = np.abs(np.fft.rfft2(a, s=shp)) ** 2
    key = shp
    if key not in _RAD:
        fy = np.fft.fftfreq(shp[0])[:, None]; fx = np.fft.rfftfreq(shp[1])[None, :]; df = 1.0 / max(shp)
        rb = np.rint(np.sqrt(fx ** 2 + fy ** 2) / df).astype(np.int64).ravel(); n = np.bincount(rb); _RAD[key] = (rb, n, df)
    rb, n, df = _RAD[key]; s = np.bincount(rb, weights=P.ravel()); k = np.arange(len(s)); ok = n > 0
    return k[ok] * df, np.log10(s[ok] / n[ok] + 1e-30)


def ring(f, lp):
    """one fixed background removal and band peak; returns (f0, db, z) or NaNs."""
    fit = (f >= F_FIT[0]) & (f <= F_FIT[1]); band = (f >= BAND[0]) & (f <= BAND[1])
    if fit.sum() < 10 or band.sum() < 3: return np.nan, np.nan, np.nan, None
    lf = np.log10(np.maximum(f, 1e-9))
    def resid(mask):
        c = np.polyfit(lf[mask], lp[mask], 2); return lp - np.polyval(c, lf)
    r1 = resid(fit); i1 = np.flatnonzero(band)[np.argmax(r1[band])]; f1 = f[i1]
    r = resid(fit & ~((f >= f1 / EXCL) & (f <= f1 * EXCL)))
    bi = np.flatnonzero(band); i = bi[np.argmax(r[bi])]
    if 0 < i < len(f) - 1:
        den = r[i - 1] - 2 * r[i] + r[i + 1]; d = 0.5 * (r[i - 1] - r[i + 1]) / den if den != 0 else 0.0; d = float(np.clip(d, -0.5, 0.5))
    else: d = 0.0
    f0 = f[i] + d * (f[1] - f[0]); out = band & ~((f >= f0 / EXCL) & (f <= f0 * EXCL)); rs = r[out]
    sd = 1.4826 * np.median(np.abs(rs - np.median(rs))) if rs.size > 3 else np.nan
    return float(f0), float(r[i]), float((r[i] - np.median(rs)) / sd) if sd and np.isfinite(sd) and sd > 0 else np.nan, r


def tile_origins():
    ny = int(np.ceil((H - TILE) / 128)) + 1; nx = int(np.ceil((W - TILE) / 128)) + 1
    return [(int(round(y)), int(round(x))) for y in np.linspace(0, H - TILE, ny) for x in np.linspace(0, W - TILE, nx)]


def legible_windows(pk3):
    sys.path.insert(0, L + 'source'); import deploy_hexkonan as DK
    if pk3 is None or len(pk3) < 20: return []
    pk3 = np.asarray(pk3, float); hi = pk3[pk3[:, 2] >= 0.5, :2]; mid = pk3[pk3[:, 2] >= 0.35, :2]; ref = hi if len(hi) > 20 else mid
    return DK.windows(ref, 380, 300, k=3) if len(ref) >= 10 else []


def frame(img, pk3):
    f, lp = radial(img); F0, DB, Z, _ = ring(f, lp)
    wins = legible_windows(pk3); T = []
    for y0, x0 in tile_origins():
        tf, tl = radial(img[y0:y0 + TILE, x0:x0 + TILE], PAD); t0, tdb, tz, _ = ring(tf, tl); cy, cx = y0 + TILE / 2, x0 + TILE / 2
        leg = any(wx <= cx < wx + 380 and wy <= cy < wy + 300 for wx, wy in wins); T.append((t0, tdb, tz, leg))
    T = np.array(T, float); pos = T[:, 1] > 0; leg = pos & (T[:, 3] > 0)
    med = lambda v: float(np.median(v)) if v.size else np.nan
    lsd = lambda v: float(np.std(np.log(v))) if v.size >= 3 else np.nan
    iqr = lambda v: float((np.percentile(v, 75) - np.percentile(v, 25)) / np.median(v)) if v.size >= 3 else np.nan
    return dict(D_full=float(D_of_f(F0)) if np.isfinite(F0) else np.nan, db_full=DB, z_full=Z, f_full=F0,
                D_tile_med=med(D_of_f(T[pos, 0])), D_leg_med=med(D_of_f(T[leg, 0])), z_tile_med=med(T[pos, 2]), db_tile_med=med(T[pos, 1]),
                n_tiles_pos=int(pos.sum()), n_tiles_leg=int(leg.sum()), n_windows=len(wins),
                tile_logsd=lsd(D_of_f(T[pos, 0])), tile_iqr_rel=iqr(D_of_f(T[pos, 0])), leg_logsd=lsd(D_of_f(T[leg, 0])))   # within-frame spread of the tile estimates


# ---------------- data access ----------------
MM = None; PK = None
def init():
    global MM, PK
    MM = np.memmap(R + '../datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, H, W))
    PK = {}
    for fo in range(5):
        z = np.load(R + f'lane_cv_20261108/out/tta_peaks/seeds3_tta4/fold{fo}/fold{fo}_sg2_peaks.npz', allow_pickle=True); pk, of, ix = z['peaks'], z['offsets'], z['indices']
        for k, d in enumerate(ix): PK[int(d)] = (pk, int(of[k]), int(of[k + 1]))
def pk3_of(d):
    if d not in PK: return None
    pk, a, b = PK[d]; return np.asarray(pk[a:b], float)
def work(d):
    try: return d, frame(np.asarray(MM[d]), pk3_of(d))
    except Exception as e: return d, dict(error=repr(e))


def rows_and_labels():
    zc = np.load(R + 'lane_cd_20261108/reads/freeze_cd_oof_20261111.npz', allow_pickle=True)
    ids = np.array([str(i).strip() for i in zc['ids']]); P2 = np.array([bool(re.match(r'^\d{4}-\d{4}', i)) for i in ids])
    return zc, ids, P2


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'probe':
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        sys.path.insert(0, L + 'inference'); init(); zc, ids, P2 = rows_and_labels(); idx = zc['idx'].astype(int); y = zc['y_cd']
        a = np.asarray(MM[idx[:0]]) if False else None
        foc = np.array([np.var(np.asarray(MM[d][::4, ::4], np.float32)[1:-1, 1:-1] * 4 - np.asarray(MM[d][::4, ::4], np.float32)[:-2, 1:-1] - np.asarray(MM[d][::4, ::4], np.float32)[2:, 1:-1]
                                - np.asarray(MM[d][::4, ::4], np.float32)[1:-1, :-2] - np.asarray(MM[d][::4, ::4], np.float32)[1:-1, 2:]) for d in idx[::7]])
        sub = np.arange(len(idx))[::7]; okk = y[sub] > 0; o = sub[okk][np.argsort(foc[okk])]
        pick = [o[int(q * (len(o) - 1))] for q in (0.99, 0.97, 0.95, 0.05, 0.03, 0.01)]
        fig, ax = plt.subplots(2, 6, figsize=(30, 9.5)); res = []
        for j, k in enumerate(pick):
            d = int(idx[k]); img = np.asarray(MM[d]); r = frame(img, pk3_of(d)); f, lp = radial(img); f0, db, z, resid = ring(f, lp)
            ax[0, j].imshow(img[::2, ::2], cmap='gray'); ax[0, j].set_axis_off()
            for wx, wy in legible_windows(pk3_of(d)): ax[0, j].add_patch(plt.Rectangle((wx / 2, wy / 2), 190, 150, fill=False, ec='yellow', lw=1.5))
            ax[0, j].set_title(f"{'sharp' if j < 3 else 'poor'} | {ids[k]} ({'P2' if P2[k] else 'P1'}) | label CD {y[k]:.0f}", fontsize=11)
            m = (f >= 0.008) & (f <= 0.12); ax[1, j].plot(f[m], resid[m], 'k-', lw=1)
            ax[1, j].axvspan(*BAND, color='0.9', zorder=0); ax[1, j].axvline(f0, color='r', lw=2, label=f'picked full frame: {D_of_f(f0):.0f}')
            if np.isfinite(r['D_tile_med']): ax[1, j].axvline(f_of_D(r['D_tile_med']), color='orange', ls=':', lw=2, label=f"tile median: {r['D_tile_med']:.0f}")
            ax[1, j].axvline(f_of_D(y[k]), color='g', ls='--', lw=1.5, label=f'label (display only): {y[k]:.0f}')
            ax[1, j].set_xlabel('spatial frequency (cycles / px); grey = band 500-5000 cells/mm2'); ax[1, j].set_ylabel('log10 power above the 1/f fit')
            ax[1, j].set_title(f"full-frame peak z {z:.1f}, height {db:.2f}; legible-tile median {r['D_leg_med']:.0f}", fontsize=10); ax[1, j].legend(fontsize=9, loc='upper right')
            res.append(dict(frame=d, id=ids[k], source='P2' if P2[k] else 'P1', label=float(y[k]), **r))
        fig.suptitle('Fourier ring estimate of cell density (D-CDSIMPLE-001 a): radial power spectrum after one fixed 1/f fit; top 3 sharp / bottom 3 poor by Laplacian variance', fontsize=14)
        fig.tight_layout(); fig.savefig(L + 'out/fft_spacing_probe.png', dpi=70); json.dump(res, open(L + 'out/fft_spacing_probe.json', 'w'), indent=1)
        print('SUMMARY', json.dumps(res), flush=True)
    elif mode == 'eval':
        sys.path.insert(0, R + 'meta_ceiling_20261103/source'); import best_pipeline_20261106 as BP
        zc, ids, P2 = rows_and_labels(); z = np.load(OUTNPZ, allow_pickle=True); assert (z['idx'] == zc['idx']).all() and (zc['idx'].astype(int) == BP.idx.astype(int)).all()
        y = zc['y_cd']; fold = zc['fold']; ok = y > 0
        mask = np.where(np.isfinite(zc['mask_cd']) & (zc['mask_cd'] > 0) & (np.nan_to_num(zc['mask_n_int']) >= 50), zc['mask_cd'], np.nan)      # cd_mm2, CD combine's usable rows
        det = np.load(R + 'error_case_analysis_20261106/tables/forensics_rows.npz')['det'][:, 0].astype(float)
        COLS = {'fft_full': z['D_full'], 'fft_tile_med': z['D_tile_med'], 'fft_legible_tile_med': z['D_leg_med'], 'cd_mm2 (mask, >=50 cells)': mask, 'detector density (forensics det col0)': det}
        def corr(v, rows):
            m = rows & ok & np.isfinite(v) & (v > 0); return dict(r=round(float(np.corrcoef(np.log(v[m]), np.log(y[m]))[0, 1]), 4) if m.sum() > 10 else None, n=int(m.sum()))
        def nested(v):
            P = np.full(len(y), np.nan); m = ok & np.isfinite(v) & (v > 0)
            for f in range(5):
                tr = m & (fold != f); b, a = np.polyfit(np.log(v[tr]), np.log(y[tr]), 1); te = m & (fold == f); P[te] = np.exp(a + b * np.log(v[te]))
            e = np.abs(P - y) / np.where(ok, y, 1) * 100
            return dict(mape=round(float(np.nanmean(e[m])), 3), mape_P1=round(float(np.nanmean(e[m & ~P2])), 3), mape_P2=round(float(np.nanmean(e[m & P2])), 3), rows=int(m.sum()),
                        coverage=round(float(m.sum() / ok.sum()), 4), folds=[round(float(np.nanmean(e[m & (fold == f)])), 3) for f in range(5)])
        allr = np.ones(len(y), bool); mrows = np.isfinite(mask)
        out = {k: dict(all=corr(v, allr), P1=corr(v, ~P2), P2=corr(v, P2), on_cd_mm2_rows=corr(v, mrows), nested_loglin=nested(v)) for k, v in COLS.items()}
        conf = z['z_full']; out['fft_full_by_confidence'] = {f'z_full >= {q}': corr(z['D_full'], conf >= q) for q in (3, 5, 8)}
        stack = zc['oof_cd']; rs = np.log(np.where(ok, y, 1.0)) - np.log(stack)      # frozen CD stack (D-CD-001) OOF residual, as computed (6.1741)
        fft = [k for k in COLS if k.startswith('fft')]; best = min(fft, key=lambda k: out[k]['nested_loglin']['mape'])
        def l1(v):
            P = np.full(len(y), np.nan); m = ok & np.isfinite(v) & (v > 0)
            for f in range(5):
                tr = m & (fold != f); b, a = np.polyfit(np.log(v[tr]), np.log(y[tr]), 1); te = m & (fold == f); P[te] = np.exp(a + b * np.log(v[te]))
            rm = np.log(np.where(ok, y, 1.0)) - np.log(P); dv = np.log(P) - np.log(stack); mm = m & np.isfinite(rm)
            return dict(residual_corr_with_stack=round(float(np.corrcoef(rm[mm], rs[mm])[0, 1]), 4), deviation_vs_stack_residual_r=round(float(np.corrcoef(dv[mm], rs[mm])[0, 1]), 4), rows=int(mm.sum()))
        out['L1'] = dict(best_variant=best, **{k: l1(COLS[k]) for k in fft + ['cd_mm2 (mask, >=50 cells)']})
        out['spread_vs_stack_abs_residual'] = {k: corr_abs for k, corr_abs in ((s, round(float(np.corrcoef(z[s][np.isfinite(z[s]) & ok], np.abs(rs)[np.isfinite(z[s]) & ok])[0, 1]), 4)) for s in ('tile_logsd', 'tile_iqr_rel', 'leg_logsd'))}
        out['stack_ref'] = dict(file='lane_cd_20261108/reads/freeze_cd_oof_20261111.npz oof_cd', mape=round(float(np.mean(np.abs(stack[ok] - y[ok]) / y[ok]) * 100), 4))
        out['note'] = 'r = Pearson on log values vs log CD label; nested_loglin = log y ~ a + b log column fitted on the other 4 folds, MAPE on the rows where the column exists; L1: residual_corr = r(log y - log calibrated column, log y - log stack), deviation r = r(log calibrated column - log stack, log y - log stack)'
        print('SUMMARY', json.dumps(out), flush=True); json.dump(out, open(L + 'out/fft_spacing_eval.json', 'w'), indent=1)
    elif mode == 'all':
        from multiprocessing import Pool
        zc, ids, P2 = rows_and_labels(); idx = zc['idx'].astype(int); t = time.time()
        with Pool(int(sys.argv[2]), initializer=init) as p: res = dict(p.imap_unordered(work, [int(d) for d in idx], chunksize=8))
        keys = ['D_full', 'db_full', 'z_full', 'f_full', 'D_tile_med', 'D_leg_med', 'z_tile_med', 'db_tile_med', 'n_tiles_pos', 'n_tiles_leg', 'n_windows', 'tile_logsd', 'tile_iqr_rel', 'leg_logsd']
        cols = {k: np.array([res[int(d)].get(k, np.nan) for d in idx], float) for k in keys}; errs = [int(d) for d in idx if 'error' in res[int(d)]]
        np.savez_compressed(OUTNPZ, idx=idx, ids=ids, source=np.where(P2, 'P2', 'P1'), note='label-free Fourier ring CD estimate, fft_spacing.py (fixed form; see docstring); cells/mm2', **cols)
        print('SUMMARY', json.dumps(dict(frames=len(idx), errors=len(errs), seconds=round(time.time() - t), finite={k: int(np.isfinite(v).sum()) for k, v in cols.items()})), flush=True)
