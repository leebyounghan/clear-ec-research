"""Sanity check before the full Vigueras run (HEX lane, 2026-11-10, coordinator request): 200 random frames (seed 0), CNN-Edge +
CNN-ROI on the CPU (keeps GPU 1 free for the full-data CPSAM build), the same fixed resize as the full run (972 x 1296 ->
720 x 960, INTER_AREA, 1.042 um/px vs the authors' 1.038), the vig_post cell route, then cell counts and density against the
CPSAM route (preds/seg_all.npz) on the same frames. Writes runs/vig_sanity/{maps.npz, feats.npz} and logs/vig_sanity.json.
usage: vig_sanity.py [n_frames]"""
import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np, cv2, torch
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
sys.path.insert(0, LANE + '/source')
import vig_net as V
import vig_post as P

if __name__ == '__main__':
    torch.set_num_threads(4); n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out = LANE + '/runs/vig_sanity'; os.makedirs(out, exist_ok=True)
    fr = np.sort(np.random.default_rng(0).choice(8954, n, replace=False))
    MM = np.memmap('/workspace/clear-ec-research/datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
    mp = out + '/maps.npz'
    if not os.path.exists(mp):
        We, Wr = V.load('tvst_edge', 'cpu'), V.load('tvst_roi', 'cpu'); E8, R8 = [], []; t0 = time.time()
        for s in range(0, n, 4):
            X = np.stack([cv2.resize(np.asarray(MM[i]), (960, 720), interpolation=cv2.INTER_AREA) for i in fr[s:s + 4]]).astype(np.float32) / 255
            E = V.run(We, X, batch=4); Rr = V.run(Wr, E, batch=4)
            E8.append(np.round(E * 255).astype(np.uint8)); R8.append(np.round(Rr * 255).astype(np.uint8))
            print(f'{s + len(X)} frames, {(time.time() - t0) / (s + len(X)):.1f} s/frame', flush=True)
        np.savez_compressed(mp, frames=fr, edge=np.concatenate(E8), roi=np.concatenate(R8))
    fp = out + '/feats.npz'
    if not os.path.exists(fp):
        torch.set_num_threads(1)
        from multiprocessing import Pool
        z = np.load(mp); args = [(z['edge'][k], z['roi'][k]) for k in range(n)]
        with Pool(4) as pool: F = np.array(pool.map(P.seg_and_feats, args), float)
        np.savez_compressed(fp, frames=fr, F=F, names=np.array(P.NAMES_X))
    F = np.load(fp)['F']; nm = list(P.NAMES_X)
    zc = np.load(LANE + '/preds/seg_all.npz'); pos = {int(f): r for r, f in enumerate(zc['frames'])}; C = np.array([zc['F'][pos[int(f)]] if int(f) in pos else np.full(zc['F'].shape[1], np.nan) for f in fr], float); cn = [str(q) for q in zc['names']]
    rep = dict(frames=int(n), resize='972x1296 -> 720x960 INTER_AREA (1.042 um/px; authors 1.038), intensity/255, one fixed choice')
    for col in ('n_cells', 'n_int', 'cd_mm2', 'h6', 'area_cv'):
        a, b = F[:, nm.index(col)], C[:, cn.index(col)]; m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        rep[col] = dict(both_finite=int(m.sum()), pearson=float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else None,
                        log_pearson=float(np.corrcoef(np.log(a[m]), np.log(b[m]))[0, 1]) if m.sum() > 2 else None,
                        median_ratio_vig_over_cpsam=float(np.median(a[m] / b[m])) if m.any() else None, vig_median=float(np.nanmedian(a)), cpsam_median=float(np.nanmedian(b)))
    rep['usable_n_int_ge_50'] = dict(vig=int((np.nan_to_num(F[:, nm.index('n_int')]) >= 50).sum()), cpsam=int((np.nan_to_num(C[:, cn.index('n_int')]) >= 50).sum()))
    rep['vig_fc_median'] = float(np.nanmedian(F[:, nm.index('vig_fc')])); rep['vig_roi_frac_median'] = float(np.nanmedian(F[:, nm.index('vig_roi_frac')]))
    print(json.dumps(rep, indent=1), flush=True); json.dump(rep, open(LANE + '/logs/vig_sanity.json', 'w'), indent=1)
