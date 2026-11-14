"""Image-level inputs of the HEX chain for unseen images (HEX lane, 2026-11-10).

Inputs, for M test images:
  --images  uint8 memmap (M, 972, 1296), the same raw frame representation as datasets/clear_ec_fullframe_20260904/img_full_raw.dat
  --peaks   root with <detector>/fold0/fold0_sg2_peaks.npz (per-fold peak-file format, indices 0..M-1), produced by the full-data
            detectors (source/det_peaks_image.py); detector names as in feat_fuse.SETS (pre3 = external_srep/preonly,
            external_srep/s47_preonly, external_srep/s48_preonly; all16 adds the 9 sg2 specialists and 4 frozen heads)
  --cd      npz with key cd (M): the hybrid-B CD of each image (0.5 P1 + 0.5 recipe A, as in submission 1), not the new CD
Outputs, written to --out:
  features.npz  img (M x 4: mean, contrast, black_frac, focus), prior_h6c (M), prior_all (M x k, names), ridge_k3 (M x 5),
                ridge_fall16_h6_s15 (M), seg (M x 17; only with --seg)
  g_pre46.npz, g_pre47.npz, g_pre48.npz, g_fall16.npz   cell graphs (inputs of the graph networks; gnn_hex_infer.py)
Every number is computed by the same per-frame functions that built the training features (feat_prior.one, graph_prep_lane.one,
fuse_peaks.one, feat_ridge.one, seg_run.feats); this script only swaps their module globals for the test images.
  --pseudo K  parity check: K training frames are written as a pseudo-test set (their stored OOF peaks, hybrid-B CD and raw frames)
              and every output is compared with the stored training features.
usage: hex_test_features.py --images X.dat --n M --peaks ROOT --cd CD.npz --out DIR [--seg] [--workers 4]
       hex_test_features.py --pseudo 40 --out DIR [--seg]"""
import os, sys, argparse, json
import numpy as np
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'; ROOT = '/workspace/clear-ec-research'
GREEN = f'{ROOT}/experiments/green_track_20261028'
sys.path.insert(0, f'{LANE}/source'); sys.path.insert(0, f'{ROOT}/experiments/meta_ceiling_20261103/source')
H_, W_ = 972, 1296


def load_peaks(root, det):
    out = {}
    for f in range(5):
        p = f'{root}/{det}/fold{f}/fold{f}_sg2_peaks.npz'
        if not os.path.exists(p): continue
        z = np.load(p, allow_pickle=True); pk, of, ix = z['peaks'], z['offsets'], z['indices']
        for k, i in enumerate(ix): out[int(i)] = pk[of[k]:of[k + 1]]
    return out


def thr(d, t): return {k: v[v[:, 2] >= t] for k, v in d.items()}


def image_stats(MM, M):
    """eval_harness_20261031/source/eval_oof_20261031.py image_stats, per image"""
    out = np.zeros((M, 4))
    for n in range(M):
        a = np.asarray(MM[n, ::2, ::2], dtype=np.float32); lap = a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
        out[n] = [a.mean(), a.std(), np.mean(a < 5), lap.var()]
    return out


def pool_map(fn, M, workers):
    from multiprocessing import get_context
    with get_context('fork').Pool(workers) as pool: return dict(pool.imap_unordered(fn, range(M), chunksize=4))


def graphs(GP, PK, M, node_t, workers):
    """graph_prep_lane.main's packing, on PK = {image: raw peaks}"""
    GP.PK = PK; GP.NODE_T = node_t; res = pool_map(GP.one, M, workers)
    X, E_, wid, offs, eoff, XY = [], [], [], [0], [0], []; n = 0; m = 0
    for r in range(M):
        for w, (f, e, xy) in enumerate(res.get(r) or []):
            XY.append(xy); X.append(f); E_.append(e + n); wid.append(np.full(len(f), w, np.int8)); n += len(f); m += len(e)
        offs.append(n); eoff.append(m)
    cat = lambda a, s: np.concatenate(a) if a else np.zeros(s, np.float32)
    return dict(xy=cat(XY, (0, 2)), X=cat(X, (0, 9)), E=cat(E_, (0, 2)).astype(np.int32), wid=cat(wid, (0,)).astype(np.int8), offs=np.array(offs), eoff=np.array(eoff), feats=np.array(GP.FEATS))


_FR = [None]


def _ridge_one(r):
    o = _FR[0].one(r); return o[0], o[2]


def ridge(FR, G, MM, M, workers):
    """feat_ridge.main's per-frame features (h6_s15, h6_s25, h6_s35, h6_dark, dark_mean) on graph G"""
    FR.XY, FR.OFFS, FR.WID, FR.EOFF, FR.MM = G['xy'], G['offs'], G['wid'], G['eoff'], MM; _FR[0] = FR; res = pool_map(_ridge_one, M, workers)
    out = np.full((M, 5), np.nan)
    for r, fr in res.items():
        if fr is not None: w = fr[:, 0]; out[r] = (fr[:, 1:] * w[:, None]).sum(0) / w.sum()
    return out


def build(MM, M, peaks_root, cd, outd, workers=4, do_seg=False, peaks=None):
    """peaks_root: per-fold peak files on disk, or peaks = {detector: [M arrays (x, y, conf)]} in memory (then peaks_root is unused)"""
    os.makedirs(outd, exist_ok=True)
    import feat_fuse as FF
    sys.argv = ['feat_prior.py', '/dev/null', 'pre3']; import feat_prior as FP
    sys.argv = ['fuse_peaks.py', '/dev/null', 'all16']; import fuse_peaks as FU
    sys.argv = ['graph_prep_lane.py', 'unused.npz', '', '3']; import graph_prep_lane as GP
    FF.T = 0.2; pre3, all16 = FF.SETS['pre3'], FF.SETS['all16']
    raw = {d: ({i: np.asarray(q, np.float32) for i, q in enumerate(peaks[d])} if peaks is not None else load_peaks(peaks_root, d)) for d in (all16 if peaks is None or all(d in peaks for d in all16) else pre3)}
    if any(d not in raw for d in all16): print('only the pre3 peaks given: ridge_fall16 is skipped', flush=True)
    out = dict(img=image_stats(MM, M))
    # prior: count-matched hex6 from the CD spacing prior (pre3 consensus, legible windows from pre-only s46 >= 0.5)
    FP.PK = [thr(raw[d], 0.2) for d in pre3]; FP.PRE = FP.PK[0]; FP.CDF = np.asarray(cd, float); res = pool_map(FP.one, M, workers)
    keys = sorted({k for o in res.values() if o for k in o}); PA = np.full((M, len(keys)), np.nan)
    for r, o in res.items():
        if o: PA[r] = [o.get(k, np.nan) for k in keys]
    out['prior_all'] = PA; out['prior_names'] = np.array(keys); out['prior_h6c'] = PA[:, keys.index('cm1.0_h6c')] if 'cm1.0_h6c' in keys else np.full(M, np.nan)
    # cell graphs of the three pre-only detectors (graph networks; pre46 also feeds ridge evidence k3)
    G = {}
    for nm, d in (('pre46', 'external_srep/preonly'), ('pre47', 'external_srep/s47_preonly'), ('pre48', 'external_srep/s48_preonly')):
        G[nm] = graphs(GP, raw[d], M, 0.35, workers); np.savez(f'{outd}/g_{nm}.npz', **G[nm])
    # 16-detector consensus peaks -> graph (node threshold 0.25)
    if all(d in raw for d in all16):
        FU.PK = [thr(raw[d], 0.2) for d in all16]; fused = pool_map(FU.one, M, workers); fused = {r: v for r, v in fused.items() if v is not None}
        G['fall16'] = graphs(GP, fused, M, 0.25, workers); np.savez(f'{outd}/g_fall16.npz', **G['fall16'])
    # ridge evidence on the pre46 graph (k3) and the fall16 graph
    sys.argv = ['feat_ridge.py', f'{outd}/g_pre46.npz', 'test']; import feat_ridge as FR
    out['ridge_k3'] = ridge(FR, G['pre46'], MM, M, workers)
    if 'fall16' in G: out['ridge_fall16_h6_s15'] = ridge(FR, G['fall16'], MM, M, workers)[:, 0]
    if do_seg: out['seg'] = seg_features(MM, M)
    np.savez_compressed(f'{outd}/features.npz', **out); return out, G


def build_arrays(imgs, peaks, cd, outd, workers=4, do_seg=False):
    """in-memory entry: imgs uint8 (M, 972, 1296) raw frames, peaks {detector name as in feat_fuse.SETS: list of M arrays
    (x, y, confidence), single-view sg2 peaks, threshold 0.2, NMS 12 px}, cd (M) hybrid-B CD = 0.5 A_cd + 0.5 P1_cd. Returns
    (features dict, graphs dict) and writes them to outd."""
    imgs = np.asarray(imgs, np.uint8); return build(imgs, len(imgs), None, np.asarray(cd, float), outd, workers, do_seg, peaks=peaks)


def seg_features(MM, M):
    """seg_run.py: loose Cellpose-SAM (flow 0.8, cellprob -2, CLAHE), then seg_run.feats on each mask"""
    os.environ.update(CP_FLOW='0.8', CP_PROB='-2', CP_CLAHE='1'); sys.argv = ['seg_run.py', 'seg_loose']; import seg_run as SR
    from cellpose import models
    model = models.CellposeModel(gpu=True, pretrained_model='cpsam_models/cpsam_v2'); F = np.full((M, len(SR.NAMES)), np.nan)
    for s in range(0, M, 4):
        imgs = [np.asarray(MM[i]) for i in range(s, min(M, s + 4))]
        if SR.CLAHE is not None: imgs = [SR.CLAHE.apply(im) for im in imgs]
        masks = model.eval(imgs, batch_size=8, **SR.KW)[0]
        for k, Mk in enumerate(masks): F[s + k] = SR.feats(np.asarray(Mk, np.int32))
    return F


def pseudo(K, outd, workers, do_seg):
    """K training frames as a pseudo-test set; compares every output with the stored training features"""
    sys.argv = ['feat_fuse.py', '/dev/null', 'all16']; import feat_fuse as FF
    import hexlib as L
    rng = np.random.default_rng(0); frames = np.sort(rng.choice(L.idx.astype(int), K, replace=False)); os.makedirs(outd, exist_ok=True)
    big = np.memmap(f'{ROOT}/datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, H_, W_))
    MM = np.memmap(f'{outd}/pseudo_images.dat', dtype=np.uint8, mode='w+', shape=(K, H_, W_)); MM[:] = big[frames]; MM.flush()
    proot = f'{outd}/peaks'
    for d in FF.SETS['all16']:
        src = load_peaks(GREEN, d); pk = [src.get(int(f), np.zeros((0, 3), np.float32)) for f in frames]
        os.makedirs(f'{proot}/{d}/fold0', exist_ok=True)
        np.savez(f'{proot}/{d}/fold0/fold0_sg2_peaks.npz', peaks=np.concatenate(pk), offsets=np.r_[0, np.cumsum([len(q) for q in pk])], indices=np.arange(K))
    _b = L.BP._b; cdf = np.full(8954, np.nan); cdf[_b['idx']] = _b['B'][:, 0]; cd = cdf[frames]
    out, G = build(MM, K, proot, cd, outd, workers, do_seg)
    rep = {}
    def cmp(name, a, b):
        a, b = np.asarray(a, float), np.asarray(b, float); both = np.isfinite(a) & np.isfinite(b); nan_mismatch = int((np.isfinite(a) != np.isfinite(b)).sum())
        rep[name] = dict(max_abs_diff=float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0, nan_mismatch=nan_mismatch, n=int(both.sum()))
    st = np.load(f'{ROOT}/experiments/eval_harness_20261031/image_stats.npz'); pos = {int(i): k for k, i in enumerate(st['idx'])}; rows = [pos[int(f)] for f in frames]
    cmp('img', out['img'], np.c_[st['mean'], st['contrast'], st['black_frac'], st['focus']][rows])
    zp = np.load(f'{LANE}/feats/prior_pre3.npz'); nm = [str(q) for q in zp['names']]; cmp('prior_h6c', out['prior_h6c'], zp['F'][frames][:, nm.index('cm1.0_h6c')])
    cmp('prior_all', out['prior_all'], zp['F'][frames][:, [nm.index(k) for k in out['prior_names']]])
    cmp('ridge_k3', out['ridge_k3'], np.load(f'{LANE}/feats/ridge_k3.npz')['F'][frames][:, :5])
    zf = np.load(f'{LANE}/feats/ridge_fall16.npz'); cmp('ridge_fall16_h6_s15', out['ridge_fall16_h6_s15'], zf['F'][frames][:, [str(q) for q in zf['names']].index('h6_s15')])
    for nm_, ref in (('pre46', f'{LANE}/feats/g_k3.npz'), ('pre47', f'{ROOT}/experiments/meta_ceiling_20261103/sweep/cell_graphs_pre47.npz'), ('pre48', f'{ROOT}/experiments/meta_ceiling_20261103/sweep/cell_graphs_pre48.npz'), ('fall16', f'{LANE}/feats/g_fall16.npz')):
        R = np.load(ref); g = G[nm_]; d = 0.0; nn_ = 0
        for k, f in enumerate(frames):
            a, b = g['X'][g['offs'][k]:g['offs'][k + 1]], R['X'][R['offs'][f]:R['offs'][f + 1], :g['X'].shape[1]]
            if a.shape != b.shape: nn_ += 1; continue
            if len(a): d = max(d, float(np.abs(a - b).max()))
        rep[f'graph_{nm_}'] = dict(max_abs_diff_nodes=d, frames_with_other_node_count=nn_)
    if do_seg:
        zs = np.load(f'{LANE}/preds/seg_all.npz'); ps = {int(f): k for k, f in enumerate(zs['frames'])}; cmp('seg', out['seg'], zs['F'][[ps[int(f)] for f in frames]])
    json.dump(dict(frames=frames.tolist(), report=rep), open(f'{outd}/parity.json', 'w'), indent=1); print(json.dumps(rep, indent=1), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--images'); ap.add_argument('--n', type=int); ap.add_argument('--peaks'); ap.add_argument('--cd'); ap.add_argument('--out', required=True)
    ap.add_argument('--seg', action='store_true'); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--pseudo', type=int, default=0); a = ap.parse_args()
    if a.pseudo: pseudo(a.pseudo, a.out, a.workers, a.seg)
    else:
        MM = np.memmap(a.images, dtype=np.uint8, mode='r', shape=(a.n, H_, W_)); build(MM, a.n, a.peaks, np.load(a.cd)['cd'], a.out, a.workers, a.seg); print('done', a.out, flush=True)
