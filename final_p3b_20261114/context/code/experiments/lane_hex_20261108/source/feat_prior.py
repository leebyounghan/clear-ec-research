"""Spacing prior from predicted CD -> count-matched points -> hex6 (HEX lane, 2026-11-08). Per frame: expected cell area
A = 1e6 / CD_B um^2 (B = submission-1 OOF CD), spacing d = sqrt(2A/sqrt3). Candidates: stored peaks >= 0.2 of a detector set,
fused by linking within 0.4 d (votes = distinct detectors, score = summed confidence / D), greedy NMS at 0.55 d by score.
In each of the three legible windows (380x300, pre-only s46 peaks >= 0.5) keep the top rho * area / A candidates by score and
take hex6 of bounded cells (plus the spacing-filtered 'clean' hex6). rho in {0.85, 1.0, 1.15}. Also the plain 0.35-threshold
hex6 of the same fused set in the same windows, as the control. usage: feat_prior.py OUT SET"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
from multiprocessing import get_context
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import feat_fuse as FF
import legible_window_features_20261105 as LW
import ceiling_meta_cpu_20261103 as C
OUT = sys.argv[1]; SET = sys.argv[2]; DETS = FF.SETS[SET]; D = len(DETS); UM = 0.7716049; RHOS = (0.85, 1.0, 1.15)
CDF = None   # frame-indexed OOF CD of the training frames, loaded in main() only (11-10: nothing is read at import, so the inference path can import this module)
PK = []; PRE = {}


def nms(P, s, r):
    o = np.argsort(-s); keep = np.ones(len(P), bool); tr = cKDTree(P)
    for i in o:
        if not keep[i]: continue
        for j in tr.query_ball_point(P[i], r):
            if j != i and s[j] <= s[i]: keep[j] = False
    return keep


def one(fr):
    qs = [pk.get(fr) for pk in PK]
    if any(q is None for q in qs) or not np.isfinite(CDF[fr]): return fr, None
    A = 1e6 / CDF[fr] / UM ** 2; d = np.sqrt(2 * A / np.sqrt(3))
    P = np.concatenate([q[:, :2] for q in qs]).astype(np.float64); Cf = np.concatenate([q[:, 2] for q in qs]); DI = np.concatenate([np.full(len(q), k) for k, q in enumerate(qs)])
    if len(P) < 30: return fr, None
    pr = cKDTree(P).query_pairs(0.4 * d, output_type='ndarray'); n = len(P)
    Am = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n)) if len(pr) else coo_matrix((n, n)); nc, lab = connected_components(Am, directed=False)
    sw = np.bincount(lab, Cf, nc); X = np.c_[np.bincount(lab, Cf * P[:, 0], nc) / sw, np.bincount(lab, Cf * P[:, 1], nc) / sw]
    # score: best confidence per detector summed / D  (one vote per detector)
    key = lab * 64 + DI; o = np.lexsort((-Cf, key)); first = np.r_[True, key[o][1:] != key[o][:-1]]; sc = np.bincount(lab[o][first], Cf[o][first], nc) / D
    k = nms(X, sc, 0.55 * d); X, sc = X[k], sc[k]
    pre = PRE.get(fr); hi = pre[pre[:, 2] >= 0.5, :2].astype(np.float64) if pre is not None else None
    if hi is None or len(hi) < 10: return fr, None
    out = {'d_exp': d}; wins = LW.windows(hi, 380, 300, k=3); rows = {}
    for x0, y0 in wins:
        m = (X[:, 0] >= x0) & (X[:, 0] < x0 + 380) & (X[:, 1] >= y0) & (X[:, 1] < y0 + 300); Xw, sw_ = X[m], sc[m]; o = np.argsort(-sw_)
        nexp = 380 * 300 / A
        for rho in RHOS:
            kk = o[:int(round(rho * nexp))]; rows.setdefault(f'cm{rho}', []).append(FF.hexstats(Xw[kk]))
        rows.setdefault('t35', []).append(FF.hexstats(Xw[sw_ >= 0.35]))
        rows.setdefault('t50', []).append(FF.hexstats(Xw[sw_ >= 0.5]))
        out.setdefault('fill', []).append(float((sw_ >= 0.35).sum() / nexp))
    for nm, ws in rows.items():
        ws = np.array(ws, float)
        with np.errstate(all='ignore'):
            out[f'{nm}_h6'] = float(np.nanmean(ws[:, 0])) if np.isfinite(ws[:, 0]).any() else np.nan
            out[f'{nm}_h6c'] = float(np.nanmean(ws[:, 4])) if np.isfinite(ws[:, 4]).any() else np.nan
            out[f'{nm}_f5'] = float(np.nanmean(ws[:, 1])) if np.isfinite(ws[:, 1]).any() else np.nan
            out[f'{nm}_f7'] = float(np.nanmean(ws[:, 2])) if np.isfinite(ws[:, 2]).any() else np.nan
    out['fill'] = float(np.mean(out['fill']))
    return fr, out


def main():
    global PK, PRE, CDF
    if CDF is None:
        _b = C.E.load_base(); CDF = np.full(8954, np.nan); CDF[_b['idx']] = _b['B'][:, 0]
    FF.T = 0.2; PK = [FF.load(d) for d in DETS]; PRE = FF.load('external_srep/preonly') if DETS[0] != 'external_srep/preonly' else PK[0]
    PRE = {k: v for k, v in PRE.items()}
    print('dets', D, flush=True); res = {}
    with get_context('fork').Pool(int(os.environ.get('NW', '3'))) as pool:
        for k, (r, o) in enumerate(pool.imap_unordered(one, range(8954), chunksize=16)):
            res[r] = o
            if k % 2000 == 0: print(k, flush=True)
    keys = sorted({k for o in res.values() if o for k in o})
    F = np.full((8954, len(keys)), np.nan, np.float32)
    for r, o in res.items():
        if o:
            for j, k in enumerate(keys): F[r, j] = o.get(k, np.nan)
    np.savez_compressed(OUT, F=F, names=np.array(keys), dets=np.array(DETS)); print('done', F.shape, flush=True)


if __name__ == '__main__':
    main()
