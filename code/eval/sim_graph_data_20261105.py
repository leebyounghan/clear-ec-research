"""Simulated cell graphs with exact labels (2026-11-05). Geometry alone taught a network something image values could not, because geometry leaves nothing to memorise. So no images here: exact tessellations from the renderer's geometry code (power diagrams relaxed by Lloyd steps, drawn afresh over a wide range of order, CD log-normal around the label median, with a jittered-lattice family that reaches the highly regular mosaics where the real gain sits) plus a detector noise model measured on the 25 expert sets and tuned so graph statistics match real frames. No human annotation or challenge label enters a sample.
Noise model, per draw: a smooth quality field Q; hard misses growing as Q falls; confidence of a found cell from a Beta rising with Q; jitter sigma per frame; false peaks at 0.4-1.0 spacing from a true cell, with the measured confidence distribution, more where Q is low. Windows and graph features exactly as on real frames: three most legible hull-sized windows by peaks >= 0.5, nodes >= 0.35, Voronoi features. Target: log share of six Voronoi neighbours on the true centres, log area CV and log CD of the true cells in those windows (the markers' Voronoi hex6 matches device HEX within 1-2 % on clean sets).
Writes sweep/<OUT>: X, E, offs, eoff, T (frames x 3), frame id, draw parameters. usage: sim_graph_data_20261105.py OUT DRAWS FRAMES. Forked pool, clamp(16 - load1 - 1, 2, 12)."""
import json, os, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import cv2
import numpy as np
from scipy.spatial import Delaunay, Voronoi, cKDTree
cv2.setNumThreads(0)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import synth_endothelium_20261103 as S
import legible_window_features_20261105 as L
A = '/workspace/clear-ec-research/experiments/meta_ceiling_20261103'; FH, FW = 972, 1296; W, H = 380, 300; UM = S.UM_PER_PX
RELAX = (2, 12); JIT = (0.0, 0.14); SSD = (0.03, 0.40); LATTICE = 0.75; LJIT = (0.17, 0.33); LVAR = 0.5; LSSD = (0.08, 0.38)
FALSE_Q = np.array([0.20, 0.234, 0.268, 0.35, 0.469, 0.702, 0.837, 0.879, 0.92]); FALSE_P = np.array([0, .05, .10, .25, .50, .75, .90, .95, 1.0])
OFF_Q = np.array([0.35, 0.405, 0.466, 0.527, 0.606, 0.732, 0.892, 0.998, 1.15])
NOISE = dict(q_lo=0.0, q_hi=1.0, miss=0.85, miss_pow=1.5, a0=1.6, a1=15.0, b=1.4, fp_lo=0.02, fp_hi=0.20, jit_med=0.08, jit_sd=0.35)


def smooth_field(rng, s):
    h, w = max(3, int(FH / s)), max(3, int(FW / s)); f = cv2.resize(rng.normal(0, 1, (h, w)).astype(np.float32), (FW, FH), interpolation=cv2.INTER_CUBIC); return (f - f.min()) / (np.ptp(f) + 1e-6)


def noisy_peaks(rng, c, a):
    n = NOISE; g = rng.uniform(n['q_lo'], n['q_hi']); F = smooth_field(rng, rng.uniform(150, 450)); Q = np.clip(g + rng.uniform(0.2, 1.0) * (F - 0.5), 0, 1); qc = Q[np.clip(c[:, 1].astype(int), 0, FH - 1), np.clip(c[:, 0].astype(int), 0, FW - 1)]
    keep = rng.random(len(c)) >= n['miss'] * (1 - qc) ** n['miss_pow']; conf = 0.05 + 0.87 * rng.beta(n['a0'] + n['a1'] * qc[keep], n['b']); sg = n['jit_med'] * np.exp(rng.normal(0, n['jit_sd'])) * a; p = c[keep] + rng.normal(0, sg, (keep.sum(), 2))
    rate = rng.uniform(n['fp_lo'], n['fp_hi']) * (1.3 - qc.mean()); nf = rng.poisson(rate * len(c)); w = (1.15 - qc); w = w / w.sum(); src = rng.choice(len(c), nf, p=w); th = rng.uniform(0, 2 * np.pi, nf); r = np.interp(rng.random(nf), FALSE_P, OFF_Q) * a; fp = c[src] + np.c_[r * np.cos(th), r * np.sin(th)]; cf = np.interp(rng.random(nf), FALSE_P, FALSE_Q)
    pk = np.r_[np.c_[p, conf], np.c_[fp, cf]]; pk = pk[(pk[:, 0] >= 0) & (pk[:, 0] < FW) & (pk[:, 1] >= 0) & (pk[:, 1] < FH) & (pk[:, 2] >= 0.2)]; return pk, dict(g=float(g), jitter=float(sg / a), fp_rate=float(rate))


def graph(q, x0, y0):
    p = q[:, :2].astype(np.float64); c = q[:, 2]; n = len(p)
    if n < 12: return None
    try:
        vor = Voronoi(p); tri = Delaunay(p)
    except Exception:
        return None
    ar = np.full(n, np.nan); s = np.zeros(n); ub = np.zeros(n)
    for i in range(n):
        g = vor.regions[vor.point_region[i]]
        if -1 in g or not g: ub[i] = 1; s[i] = max(len(g) - 1, 3); continue
        v = vor.vertices[g]; s[i] = len(g)
        if (tri.find_simplex(v) < 0).any(): ub[i] = 1
        x, z = v[:, 0], v[:, 1]; ar[i] = 0.5 * abs(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1)))
    E = np.array(vor.ridge_points); d = np.linalg.norm(p[E[:, 0]] - p[E[:, 1]], axis=1); cnt = np.bincount(E.ravel(), minlength=n).astype(float); dm = (np.bincount(E[:, 0], d, n) + np.bincount(E[:, 1], d, n)) / np.maximum(cnt, 1); cn = (np.bincount(E[:, 0], c[E[:, 1]], n) + np.bincount(E[:, 1], c[E[:, 0]], n)) / np.maximum(cnt, 1); nn = cKDTree(p).query(p, k=2)[0][:, 1]
    med_a = np.nanmedian(ar[ub == 0]) if (ub == 0).sum() > 3 else np.nanmedian(ar); F = np.c_[c, np.nan_to_num(np.log(np.maximum(ar, 1) / max(med_a, 1)), nan=0.0, posinf=0.0, neginf=0.0).clip(-2, 2), np.log(dm / np.median(dm)).clip(-2, 2), np.log(nn / np.median(nn)).clip(-2, 2), s.clip(3, 9), cn, ub, (p[:, 0] - x0) / W, (p[:, 1] - y0) / H]
    return F.astype(np.float32), E.astype(np.int32)


def one(a_):
    fid, draws, seed = a_
    rng = np.random.default_rng(seed); cd = float(np.clip(2830 * np.exp(rng.normal(0, 0.18)), 1300, 3900)); lat = rng.random() < LATTICE
    if lat:
        # jittered hexagonal lattice with spatially varying disorder and per-cell size weights: reaches the regular mosaics, HEX up to about 0.85
        p, a = S.centres(rng, cd, 0.0); jm = float(rng.uniform(*LJIT)); F = smooth_field(rng, rng.uniform(120, 400)); jx = jm * (1 + LVAR * (2 * F - 1)); jc = jx[np.clip(p[:, 1].astype(int), 0, FH - 1), np.clip(p[:, 0].astype(int), 0, FW - 1)]; p = p + rng.normal(0, 1, p.shape) * (jc * a)[:, None]; ssd = float(rng.uniform(*LSSD)); w = rng.lognormal(0, ssd, len(p)); w = (w / w.mean() - 1) * a ** 2 * 0.5; lab = S.power_labels(p, w); relax = -1; jit = jm
    else:
        relax = int(rng.integers(RELAX[0], RELAX[1] + 1)); jit = float(rng.uniform(*JIT)); ssd = float(np.exp(rng.uniform(np.log(SSD[0]), np.log(SSD[1])))); lab, _, a = S.geometry(rng, cd, jit, ssd, relax)
    lab = lab.astype(np.int64); n = int(lab.max()) + 1; area = np.bincount(lab.ravel(), minlength=n).astype(float); yy, xx = np.mgrid[0:FH, 0:FW]; ll = lab.ravel(); cx = np.bincount(ll, xx.ravel(), n) / np.maximum(area, 1); cy = np.bincount(ll, yy.ravel(), n) / np.maximum(area, 1); edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]); ok = area > 0; ok[edge] = False; ids = np.where(ok)[0]; C = np.c_[cx[ids], cy[ids]]; AR = area[ids] * UM ** 2; out = []
    vor = Voronoi(C); tri = Delaunay(C); SID = np.zeros(len(C), int)
    for i in range(len(C)):
        g = vor.regions[vor.point_region[i]]
        if -1 in g or not g or (tri.find_simplex(vor.vertices[g]) < 0).any(): continue
        SID[i] = len(g)
    for d in range(draws):
        pk, info = noisy_peaks(rng, C, a); q = pk[pk[:, 2] >= 0.35]; hi = pk[pk[:, 2] >= 0.5, :2].astype(np.float64)
        if len(hi) < 10 or len(q) < 16: continue
        Xs, Es = [], []; sel = np.zeros(len(C), bool); base = 0
        for x0, y0 in L.windows(hi, W, H, k=3):
            m = (q[:, 0] >= x0) & (q[:, 0] < x0 + W) & (q[:, 1] >= y0) & (q[:, 1] < y0 + H); g = graph(q[m], x0, y0); sel |= (C[:, 0] >= x0) & (C[:, 0] < x0 + W) & (C[:, 1] >= y0) & (C[:, 1] < y0 + H)
            if g is not None: Xs.append(g[0]); Es.append(g[1] + base); base += len(g[0])
        b_ = sel & (SID > 0)
        if not Xs or b_.sum() < 12: continue
        t = [np.log(max((SID[b_] == 6).mean(), 0.02)), np.log(max(AR[sel].std() / AR[sel].mean(), 0.02)), np.log(1e6 / AR[sel].mean())]; out.append((fid, np.concatenate(Xs), np.concatenate(Es), np.array(t, np.float32), [info['g'], info['jitter'], info['fp_rate'], relax, jit, ssd]))
    return out


def main():
    o, draws, frames = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1)))
    with get_context('fork').Pool(workers) as pool: res = [r for v in pool.imap_unordered(one, [(i, draws, 5000 + i) for i in range(frames)], chunksize=2) for r in v]
    X, E, T, FID, PR, offs, eoff = [], [], [], [], [], [0], [0]; n = m = 0
    for fid, x, e, t, pr in res:
        X.append(x); E.append(e + n); T.append(t); FID.append(fid); PR.append(pr); n += len(x); m += len(e); offs.append(n); eoff.append(m)
    X = np.concatenate(X); np.savez(f'{A}/sweep/{o}', X=X, E=np.concatenate(E), T=np.array(T), fid=np.array(FID), params=np.array(PR, np.float32), offs=np.array(offs), eoff=np.array(eoff), noise=json.dumps(NOISE)); T_ = np.array(T)
    Q = (5, 10, 25, 50, 75, 90, 95); print('frames', len(T_), 'nodes', n, 'edges', m, 'workers', workers); print('sim conf quantiles', np.round(np.percentile(X[:, 0], Q), 3).tolist(), '| real [0.413, 0.478, 0.652, 0.816, 0.88, 0.905, 0.914]'); print('sim sides share 3..9', np.round(np.bincount(X[:, 4].astype(int), minlength=10)[3:10] / len(X), 3).tolist(), '| real [0.022, 0.073, 0.289, 0.44, 0.152, 0.021, 0.003]'); print('sim unbounded', round(float(X[:, 6].mean()), 3), 'log area SD', round(float(X[:, 1].std()), 3), 'log dist SD', round(float(X[:, 2].std()), 3), '| real 0.288, 0.589, 0.278'); k = np.diff(offs); print('nodes per frame p5 50 95', np.percentile(k, (5, 50, 95)).tolist(), '| real [208, 483, 642]'); print('targets exp p5 50 95: HEX, CV, CD', np.round(np.exp(np.percentile(T_, (5, 50, 95), axis=0)), 3).tolist(), '| real labels HEX 0.40 0.54 0.70, CV 0.27 0.36 0.53, CD 2000 2830 3400', flush=True); print('r of targets: HEX-CV', round(float(np.corrcoef(T_[:, 0], T_[:, 1])[0, 1]), 3), '| real about -0.6', flush=True)


if __name__ == '__main__':
    main()
