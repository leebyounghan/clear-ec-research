"""Cell-route lane (2026-11-09): 25 expert-marker frame gate for point sets fed to the Konan k=2 rule.
For each variant: points per frame -> cells whose seeds lie inside the marker hull (Delaunay over all points near the box) ->
CV (area SD/mean) and HEX (6-sided share) for k=2 and k=0; miss / FP / jitter (12 px, Hungarian, in hull) against the markers.
Read against device CV / HEX: Pearson r (log for CV) and LOO-scaled MAPE (as konan25_20261109.py).
Reference: markers k=2 CV r 0.899 / 7.64, HEX r 0.845 / 7.11. usage: gate25.py [variant-group ...]; CPU, WORKERS (default 3)."""
import json, os, sys, time
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy import ndimage as ndi
sys.path[:0] = ['/workspace/clear-ec-research/experiments/lane_cellroute_20261109/source', '/workspace/clear-ec-research/experiments/lane_cv_20261108/source']
import cr_lib as L
import peak_select25_20261108 as P
ROOT = '/workspace/clear-ec-research/experiments/'
OUT = ROOT + 'lane_cellroute_20261109/out/'
names = P.names; MK = P.MK; rows = [P.pos[n] for n in names]; DS = [int(P.idx[r]) for r in rows]; FO = [int(P.fold[r]) for r in rows]
gt_cv = np.array([P.ann[n]['gt']['CV'] for n in names]); gt_hex = np.array([P.ann[n]['gt']['HEX'] for n in names])
SPM = [float(np.median(cKDTree(m).query(m, k=2)[0][:, 1])) for m in MK]


def load_peaks(tag):
    out = {}
    for f in sorted(set(FO)):
        z = np.load(ROOT + f'lane_cv_20261108/out/tta_peaks/{tag}/fold{f}/fold{f}_sg2_peaks.npz', allow_pickle=True); pk, of, ix = z['peaks'], z['offsets'], z['indices']
        pos = {int(i): k for k, i in enumerate(ix)}
        for d, ff in zip(DS, FO):
            if ff == f and d in pos: out[d] = pk[of[pos[d]]:of[pos[d] + 1]].astype(float)
    return out


def load_seg():
    out = {}; want = set(DS)
    for c in range(140):
        fn = ROOT + f'lane_hex_20261108/seg_loose/c{c:03d}.npz'
        if not os.path.exists(fn): continue
        z = np.load(fn); fr = z['frames']; hit = [k for k, d in enumerate(fr) if int(d) in want]
        MS = z['masks'] if hit else None
        for k in hit:
            m = MS[k]; lab = np.unique(m); lab = lab[lab > 0]
            if len(lab) == 0: out[int(fr[k])] = np.zeros((0, 3)); continue
            cy, cx = np.array(ndi.center_of_mass(np.ones_like(m), m, lab)).T; ar = ndi.sum(np.ones_like(m), m, lab)
            out[int(fr[k])] = np.c_[cx, cy, np.full(len(lab), 0.5), ar]
    return out


PEAKS = {t: load_peaks(t) for t in ('seeds3_tta4', 's46_tta4', 's47_tta4', 's48_tta4', 's49_tta4', 's50_tta4')}
t0 = time.time(); SEG = load_seg(); print('seg centroids for', len(SEG), 'frames', round(time.time() - t0, 1), 's', flush=True)


def frame_spacing(i):
    pk = PEAKS['seeds3_tta4'][DS[i]]; hi = pk[pk[:, 2] >= 0.5, :2]
    return float(np.median(cKDTree(hi).query(hi, k=2)[0][:, 1])) if len(hi) > 20 else SPM[i]


def vote(sources, s, frac=0.45, min_votes=2):
    """Greedy cross-source clustering: highest-confidence point first, takes the nearest unassigned point of each other source within frac*s."""
    allp = np.concatenate([np.c_[q[:, :3], np.full(len(q), j)] for j, q in enumerate(sources) if len(q)]) if sources else np.zeros((0, 4))
    order = np.argsort(-allp[:, 2]); used = np.zeros(len(allp), bool); tr = cKDTree(allp[:, :2]); P_, W_ = [], []
    for a in order:
        if used[a]: continue
        idx = [b for b in tr.query_ball_point(allp[a, :2], frac * s) if not used[b]]; mem = {int(allp[a, 3]): a}
        for b in sorted(idx, key=lambda b: np.linalg.norm(allp[b, :2] - allp[a, :2])):
            sj = int(allp[b, 3])
            if sj not in mem: mem[sj] = b
        mm = list(mem.values()); used[mm] = True
        if len(mm) >= min_votes: P_.append(allp[mm, :2].mean(0)); W_.append(len(mm) / len(sources))
    return np.asarray(P_).reshape(-1, 2), np.asarray(W_)


def corrupt(mk, s, rng, miss=0.0, fp=0.0, jit=0.0):
    p = mk.copy()
    if jit > 0: p = p + rng.normal(0, jit * s / 1.2533, p.shape)
    if miss > 0: p = p[rng.random(len(p)) >= miss]
    if fp > 0:
        hull = Delaunay(mk); lo, hi = mk.min(0), mk.max(0); n = int(round(fp / (1 - fp) * len(p))); add = []
        while len(add) < n:
            c = lo + rng.random(2) * (hi - lo)
            if hull.find_simplex(c) >= 0: add.append(c)
        p = np.r_[p, np.asarray(add).reshape(-1, 2)]
    return p


def points(i, v):
    """Returns (points (n,2), weights, n_real_added, n_virtual_added) for frame i under variant dict v."""
    d = DS[i]; s_img = frame_spacing(i); src = v['src']; rng = np.random.default_rng(1000 * v.get('draw', 0) + i)
    if src == 'markers':
        p = corrupt(MK[i], SPM[i], rng, v.get('miss', 0), v.get('fp', 0), v.get('jit', 0)); w = np.ones(len(p)); cand = None
    elif src in PEAKS:
        pk = PEAKS[src][d]; t = v.get('t', 0.35); q = pk[pk[:, 2] >= t]; p, w = q[:, :2], q[:, 2]; c = pk[(pk[:, 2] >= v.get('tc', 0.2)) & (pk[:, 2] < t)]; cand = c if v.get('cand') else None
    elif src == 'seg':
        q = SEG[d]; q = q[q[:, 3] >= v.get('amin', 0.15) * s_img ** 2] if len(q) else q; p, w = q[:, :2], q[:, 2]; cand = None
    elif src == 'vote':
        t = v.get('t', 0.35); S_ = [PEAKS[k][d] for k in v['members'] if k != 'seg']; S_ = [q[q[:, 2] >= t] for q in S_]
        if 'seg' in v['members']:
            q = SEG[d]; S_.append(q[q[:, 3] >= v.get('amin', 0.15) * s_img ** 2] if len(q) else np.zeros((0, 4)))
        p, w = vote(S_, s_img, v.get('vfrac', 0.45), v.get('minv', 2)); cand = None
        if v.get('cand'):
            pk = PEAKS['seeds3_tta4'][d]; cand = pk[pk[:, 2] >= 0.2]
    else: raise ValueError(src)
    nr = nv = 0
    if v.get('nms'):
        if v.get('fill'): p, w, nr, nv = L.clean2(p, w, s_img, cand, v['nms'], v.get('cR', 0.85), v.get('mmax', 3))
        else: p, w = L.nms(p, w, s_img, v['nms'])
    elif v.get('clean'): p, w, nr, nv = L.clean(p, w, cand, v.get('mfrac', 0.5), v.get('cR', 0.85), v.get('mmax', 3))
    elif v.get('merge'): p, w = L.merge_close(p, w, v.get('mfrac', 0.5))
    return p, w, nr, nv


def one(args):
    i, vname, v = args; mk = MK[i]; p, w, nr, nv = points(i, v)
    if v['src'] == 'markers' and v.get('all_cells'): sel = None; q = p
    else:
        lo, hi = mk.min(0) - 4 * SPM[i], mk.max(0) + 4 * SPM[i]; box = (p >= lo).all(1) & (p <= hi).all(1); q = p[box]
        sel = Delaunay(mk).find_simplex(q) >= 0 if len(q) else np.zeros(0, bool)
        if v.get('creg') is not None and len(q) > 8:
            wq = w[box]; nb = L.neighbours(Delaunay(q)); sel = sel & np.array([wq[j] >= v['creg'] and (wq[nb[j]] >= v['creg']).all() for j in range(len(q))])
    if v.get('click'):
        rng = np.random.default_rng(7 + i); acc = []
        for _ in range(v.get('K', 8)):
            r_ = L.konan_cells(q + rng.normal(0, v['click'] * SPM[i], q.shape), sel); acc.append([*L.summarize(r_, 0), *L.summarize(r_, 2)])
        acc = np.nanmean(np.array(acc, float), 0); cv0, hx0, n0, cv2, hx2, n2 = acc
    else:
        res = L.konan_cells(q, sel)
        cv0, hx0, n0 = L.summarize(res, 0); cv2, hx2, n2 = L.summarize(res, 2)
    qi = q if sel is None else q[sel]; mm = P.match(qi, mk, SPM[i])
    return vname, i, [cv0, hx0, cv2, hx2, n2, mm[0], mm[1], mm[2], nr, nv]


def read(x, g, log=True):
    m = np.isfinite(x) & (x > 0)
    if m.sum() < 15: return (np.nan, np.nan, int(m.sum()))
    lx, lg = (np.log(x[m]), np.log(g[m])) if log else (x[m], g[m]); r = np.corrcoef(lx, lg)[0, 1]
    lx, lg = np.log(x[m]), np.log(g[m]); e = []
    for j in range(m.sum()):
        o = np.arange(m.sum()) != j; e.append(abs(np.exp(lx[j] - lx[o].mean() + lg[o].mean()) - g[m][j]) / g[m][j])
    return (round(float(r), 3), round(100 * float(np.mean(e)), 2), int(m.sum()))


def variants(groups):
    V = {}
    if 'ref' in groups:
        V['markers all-cells'] = dict(src='markers', all_cells=True); V['markers hull-read'] = dict(src='markers')
        V['markers clean (identity check)'] = dict(src='markers', clean=True); V['markers clean cR1.0'] = dict(src='markers', clean=True, cR=1.0)
        V['markers merge only'] = dict(src='markers', merge=True); V['markers fill only cR1.0'] = dict(src='markers', clean=True, cR=1.0, mfrac=0.0)
    if 'corrupt' in groups:
        for nm, kw in (('miss15', dict(miss=0.15)), ('fp16.5', dict(fp=0.165)), ('jit0.215', dict(jit=0.215)), ('all3', dict(miss=0.15, fp=0.165, jit=0.215))):
            for dr in range(4):
                V[f'markers {nm} d{dr}'] = dict(src='markers', draw=dr, **kw); V[f'markers {nm} +clean d{dr}'] = dict(src='markers', draw=dr, clean=True, **kw)
                V[f'markers {nm} +clean1.0 d{dr}'] = dict(src='markers', draw=dr, clean=True, cR=1.0, **kw)
    if 'peaks' in groups:
        for t in (0.2, 0.35, 0.5):
            V[f'seeds3 t{t}'] = dict(src='seeds3_tta4', t=t); V[f'seeds3 t{t} merge'] = dict(src='seeds3_tta4', t=t, merge=True); V[f'seeds3 t{t} clean'] = dict(src='seeds3_tta4', t=t, clean=True)
        V['seeds3 t0.35 clean+cand0.2'] = dict(src='seeds3_tta4', t=0.35, clean=True, cand=True); V['seeds3 t0.5 clean+cand0.2'] = dict(src='seeds3_tta4', t=0.5, clean=True, cand=True)
        V['seeds3 t0.35 clean1.0'] = dict(src='seeds3_tta4', t=0.35, clean=True, cR=1.0); V['seeds3 t0.35 clean1.0+cand'] = dict(src='seeds3_tta4', t=0.35, clean=True, cR=1.0, cand=True)
        V['seeds3 t0.2 clean1.0'] = dict(src='seeds3_tta4', t=0.2, clean=True, cR=1.0)
        V['seg'] = dict(src='seg'); V['seg clean'] = dict(src='seg', clean=True)
    if 'union' in groups:
        M3 = ['s46_tta4', 's47_tta4', 's48_tta4']
        for mv in (1, 2, 3):
            V[f'vote3 min{mv}'] = dict(src='vote', members=M3, minv=mv); V[f'vote3 min{mv} clean'] = dict(src='vote', members=M3, minv=mv, clean=True)
        for mv in (1, 2, 3):
            V[f'vote3+seg min{mv}'] = dict(src='vote', members=M3 + ['seg'], minv=mv); V[f'vote3+seg min{mv} clean'] = dict(src='vote', members=M3 + ['seg'], minv=mv, clean=True)
        V['vote3+seg min2 clean+cand'] = dict(src='vote', members=M3 + ['seg'], minv=2, clean=True, cand=True)
        V['vote3+seg min2 clean1.0'] = dict(src='vote', members=M3 + ['seg'], minv=2, clean=True, cR=1.0); V['vote3 min2 clean1.0'] = dict(src='vote', members=M3, minv=2, clean=True, cR=1.0)
        V['vote3 t0.2 min2 clean'] = dict(src='vote', members=M3, minv=2, clean=True, t=0.2)
    if 'nms' in groups:
        M4 = ['s46_tta4', 's47_tta4', 's48_tta4', 'seg']; M3 = M4[:3]
        for t in (0.2, 0.35):
            V[f'seeds3 t{t} nms0.7'] = dict(src='seeds3_tta4', t=t, nms=0.7); V[f'seeds3 t{t} nms0.7+fill'] = dict(src='seeds3_tta4', t=t, nms=0.7, fill=True)
            V[f'seeds3 t{t} nms0.7+fill1.0'] = dict(src='seeds3_tta4', t=t, nms=0.7, fill=True, cR=1.0)
        V['seeds3 t0.2 nms0.6+fill'] = dict(src='seeds3_tta4', t=0.2, nms=0.6, fill=True); V['seeds3 t0.2 nms0.8+fill'] = dict(src='seeds3_tta4', t=0.2, nms=0.8, fill=True)
        V['seeds3 t0.5 nms0.7+fill+cand'] = dict(src='seeds3_tta4', t=0.5, nms=0.7, fill=True, cand=True, tc=0.2)
        V['vote3+seg min1 nms0.7'] = dict(src='vote', members=M4, minv=1, nms=0.7); V['vote3+seg min1 nms0.7+fill'] = dict(src='vote', members=M4, minv=1, nms=0.7, fill=True)
        V['vote3 t0.2 min1 nms0.7+fill'] = dict(src='vote', members=M3, minv=1, t=0.2, nms=0.7, fill=True)
        V['seg nms0.7'] = dict(src='seg', nms=0.7); V['seg nms0.7+fill'] = dict(src='seg', nms=0.7, fill=True)
    if 'click' in groups:
        M4 = ['s46_tta4', 's47_tta4', 's48_tta4', 'seg']
        for sg in (0.1, 0.16, 0.22):
            V[f'seeds3 t0.2 click{sg}'] = dict(src='seeds3_tta4', t=0.2, click=sg); V[f'seeds3 t0.2 clean click{sg}'] = dict(src='seeds3_tta4', t=0.2, clean=True, click=sg)
            V[f'vote3+seg min1 click{sg}'] = dict(src='vote', members=M4, minv=1, click=sg)
    if 'click5' in groups:
        M6 = ['s46_tta4', 's47_tta4', 's48_tta4', 's49_tta4', 's50_tta4', 'seg']
        V['vote5+seg min1'] = dict(src='vote', members=M6, minv=1); V['vote5+seg min1 click0.16'] = dict(src='vote', members=M6, minv=1, click=0.16)
        V['vote5+seg min2 click0.16'] = dict(src='vote', members=M6, minv=2, click=0.16); V['vote5+seg min3 click0.16'] = dict(src='vote', members=M6, minv=3, click=0.16)
        V['vote3+seg min1 click0.16 K16'] = dict(src='vote', members=M6[:3] + ['seg'], minv=1, click=0.16, K=16)
    if 'creg' in groups:
        M4 = ['s46_tta4', 's47_tta4', 's48_tta4', 'seg']
        for c in (0.35, 0.5):
            V[f'seeds3 t0.2 creg{c}'] = dict(src='seeds3_tta4', t=0.2, creg=c); V[f'seeds3 t0.2 clean creg{c}'] = dict(src='seeds3_tta4', t=0.2, clean=True, cand=False, creg=c)
        V['vote3+seg min1 creg0.5'] = dict(src='vote', members=M4, minv=1, creg=0.5); V['vote3+seg min1 creg0.75'] = dict(src='vote', members=M4, minv=1, creg=0.75)
        V['vote3+seg min1 clean creg0.5'] = dict(src='vote', members=M4, minv=1, clean=True, creg=0.5)
    return V


if __name__ == '__main__':
    groups = sys.argv[1:] or ['ref', 'corrupt', 'peaks', 'union']; V = variants(groups)
    jobs = [(i, vn, v) for vn, v in V.items() for i in range(len(names))]; R = {vn: np.full((len(names), 10), np.nan) for vn in V}
    with get_context('fork').Pool(int(os.environ.get('WORKERS', 3))) as pool:
        for vn, i, r in pool.imap_unordered(one, jobs, chunksize=4): R[vn][i] = r
    out = {}
    def row(vn, A):
        return dict(miss=round(float(np.nanmean(A[:, 5])), 3), fp=round(float(np.nanmean(A[:, 6])), 3), jit=round(float(np.nanmean(A[:, 7])), 3), cells=float(np.nanmedian(A[:, 4])),
                    added_real=float(np.nanmean(A[:, 8])), added_virt=float(np.nanmean(A[:, 9])),
                    cv_k2=read(A[:, 2], gt_cv), hex_k2=read(A[:, 3], gt_hex, log=False), cv_k0=read(A[:, 0], gt_cv), hex_k0=read(A[:, 1], gt_hex, log=False))
    for vn in V: out[vn] = row(vn, R[vn])
    # average corrupted draws
    import re as _re
    base = sorted(set(_re.sub(r' d\d$', '', vn) for vn in V if _re.search(r' d\d$', vn)))
    for b in base:
        ks = [vn for vn in V if _re.sub(r' d\d$', '', vn) == b]; rs = [out[k] for k in ks]
        out[b + ' (mean of draws)'] = {k: (tuple(np.round(np.nanmean([r[k][j] for r in rs]), 3) for j in range(2)) if isinstance(rs[0][k], tuple) else round(float(np.nanmean([r[k] for r in rs])), 3)) for k in rs[0]}
    for vn, o in out.items():
        if _re.search(r' d\d$', vn): continue
        print(vn.ljust(34), 'miss %.3f fp %.3f jit %.3f cells %4.0f +r %.1f +v %.1f | CV k2 r %s MAPE %s | HEX k2 r %s MAPE %s | CV k0 %s %s | HEX k0 %s %s' % (
            o['miss'], o['fp'], o['jit'], o['cells'], o['added_real'], o['added_virt'], o['cv_k2'][0], o['cv_k2'][1], o['hex_k2'][0], o['hex_k2'][1], o['cv_k0'][0], o['cv_k0'][1], o['hex_k0'][0], o['hex_k0'][1]), flush=True)
    os.makedirs(OUT, exist_ok=True); json.dump({k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in o.items()} for k, o in out.items()}, open(OUT + 'gate25_' + '_'.join(groups) + '.json', 'w'), indent=1)
    np.savez_compressed(OUT + 'gate25_' + '_'.join(groups) + '_perframe.npz', **{k.replace(' ', '_'): v for k, v in R.items()})
