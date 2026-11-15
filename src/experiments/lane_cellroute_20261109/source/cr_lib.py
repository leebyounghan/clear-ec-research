"""Cell-route lane (2026-11-09): shared code. Konan centre-method weighted bisectors (KG takahashi1996konanpatent; same rule as
lane_cv_20261108/source/konan25_20261109.py: cut point on each Delaunay edge i-j divides it d_i^k : d_j^k, d = mean neighbour distance;
interior cells = not on the hull, max angular neighbour gap < 100 deg; HEX = share of 6-sided cells, sides counted with edge > 0.08 d_i),
plus point cleaning: merge close pairs, fill Delaunay holes with real low-confidence candidates or virtual points."""
import numpy as np
from scipy.spatial import Delaunay, cKDTree

SQ3 = np.sqrt(3.0)


def _clip(poly, n, c):
    out = []
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        fa, fb = (a - c) @ n, (b - c) @ n
        if fa <= 0: out.append(a)
        if fa * fb < 0: out.append(a + (b - a) * (fa / (fa - fb)))
    return np.array(out) if out else np.zeros((0, 2))


def neighbours(tri):
    ip, ix = tri.vertex_neighbor_vertices
    return [ix[ip[i]:ip[i + 1]] for i in range(len(ip) - 1)]


def konan_cells(p, sel=None, ks=(0, 2)):
    """Cells for points p (n,2). sel: boolean mask of seeds whose cells are read (default all). Returns {k: (areas, sides)}."""
    res = {k: ([], []) for k in ks}
    if len(p) < 8: return res
    try: tri = Delaunay(p)
    except Exception: return res
    hull = np.zeros(len(p), bool); hull[np.unique(tri.convex_hull)] = True; nb = neighbours(tri)
    d = np.array([np.linalg.norm(p[j] - p[i], axis=1).mean() if len(j) else np.nan for i, j in enumerate(nb)])
    for i in range(len(p)):
        if hull[i] or len(nb[i]) < 3 or (sel is not None and not sel[i]): continue
        v = p[nb[i]] - p[i]; ang = np.sort(np.arctan2(v[:, 1], v[:, 0])); gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
        if gaps.max() > np.deg2rad(100): continue
        L = np.linalg.norm(v, axis=1); nv = v / L[:, None]
        for k in ks:
            R = 3 * d[i]; poly = p[i] + np.array([[-R, -R], [R, -R], [R, R], [-R, R]])
            t = d[i] ** k / (d[i] ** k + d[nb[i]] ** k)
            for jj in range(len(nb[i])):
                poly = _clip(poly, nv[jj], p[i] + t[jj] * v[jj])
                if len(poly) < 3: break
            if len(poly) < 3: continue
            x, y = poly[:, 0], poly[:, 1]; res[k][0].append(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
            e = np.linalg.norm(poly - np.roll(poly, -1, 0), axis=1); res[k][1].append(int((e > 0.08 * d[i]).sum()))
    return res


def summarize(res, k):
    a, s = np.asarray(res[k][0]), np.asarray(res[k][1])
    if len(a) < 10: return np.nan, np.nan, len(a)
    return a.std() / a.mean(), float(np.mean(s == 6)), len(a)


def local_spacing(p, tri=None):
    """Per-point spacing: median over the point and its Delaunay neighbours of each one's median edge length."""
    tri = tri if tri is not None else Delaunay(p); nb = neighbours(tri)
    e = np.array([np.median(np.linalg.norm(p[j] - p[i], axis=1)) if len(j) else np.nan for i, j in enumerate(nb)])
    return np.array([np.nanmedian(np.r_[e[i], e[j]]) for i, j in enumerate(nb)])


def merge_close(p, w, frac=0.5, iters=4):
    """Merge pairs closer than frac * local spacing into their weight-weighted mean (closest pairs first). w: confidences."""
    p, w = p.copy(), w.copy()
    for _ in range(iters):
        if len(p) < 8: break
        s = local_spacing(p); tr = cKDTree(p); pairs = tr.query_pairs(frac * np.nanmax(s), output_type='ndarray')
        if len(pairs) == 0: break
        d = np.linalg.norm(p[pairs[:, 0]] - p[pairs[:, 1]], axis=1); thr = frac * 0.5 * (s[pairs[:, 0]] + s[pairs[:, 1]]); ok = d < thr
        if not ok.any(): break
        pairs, d = pairs[ok], d[ok]; used = np.zeros(len(p), bool); keep = np.ones(len(p), bool); newp, neww = [], []
        for a, b in pairs[np.argsort(d)]:
            if used[a] or used[b]: continue
            used[a] = used[b] = True; keep[a] = keep[b] = False; ww = w[a] + w[b] + 1e-9
            newp.append((w[a] * p[a] + w[b] * p[b]) / ww); neww.append(max(w[a], w[b]))
        p = np.r_[p[keep], np.asarray(newp)]; w = np.r_[w[keep], np.asarray(neww)]
    return p, w


def _circ(P3):
    a = np.linalg.norm(P3[:, 1] - P3[:, 2], axis=1); b = np.linalg.norm(P3[:, 0] - P3[:, 2], axis=1); c = np.linalg.norm(P3[:, 0] - P3[:, 1], axis=1)
    area = 0.5 * np.abs((P3[:, 1, 0] - P3[:, 0, 0]) * (P3[:, 2, 1] - P3[:, 0, 1]) - (P3[:, 2, 0] - P3[:, 0, 0]) * (P3[:, 1, 1] - P3[:, 0, 1]))
    return a * b * c / (4 * np.maximum(area, 1e-9)), area


def fill_gaps(p, w, cand=None, cR=0.85, mmax=3, min_sep=0.6, lloyd=8):
    """Holes = connected groups of Delaunay triangles with circumradius > cR * local spacing, not touching the hull.
    Missing count m from Euler on the hole: triangles T = b - 2 + 2 (m + interior). Fill with real candidates (cand: (n,3) x,y,conf,
    farthest-from-points first by confidence, >= min_sep * s apart) and then virtual points by Lloyd with the hole's vertices fixed."""
    if len(p) < 12: return p, w, 0, 0
    tri = Delaunay(p); s = local_spacing(p, tri); S = tri.simplices; R, A = _circ(p[S]); st = s[S].mean(1)
    big = R > cR * st; touches = (tri.neighbors == -1).any(1)
    lab = -np.ones(len(S), int); comp = 0; newp, neww = [], []; n_real = n_virt = 0
    for t0 in np.where(big)[0]:
        if lab[t0] >= 0: continue
        stack = [t0]; lab[t0] = comp; mem = []
        while stack:
            t = stack.pop(); mem.append(t)
            for u in tri.neighbors[t]:
                if u >= 0 and big[u] and lab[u] < 0: lab[u] = comp; stack.append(u)
        comp += 1
        if touches[mem].any(): continue
        mem = np.array(mem); verts = np.unique(S[mem]); edges = {}
        for t in mem:
            for q in range(3):
                e = tuple(sorted((S[t, q], S[t, (q + 1) % 3]))); edges[e] = edges.get(e, 0) + 1
        bverts = np.unique([v for e, c in edges.items() if c == 1 for v in e]); b = len(bverts); inner = len(verts) - b
        sc = float(np.median(s[bverts])); T = A[mem].sum() / (SQ3 / 4 * sc ** 2); m = int(round((T - b + 2) / 2 - inner))
        if m < 1 or m > mmax: continue
        lo, hi = p[verts].min(0), p[verts].max(0); g = np.mgrid[lo[0]:hi[0]:1.5, lo[1]:hi[1]:1.5].reshape(2, -1).T
        inside = np.isin(tri.find_simplex(g), mem); g = g[inside]
        if len(g) < 4: continue
        fixed = p[verts]; added = []
        if cand is not None and len(cand):
            ci = cand[np.isin(tri.find_simplex(cand[:, :2]), mem)]
            for c in ci[np.argsort(-ci[:, 2])]:
                pts = np.r_[fixed, np.asarray(added)] if added else fixed
                if np.min(np.linalg.norm(pts - c[:2], axis=1)) >= min_sep * sc: added.append(c[:2]); neww.append(c[2]); n_real += 1
                if len(added) >= m: break
        nv = m - len(added)
        if nv > 0:
            rng = np.random.default_rng(0); new = g[rng.choice(len(g), nv, replace=False)].astype(float); base = np.r_[fixed, np.asarray(added)] if added else fixed
            for _ in range(lloyd):
                G = np.r_[base, new]; lab_g = cKDTree(G).query(g)[1] - len(base)
                for q in range(nv):
                    if (lab_g == q).any(): new[q] = g[lab_g == q].mean(0)
            added.extend(list(new)); neww.extend([0.0] * nv); n_virt += nv
        newp.extend(added)
    if newp: p = np.r_[p, np.asarray(newp)]; w = np.r_[w, np.asarray(neww)]
    return p, w, n_real, n_virt


def clean(p, w, cand=None, merge_frac=0.5, cR=0.85, mmax=3, rounds=2):
    p, w = merge_close(p, w, merge_frac); nr = nv = 0
    for _ in range(rounds):
        p, w, a, b = fill_gaps(p, w, cand, cR, mmax); nr += a; nv += b
        if a + b == 0: break
    return p, w, nr, nv


def konan_cells_idx(p, ks=(0, 2)):
    """Per-seed version over the whole point set: returns seed index array and {k: (areas, sides)} aligned with it (interior cells only)."""
    out_i = []; res = {k: ([], []) for k in ks}
    if len(p) < 8: return np.zeros(0, int), {k: (np.zeros(0), np.zeros(0, int)) for k in ks}, None
    try: tri = Delaunay(p)
    except Exception: return np.zeros(0, int), {k: (np.zeros(0), np.zeros(0, int)) for k in ks}, None
    hull = np.zeros(len(p), bool); hull[np.unique(tri.convex_hull)] = True; nb = neighbours(tri)
    d = np.array([np.linalg.norm(p[j] - p[i], axis=1).mean() if len(j) else np.nan for i, j in enumerate(nb)])
    for i in range(len(p)):
        if hull[i] or len(nb[i]) < 3: continue
        v = p[nb[i]] - p[i]; ang = np.sort(np.arctan2(v[:, 1], v[:, 0])); gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
        if gaps.max() > np.deg2rad(100): continue
        L = np.linalg.norm(v, axis=1); nv = v / L[:, None]; got = {}
        for k in ks:
            R = 3 * d[i]; poly = p[i] + np.array([[-R, -R], [R, -R], [R, R], [-R, R]]); t = d[i] ** k / (d[i] ** k + d[nb[i]] ** k)
            for jj in range(len(nb[i])):
                poly = _clip(poly, nv[jj], p[i] + t[jj] * v[jj])
                if len(poly) < 3: break
            if len(poly) < 3: break
            x, y = poly[:, 0], poly[:, 1]; e = np.linalg.norm(poly - np.roll(poly, -1, 0), axis=1)
            got[k] = (0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))), int((e > 0.08 * d[i]).sum()))
        if len(got) < len(ks): continue
        out_i.append(i)
        for k in ks: res[k][0].append(got[k][0]); res[k][1].append(got[k][1])
    return np.array(out_i, int), {k: (np.array(res[k][0]), np.array(res[k][1], int)) for k in ks}, nb


def vote(sources, s, frac=0.45, min_votes=2):
    """Greedy cross-source clustering: highest-confidence point first takes the nearest unassigned point of each other source within frac*s.
    Returns cluster means and weight = share of sources in the cluster."""
    srcs = [np.c_[q[:, :3], np.full(len(q), j)] for j, q in enumerate(sources) if len(q)]
    if not srcs: return np.zeros((0, 2)), np.zeros(0)
    allp = np.concatenate(srcs); order = np.argsort(-allp[:, 2]); used = np.zeros(len(allp), bool); tr = cKDTree(allp[:, :2]); P_, W_ = [], []
    for a in order:
        if used[a]: continue
        idx = [b for b in tr.query_ball_point(allp[a, :2], frac * s) if not used[b]]; mem = {int(allp[a, 3]): a}
        for b in sorted(idx, key=lambda b: np.linalg.norm(allp[b, :2] - allp[a, :2])):
            sj = int(allp[b, 3])
            if sj not in mem: mem[sj] = b
        mm = list(mem.values()); used[mm] = True
        if len(mm) >= min_votes: P_.append(allp[mm, :2].mean(0)); W_.append(len(mm) / len(sources))
    return np.asarray(P_).reshape(-1, 2), np.asarray(W_)


def nms(p, w, s, alpha=0.7):
    """Greedy suppression by confidence: drop a point closer than alpha*s (frame spacing) to an already kept, stronger point
    (junction false positives sit ~0.58 s from true centres and have lower confidence)."""
    order = np.argsort(-w); keep = []; tr_pts = np.zeros((0, 2))
    for i in order:
        if len(keep) and np.min(np.linalg.norm(p[keep] - p[i], axis=1)) < alpha * s: continue
        keep.append(i)
    keep = np.array(keep, int); return p[keep], w[keep]


def clean2(p, w, s, cand=None, alpha=0.7, cR=0.85, mmax=3, rounds=2):
    p, w = nms(p, w, s, alpha); nr = nv = 0
    for _ in range(rounds):
        p, w, a, b = fill_gaps(p, w, cand, cR, mmax); nr += a; nv += b
        if a + b == 0: break
    return p, w, nr, nv
