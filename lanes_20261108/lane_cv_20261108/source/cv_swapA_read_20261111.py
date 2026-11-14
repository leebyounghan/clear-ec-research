"""CV lane (2026-11-11, D-SYN-003): CV read of recipe A with the synthetic-pretrained s46 in place of the plain s46 (folds 0 and 4), strict,
incumbent (plain A) recomputed in the same run with the same code (L1, L6). Input: the CD lane's swapped A
(lane_cd_20261108/reads/recipeA_swapped_syn46_20261111.npz, key A, N x 3 with each row's fold LOFO scalars, plain elsewhere; A-CD-SWAPA-PLAN).
Recipe A enters the CV column in three places, all swapped:
  base     H_cv = B_cv = 0.5 (A_cv + P1_cv)  (checked against the stored B)
  ridge    BEST - GRAPH, the simulation-embedding ridge on the residual vs H exp(GRAPH): refit per fold with the deploy recipe (cv_deploy_fit
           solve, lambda by leave-one-fold-out inside the fold; with plain A this reproduces BEST - GRAPH to 3.6e-9, O-CV-L5-001)
  dino     the plain DINOv3 term is log member - log A_cv: shifted by log A_old - log A_new
  (graph nets, quartets and frozen heads are log residuals vs H or arm-vs-std differences and are kept, as the HEX lane's swap does)
Then the six D-CV-001 components with weights and stretch refit by the strict procedure, on folds 0+4 (2-fold universe, as
synthft_read_20261111.py 04) and on five folds (A swapped on folds 0+4 rows only).
usage: cv_swapA_read_20261111.py [FILE] [--dry]   --dry: A_new = plain A (must give zero deltas, 10.0451 / 10.1160)"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import stretch_nested_20261103 as N
import ceiling_meta_cpu_20261103 as C
X = Path('/workspace/clear-ec-research/experiments'); LANE = X / 'lane_cv_20261108'
sys.path.insert(0, str(LANE / 'source')); import cv_deploy_fit_20261110 as DF
args = [a for a in sys.argv[1:] if not a.startswith('--')]; DRY = '--dry' in sys.argv
FILE = Path(args[0]) if args else X / 'lane_cd_20261108/reads/recipeA_swapped_syn46_20261111.npz'
y3 = BP.y; y = y3[:, 1]; fold = BP.fold; good = BP.good; ok = y > 0; GRIDW = np.round(np.arange(0.125, 1.0001, 0.125), 3)
pos = {int(i): n for n, i in enumerate(BP.idx)}; A0 = BP._b['A'].astype(np.float64); P1 = BP._b['P1'].astype(np.float64)
chkB = float(np.max(np.abs(0.5 * (A0[:, 1] + P1[:, 1]) / BP.B[:, 1] - 1))); assert chkB < 1e-5, f'B is not 0.5 (A + P1): {chkB}'
out = {'file': None if DRY else str(FILE), 'dry': DRY, 'check_B_equals_half_A_plus_P1_max_rel': chkB}
if DRY: A1 = A0.copy()
else:
    z = np.load(FILE, allow_pickle=True); r = np.array([pos[int(i)] for i in z['indices']]); A1 = A0.copy(); A1[r] = z['A'].astype(np.float64)
    if 'A_plain' in z.files: Ap = np.full_like(A0, np.nan); Ap[r] = z['A_plain']; out['file_A_plain_vs_ours_max_rel'] = float(np.nanmax(np.abs(Ap[:, 1] / A0[:, 1] - 1)))
chg = np.abs(A1[:, 1] / A0[:, 1] - 1) > 1e-12; out['rows_changed_by_fold'] = {f: int((chg & (fold == f)).sum()) for f in range(5)}
out['A_cv_log_ratio_new_vs_old'] = {f: dict(mean=float(np.mean(np.log(A1[fold == f, 1] / A0[fold == f, 1]))), sd=float(np.std(np.log(A1[fold == f, 1] / A0[fold == f, 1])))) for f in range(5)}
print(json.dumps(out), flush=True)
EMB = DF.emb(); MASK = DF.MASK


def ridge_pred(H, tr, base_graph, lam):
    base = H * np.exp(base_graph * MASK); pr = np.zeros(len(y))
    for Zm in EMB:
        okz = np.isfinite(Zm).all(1); Zf = np.where(okz[:, None], Zm, 0); mu, sd, w, t0 = DF.solve(Zf, okz, tr, base, 1, lam); pr += np.where(okz, ((Zf - mu) / sd) @ w + t0, 0.0) / len(EMB)
    return pr


def build(A):
    """per-fold-model base log CV (LB, 5 x N) with the ridge refit on this A's H, and the lambdas chosen."""
    H = BP.B.copy(); H[:, 1] = 0.5 * (A[:, 1] + P1[:, 1]); LB = np.zeros((5, len(y))); lams = {}
    for f in range(5):
        sc = {}
        for lam in DF.LAMS:
            e = 0.0
            for q in range(5):
                if q == f: continue
                pr = ridge_pred(H, (fold != q) & (fold != f) & good, BP.GRAPH[q], lam); m = (fold == q) & ok
                e += float((np.abs(H[m, 1] * np.exp(BP.GRAPH[q][m, 1] + pr[m]) - y[m]) / y[m]).sum())
            sc[lam] = e
        lam = min(sc, key=sc.get); lams[f] = lam; LB[f] = np.log(H[:, 1]) + BP.GRAPH[f][:, 1] + ridge_pred(H, (fold != f) & good, BP.GRAPH[f], lam)
    return LB, lams


pool = np.load(LANE / 'out/pool_20261111.npz', allow_pickle=True); names = [str(n) for n in pool['names']]
six = [k for k, n in enumerate(names) if n in ('dino', 'fz:spectrum', 'fz:cov_sqrt', 'fz:autocorr') or n.startswith('graph:tta_s46_tta4_5f+') or n.startswith('graph:gord_s46_tta4_5f+')]
assert len(six) == 6; NAMES6 = [names[k] for k in six]; idino = NAMES6.index('dino')
C6 = np.stack([np.stack([pool[f'c{k}_f{f}'] for f in range(5)]) for k in six])
ape = lambda p, r: np.abs(p[r] - y[r]) / y[r] * 100


def stretch_fit(lp, rows):
    r = rows & ok; lb = lp[r]; mu = lb.mean(); yy = y[r]; best = None
    for g in N.GRID:
        q = np.exp(g * (lb - mu) + mu); c = C.wmed(yy / q, q / yy); m = np.mean(np.abs(c * q - yy) / yy)
        if best is None or m < best[0]: best = (m, g, c, mu)
    return best[1:]


def apply_st(lp, prm): g, c, mu = prm; return c * np.exp(g * (lp - mu) + mu)


def run(LB, COMP, FOLDS):
    K = len(COMP)
    def pre(gm, w): return LB[gm] + np.tensordot(w, COMP[:, gm, :], axes=1)
    def inner(w, f):
        tot = 0.0; n = 0
        for g in FOLDS:
            if g == f: continue
            rest = [h for h in FOLDS if h not in (f, g)] or [g]
            lp = pre(g, w); prm = stretch_fit(lp, np.isin(fold, rest) & good); sel = (fold == g) & ok; tot += np.sum(np.abs(apply_st(lp[sel], prm) - y[sel]) / y[sel]); n += sel.sum()
        return tot / n * 100
    def coord(w, obj):
        cur = obj(w)
        for _ in range(2):
            for k in range(K):
                best = (cur, w[k])
                for gv in np.r_[0.0, GRIDW]:
                    if gv == w[k]: continue
                    t = w.copy(); t[k] = gv; s = obj(t)
                    if s < best[0] - 1e-9: best = (s, gv)
                w[k] = best[1]; cur = best[0]
        return w
    P = np.full(len(y), np.nan); W = {}
    for f in FOLDS:
        w = coord(np.zeros(K), lambda w, f=f: inner(w, f)); lp = pre(f, w); prm = stretch_fit(lp, np.isin(fold, [h for h in FOLDS if h != f]) & good)
        P[fold == f] = apply_st(lp[fold == f], prm); W[f] = w
    return P, W


LB0, lam0 = build(A0); out['ridge_plain_vs_BEST_max_abs'] = float(max(np.max(np.abs(LB0[f] - (np.log(BP.B[:, 1]) + BP.BEST[f][:, 1]))) for f in range(5)))
LB1, lam1 = build(A1); out['lambdas'] = dict(plain=lam0, swapped=lam1)
C6s = C6.copy(); C6s[idino] = C6[idino] + (np.log(A0[:, 1]) - np.log(A1[:, 1]))[None, :]
print(json.dumps({k: out[k] for k in ('ridge_plain_vs_BEST_max_abs', 'lambdas')}), flush=True)
for FOLDS in ((0, 4), (0, 1, 2, 3, 4)):
    U = np.isin(fold, FOLDS); tag = ''.join(map(str, FOLDS)); res = {}
    for nm, LB, CC in (('plain A (incumbent, recomputed)', LB0, C6), ('synthetic s46 in recipe A', LB1, C6s)):
        P, W = run(LB, CC, FOLDS); res[nm] = dict(cv=float(ape(P, U & ok).mean()), per_fold={f: float(ape(P, (fold == f) & ok).mean()) for f in FOLDS}, weights={f: {NAMES6[k]: float(W[f][k]) for k in range(6) if W[f][k] > 0} for f in FOLDS})
    a, b = res['plain A (incumbent, recomputed)'], res['synthetic s46 in recipe A']
    out[f'folds_{tag}'] = dict(incumbent=round(a['cv'], 4), swapped=round(b['cv'], 4), delta=round(b['cv'] - a['cv'], 4), per_fold_delta={f: round(b['per_fold'][f] - a['per_fold'][f], 4) for f in FOLDS}, weights=dict(plain=a['weights'], swapped=b['weights']))
    print(json.dumps({f'folds_{tag}': out[f'folds_{tag}']}), flush=True)
json.dump(out, open(LANE / f'out/cv_swapA_read{"_dry" if DRY else ""}_20261111.json', 'w'), indent=1, default=float); print('done', flush=True)
