"""Cell-route lane (2026-11-11, coordinator: how strictly nested is the cell-route step?). Strict re-read of the HEX step on a HEX base.
Nested in the kept step already: standardisation, ridge weights, scales c / c0 (folds != f). Fixed in advance, never tuned: alpha 100, >= 50 cells,
the feature-construction thresholds. NOT nested in the kept step: the choice of the feature version (and of keeping a step at all), made on five-fold
all-label reads of 15 HEX versions (corr.py / corr2.py / corr3.py, standalone on the lane best). Here that choice is nested: for each outer fold f,
each candidate (the 15 versions + 'none' = keep the base) is scored by inner 4-fold CV on the training folds only (fit on folds not in {f, g},
score fold g); the best candidate is refit on all folds != f and applied to fold f. Scored on all labelled rows and, separately, without the 25 expert-
marker gate frames whose labels pre-registered the point-set versions. usage: corr_strict.py BASE.npz [OUTNAME]
(this copy: l5_strict_pick.py BASE.npz, the version choice with NO outer fold, for lesson L5)"""
import sys, json
import numpy as np
R = '/workspace/clear-ec-research/experiments/'; L = R + 'lane_cellroute_20261109/'
sys.path[:0] = [R + 'meta_ceiling_20261103/source', L + 'source', R + 'lane_cv_20261108/source']
import best_pipeline_20261106 as BP
import ceiling_meta_cpu_20261103 as C
y, fold, good, idx = BP.y[:, 2], BP.fold, BP.good, BP.idx.astype(int); ok = y > 0
zb = np.load(sys.argv[1]); base = zb['final_hex']; assert (zb['idx'].astype(int) == idx).all() and (zb['fold'] == fold).all()
lg = lambda a: np.log(np.maximum(a, 1e-4))
z = np.load(L + 'feats/cellroute_feats.npz'); F1, N1 = z['F'][idx], list(z['names'])
z = np.load(L + 'feats/trim_feats.npz'); F3, N3 = z['F'][idx], list(z['names'])
z = np.load(L + 'feats/click_feats_P2_P5_m1_K8.npz'); F2, N2 = z['F'][idx], list(z['names'])
def five(F, N, p):
    g = lambda c: F[:, N.index(f'{p}_{c}')]; return [lg(g('cv_k0')), lg(g('cv_k2')), g('hex_k0'), g('hex_k2'), np.log1p(np.nan_to_num(g('n')))]
def sets(ss, regs):          # corr.py design: rows need >= 50 cells in the first set's full frame
    cols = [c for s in ss for rg in regs if f'{s}_{rg}_cv_k2' in N1 for c in five(F1, N1, f'{s}_{rg}')]
    return np.column_stack(cols), np.nan_to_num(F1[:, N1.index(f'{ss[0]}_full_n')]) >= 50
def blocks(bl):              # corr2.py / corr3.py: rows need >= 50 cells in every block
    Xs, m = [], np.ones(len(y), bool)
    for b in bl:
        if b == 'P2raw': X = np.column_stack(five(F1, N1, 'P2_win') + five(F1, N1, 'P2_full')); n = np.nan_to_num(F1[:, N1.index('P2_full_n')])
        elif b in ('P2c', 'P5c'):
            g = lambda c: F2[:, N2.index(f'{b}_win_{c}')]; X = np.column_stack(five(F2, N2, f'{b}_win') + [g('s5_k2'), g('s7_k2')]); n = np.nan_to_num(g('n'))
        else: X = np.column_stack(five(F3, N3, b)); n = np.nan_to_num(F3[:, N3.index(f'{b}_n')])
        Xs.append(X); m &= np.isfinite(X).all(1) & (n >= 50)
    return np.column_stack(Xs), m
CAND = {'P2 win+full (kept)': sets(['P2'], ['win', 'full']), 'P0 win+full': sets(['P0'], ['win', 'full']), 'P1 win+full': sets(['P1'], ['win', 'full']),
        'P3 win+full': sets(['P3'], ['win', 'full']), 'P2 win+full+creg': sets(['P2'], ['win', 'full', 'creg']), 'P0-P3 win+full': sets(['P0', 'P1', 'P2', 'P3'], ['win', 'full']),
        'P2raw+P2c+P5c': blocks(['P2raw', 'P2c', 'P5c']), 'P2c': blocks(['P2c']), 'P5c': blocks(['P5c']), 'P2raw+P2c': blocks(['P2raw', 'P2c']), 'P2raw+P5c': blocks(['P2raw', 'P5c']),
        'P2 box raw': blocks(['P2_box_raw']), 'P2 p150 raw': blocks(['P2_p150_raw']), 'P2 box trim': blocks(['P2_box_trim']), 'P2raw+P2 box raw': blocks(['P2raw', 'P2_box_raw'])}
res = np.log(np.where(ok, y, 1.0)) - np.log(base)
def design(name):
    X, m = CAND[name]; X = np.c_[X, np.log(base)]; m = m & np.isfinite(X).all(1); return np.where(m[:, None], X, 0.0), m
DES = {k: design(k) for k in CAND}
def fitpred(name, trn, app):
    """corr_newbase recipe: fit on trn (good & usable & labelled), predict rows in app (usable rows corrected, others keep the base)."""
    P = base.copy()
    if name == 'none': return P
    X, m = DES[name]; tr = trn & good & m & ok; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; A = (X[tr] - mu) / sd
    w = np.linalg.solve(A.T @ A + 100.0 * np.eye(A.shape[1]), A.T @ (res[tr] - res[tr].mean())); q = base * np.exp(((X - mu) / sd) @ w)
    c = C.wmed(y[tr] / q[tr], q[tr] / y[tr]); c0 = C.wmed(y[tr] / base[tr], base[tr] / y[tr]); s = app & m; P[s] = q[s] * c / c0; return P
ape = lambda p, r: float(np.mean(np.abs(p[r & ok] - y[r & ok]) / y[r & ok]) * 100)
names = ['none'] + list(CAND); allr = np.ones(len(y), bool)
# L5 for the strict procedure: run its version choice with no outer fold = inner CV over all five folds (fit on folds != g, score fold g), same
# candidates, same fit, same base as the deploy fit; the pick must be the deployed kept version for deploy = strict procedure with no outer fold.
sc = {}
for nm in names:
    Pin = base.copy()
    for g in range(5): Pin[fold == g] = fitpred(nm, fold != g, fold == g)[fold == g]
    sc[nm] = round(ape(Pin, allr), 5)
rank = sorted(sc.items(), key=lambda kv: kv[1]); pick = rank[0][0]
out = dict(base=sys.argv[1].split('/')[-1], pick=pick, deployed='P2 win+full (kept)', pick_equals_deployed=pick == 'P2 win+full (kept)', top5=rank[:5], none=sc['none'],
           gap_to_second=round(rank[1][1] - rank[0][1], 5))
print(json.dumps(out), flush=True); json.dump(out, open(R + "lane_cd_20261108/reads/verify_cellroute_pick_20261111.json", "w"), indent=1)
