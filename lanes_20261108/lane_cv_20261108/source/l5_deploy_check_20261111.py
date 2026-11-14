"""CV lane L5 check (2026-11-11, coordinator; ledger lesson L5): the nested stack code run with no outer fold must reproduce
deploy/cv_deploy_params.json (D-CV-001, stack file out/cv_final_20261110.npz, five-fold OOF CV 10.1101 as computed / 10.1160 strict).
  1 weights   (a) the as-computed stacker's objective (cv_stack5_20261108.fit: BP.pipe, fold-g stretch on folds != g) on all rows;
              (b) the strict code's objective (cv_search_20261111.score_all, CV-column stretch) on all rows; both from zero, grid 0.125, 2 passes.
  2 inputs    stack component vectors (out/cv_final_20261110_comps.npz) vs the pool the strict reads use (out/pool_20261111.npz).
  3 stretch   the nested code's stretch fit, no outer fold: base and graph parts from the all-rows models (in-sample, as the fold-f models
              are in-sample on the rows their outer stretch is fit on), member terms out-of-fold, all good rows. Plus the per-fold analogue
              vs deploy/cv_inputs_oof.npz stretch{f}.
  4 ridge     the deploy ridge recipe (cv_deploy_fit solve, lambda by leave-one-fold-out) run with outer fold f vs BEST[f] - GRAPH[f].
  5 scalars   the deploy recipe A scalar recipe run with outer fold f vs recipeA_lofo_scalars[f].
Writes out/l5_deploy_check_20261111.json."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import stretch_nested_20261103 as N
import ceiling_meta_cpu_20261103 as C
X = Path('/workspace/clear-ec-research/experiments'); LANE = X / 'lane_cv_20261108'; DEP = LANE / 'deploy'
sys.path.insert(0, str(LANE / 'source')); import cv_apply_20261110 as CA; import cv_deploy_fit_20261110 as DF
y3 = BP.y; y = y3[:, 1]; fold = BP.fold; good = BP.good; ok = y > 0; H = BP.H; E = BP.E
PRM = json.loads((DEP / 'cv_deploy_params.json').read_text()); D = np.load(DEP / 'cv_deploy_insample_members.npz')
o = np.load(LANE / 'out/cv_final_20261110.npz', allow_pickle=True); cz = np.load(LANE / 'out/cv_final_20261110_comps.npz', allow_pickle=True)
raw_names = [str(n) for n in o['names']]; names = [CA.canon(n) for n in raw_names]; K = len(names)
COMP = np.stack([np.stack([cz[f'c{k}_f{f}'] for f in range(5)]) for k in range(K)]); GRID = np.round(np.arange(0, 1.0001, 0.125), 3)
out = {'deploy_params_md5_note': 'deploy/cv_deploy_params.json as read', 'components': names, 'deploy_weights': [PRM['weights'][n] for n in names], 'stack_w_all': o['w_all'].tolist()}


def coord(obj, tol):
    w = np.zeros(K); cur = obj(w)
    for _ in range(2):
        for k in range(K):
            best = (cur, w[k])
            for g in GRID:
                if g == w[k]: continue
                t = w.copy(); t[k] = g; s = obj(t)
                if s < best[0] - tol: best = (s, g)
            w[k] = best[1]; cur = best[0]
    return w, cur


# 1a as-computed stacker objective, all rows
def stack_obj(w):
    c = {f: np.zeros((len(y), 3)) for f in range(5)}
    for f in range(5): c[f][:, 1] = np.tensordot(w, COMP[:, f, :], axes=1)
    return float(np.nanmean(E.ape(BP.pipe(BP.add(BP.BEST, c)), y3)[:, 1]))
wa, sa = coord(stack_obj, 1e-6); out['1a_stacker_no_outer_fold'] = dict(weights=wa.tolist(), objective=round(sa, 4), equals_deploy=bool(np.allclose(wa, out['deploy_weights'])))
print(json.dumps({'1a': out['1a_stacker_no_outer_fold']}), flush=True)
# 1b strict code objective, all rows
LB = np.stack([np.log(H[:, 1]) + BP.BEST[f][:, 1] for f in range(5)])
def stretch_fit(lp, rows):
    r = rows & ok; lb = lp[r]; mu = lb.mean(); yy = y[r]; best = None
    for g in N.GRID:
        q = np.exp(g * (lb - mu) + mu); c = C.wmed(yy / q, q / yy); m = np.mean(np.abs(c * q - yy) / yy)
        if best is None or m < best[0]: best = (m, g, c, mu)
    return best[1:]
def apply_st(lp, prm): g, c, mu = prm; return c * np.exp(g * (lp - mu) + mu)
def score_all(w):
    tot = 0.0; n = 0
    for g in range(5):
        lp = LB[g] + np.tensordot(w, COMP[:, g, :], axes=1); prm = stretch_fit(lp, (fold != g) & good); sel = (fold == g) & ok
        tot += np.sum(np.abs(apply_st(lp[sel], prm) - y[sel]) / y[sel]); n += sel.sum()
    return tot / n * 100
wb, sb = coord(score_all, 1e-9); out['1b_strict_no_outer_fold'] = dict(weights=wb.tolist(), objective=round(sb, 4), equals_deploy=bool(np.allclose(wb, out['deploy_weights'])))
print(json.dumps({'1b': out['1b_strict_no_outer_fold']}), flush=True)
# 2 inputs: stack comps vs pool
pool = np.load(LANE / 'out/pool_20261111.npz', allow_pickle=True); pn = [str(n) for n in pool['names']]
out['2_inputs_max_abs_diff_vs_pool'] = {names[k]: float(max(np.max(np.abs(cz[f'c{k}_f{f}'] - pool[f'c{pn.index(raw_names[k])}_f{f}'])) for f in range(5))) for k in range(K)}
print(json.dumps({'2': out['2_inputs_max_abs_diff_vs_pool']}), flush=True)
# 3 stretch, no outer fold: all-rows model column
G, RID, QP, QO = D['G'][:, 1], D['RID'][:, 1], D['QP'][:, 1], D['QO'][:, 1]; ins = {'graph:plain': QP - G, 'graph:order': QO - G}
for k, n in enumerate(names):
    if n not in ins: assert all(np.array_equal(COMP[k, 0], COMP[k, f]) for f in range(5)), n
w_dep = np.array(out['deploy_weights']); lp99 = np.log(H[:, 1]) + G + RID + sum(w_dep[k] * (ins[n] if n in ins else COMP[k, 0]) for k, n in enumerate(names))
st_nested = [float(v) for v in stretch_fit(lp99, good)]; C3 = H.copy(); C3[:, 1] = np.exp(lp99); st_N = [float(v) for v in N.fit_stretch(C3, y3, good)[1]]
out['3_stretch'] = dict(deploy=PRM['stretch_cv'], nested_code_no_outer_fold=st_nested, stretch_nested_fit_stretch=st_N, max_abs_diff=float(np.max(np.abs(np.array(st_nested) - np.array(PRM['stretch_cv'])))),
                        cor_matches_deploy_insample=float(np.max(np.abs((lp99 - np.log(H[:, 1])) - D['cor_cv']))))
Z = np.load(DEP / 'cv_inputs_oof.npz'); pf = {}
for f in range(5):
    wf = o[f'w{f}'].astype(float); lp = LB[f] + np.tensordot(wf, COMP[:, f, :], axes=1); s = np.array(stretch_fit(lp, (fold != f) & good)); pf[f] = float(np.max(np.abs(s - Z[f'stretch{f}'])))
out['3_stretch']['per_fold_max_abs_diff_vs_cv_inputs_oof'] = pf
print(json.dumps({'3': out['3_stretch']}), flush=True)
# 4 ridge with an outer fold
EMB = DF.emb(); MASK = DF.MASK; L = json.loads((DEP / 'cv_ridge_lambda.json').read_text()); r4 = {}
def ridge_pred(tr, base, lam):
    pr = np.zeros(len(y))
    for Zm in EMB:
        okz = np.isfinite(Zm).all(1); Zf = np.where(okz[:, None], Zm, 0); mu, sd, w, t0 = DF.solve(Zf, okz, tr, base, 1, lam); pr += np.where(okz, ((Zf - mu) / sd) @ w + t0, 0.0) / len(EMB)
    return pr
for f in range(5):
    tgt = BP.BEST[f][:, 1] - BP.GRAPH[f][:, 1]; base_f = H * np.exp(BP.GRAPH[f] * MASK); diffs = {}
    for lam in DF.LAMS: diffs[str(int(lam))] = float(np.max(np.abs(ridge_pred((fold != f) & good, base_f, lam) - tgt)))
    sc = {}
    for lam in DF.LAMS:
        e = 0.0
        for q in range(5):
            if q == f: continue
            pr = ridge_pred((fold != q) & (fold != f) & good, H * np.exp(BP.GRAPH[q] * MASK), lam); m = (fold == q) & ok; e += float((np.abs(H[m, 1] * np.exp(BP.GRAPH[q][m, 1]) * np.exp(pr[m]) - y[m]) / y[m]).sum())
        sc[lam] = e
    r4[f] = dict(lofo_lambda=float(min(sc, key=sc.get)), max_abs_diff_by_lambda=diffs, best_matching_lambda=min(diffs, key=diffs.get))
out['4_ridge_outer_fold'] = dict(deploy_lambda_cv=L['CV']['lam'], per_fold=r4)
print(json.dumps({'4': out['4_ridge_outer_fold']}), flush=True)
# 5 recipe A scalars with an outer fold
hyb = json.loads((X / 'recipeA_analysis_20261028/hybrid_oof.json').read_text()); A = BP._b['A']; raw = np.empty_like(A)
for f in range(5): raw[fold == f] = A[fold == f] / np.array(hyb['recipeA_lofo_scalars'][str(f)])
def wmed(v, w): o_ = np.argsort(v); c = np.cumsum(w[o_]); return float(v[o_][np.searchsorted(c, c[-1] / 2.)])
r5 = {}
for f in range(5):
    m = (fold != f) & (y3[:, 1] > 0); r5[f] = dict(deploy_recipe=wmed(y3[m, 1] / raw[m, 1], raw[m, 1] / y3[m, 1]), stored_lofo=float(hyb['recipeA_lofo_scalars'][str(f)][1]))
m = y3[:, 1] > 0; out['5_recipeA_scalar_cv'] = dict(per_fold=r5, all_rows_recipe=wmed(y3[m, 1] / raw[m, 1], raw[m, 1] / y3[m, 1]), deploy=PRM['recipeA_scalar_cv'])
print(json.dumps({'5': out['5_recipeA_scalar_cv']}), flush=True)
json.dump(out, open(LANE / 'out/l5_deploy_check_20261111.json', 'w'), indent=1, default=float); print('done', flush=True)
