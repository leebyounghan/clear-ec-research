"""CV lane L5 check for a reduced deploy tier (2026-11-12, D-PKG-002): the nested stack code run with no outer fold must reproduce the tier's
deploy parameters. Env CV_OOF (stack file in out/, e.g. cv_tier_t1_20261112) and CV_DEPLOY_DIR (e.g. deploy_t1).
  1 weights   (a) the as-computed stacker objective (cv_stack5 fit, BP.pipe) and (b) the strict objective (score_all), both on all rows, from zero
  3 stretch   the nested code's stretch fit with no outer fold (all-rows graph parts in-sample, member terms out-of-fold) vs the stored stretch;
              per-fold stretch vs DIR/cv_inputs_oof.npz
Writes out/l5_<CV_OOF>.json."""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import stretch_nested_20261103 as N
import ceiling_meta_cpu_20261103 as C
X = Path('/workspace/clear-ec-research/experiments'); LANE = X / 'lane_cv_20261108'; DEP = LANE / os.environ['CV_DEPLOY_DIR']; OOFN = os.environ['CV_OOF']
sys.path.insert(0, str(LANE / 'source')); import cv_apply_20261110 as CA; import cv_deploy_fit_20261110 as DF
y3 = BP.y; y = y3[:, 1]; fold = BP.fold; good = BP.good; ok = y > 0; H = BP.H; E = BP.E
PRM = json.loads((DEP / 'cv_deploy_params.json').read_text()); D = np.load(DEP / 'cv_deploy_insample_members.npz')
o = np.load(LANE / f'out/{OOFN}.npz', allow_pickle=True); cz = np.load(LANE / f'out/{OOFN}_comps.npz', allow_pickle=True)
raw_names = [str(n) for n in o['names']]; names = [CA.canon(n) for n in raw_names]; K = len(names)
COMP = np.stack([np.stack([cz[f'c{k}_f{f}'] for f in range(5)]) for k in range(K)]); GRID = np.round(np.arange(0, 1.0001, 0.125), 3)
out = {'deploy_dir': str(DEP), 'stack': OOFN, 'components': names, 'deploy_weights': [PRM['weights'][n] for n in names], 'stack_w_all': o['w_all'].tolist()}


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
json.dump(out, open(LANE / f'out/l5_{OOFN}.json', 'w'), indent=1, default=float); print('done', flush=True)
