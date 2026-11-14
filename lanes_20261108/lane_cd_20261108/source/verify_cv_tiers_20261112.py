"""Verifier (D-PKG-002): CV lane tier sets. Independent of the CV lane's code paths (reads only its saved files and BP).
CFG env = incumbent | t1 | t3_e20 | t3_nodino. Checks:
 a. DINOv3 term rebuilt here from the fold runs' single-view predictions (seed 46, epochs per tier) vs the comps file's 'dino' column;
    every other component vs the pool used by the freeze (out/pool_20261111.npz).
 b. strict five folds re-run here: per outer fold, coordinate descent (grid 0..1 step 0.125, two passes, from zero) on the inner objective
    (inner fold g stretch fit on folds not in {f, g}), outer stretch on folds != f; strict CV vs the stack file and the stated value; weights per fold.
 c. L5: the same descent with no outer fold (fold g stretch on folds != g) vs the deploy folder's weights; deploy stretch refit here from the
    folder's in-sample members with those weights vs the stored stretch; per-fold stretches vs cv_inputs_oof; cv_final_oof vs the stack file.
Writes lane_cd_20261108/reads/verify_cv_tier_<CFG>_20261112.json."""
import os, sys, json, numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import stretch_nested_20261103 as N
import ceiling_meta_cpu_20261103 as C
from pathlib import Path
X = Path('/workspace/clear-ec-research/experiments'); CV = X / 'lane_cv_20261108'; CFG = os.environ['CFG']
DIR = {'incumbent': None, 't1': 'deploy_t1', 't3_e20': 'deploy_t3', 't3_nodino': 'deploy_t3_alt'}[CFG]
EPO = {'incumbent': None, 't1': range(16, 21), 't3_e20': (20,), 't3_nodino': None}[CFG]
y = BP.y[:, 1].astype(float); fold = BP.fold; good = BP.good; ok = y > 0; A = BP._b['A'][:, 1]; pos = {int(i): n for n, i in enumerate(BP.idx)}
st = np.load(CV / f'out/cv_tier_{CFG}_20261112.npz', allow_pickle=True); cz = np.load(CV / f'out/cv_tier_{CFG}_20261112_comps.npz', allow_pickle=True)
names = [str(n) for n in cz['names']]; K = len(names); COMP = np.stack([np.stack([cz[f'c{k}_f{f}'] for f in range(5)]) for k in range(K)])
out = {'config': CFG, 'components': names}
# a. inputs
if EPO is not None:
    v = np.full(len(y), np.nan)
    for f in range(5):
        run = (X / f'lane_gpu3_20261106/runs/dinov3_b16_f{f}_s46' if f in (0, 4) else X / f'lane_cd_20261108/arms/dinov3_b16_f{f}_s46') / f'runs/fold{f}/dinov3_b16'
        L = []
        for e in EPO:
            z = np.load(run / f'predictions_epoch_{e:02d}.npz'); p = np.full(len(y), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred'][:, 1]; L.append(np.log(p))
        m = np.mean(L, 0) - np.log(A); v[fold == f] = m[fold == f]
    k = names.index('dino'); out['a_dino_rebuilt_vs_comps_max_abs'] = float(max(np.nanmax(np.abs(v - COMP[k, f])) for f in range(5)))
pool = np.load(CV / 'out/pool_20261111.npz', allow_pickle=True); pn = [str(n) for n in pool['names']]
out['a_other_components_vs_freeze_pool_max_abs'] = {n: float(max(np.max(np.abs(COMP[k, f] - pool[f'c{pn.index(n)}_f{f}'])) for f in range(5))) for k, n in enumerate(names) if n != 'dino' or CFG == 'incumbent'}
# b. strict re-run
LB = np.stack([np.log(BP.H[:, 1]) + BP.BEST[f][:, 1] for f in range(5)]); GR = np.round(np.arange(0, 1.0001, 0.125), 3)


def sfit(lp, rows):
    r = rows & ok; lb = lp[r]; mu = lb.mean(); yy = y[r]; best = None
    for g in N.GRID:
        q = np.exp(g * (lb - mu) + mu); c = C.wmed(yy / q, q / yy); m = np.mean(np.abs(c * q - yy) / yy)
        if best is None or m < best[0]: best = (m, g, c, mu)
    return best[1:]


def sap(lp, p): return p[1] * np.exp(p[0] * (lp - p[2]) + p[2])
def lpf(g, w): return LB[g] + np.tensordot(w, COMP[:, g, :], axes=1)


def obj(w, f):
    num = 0.0; den = 0
    for g in range(5):
        if g == f: continue
        lp = lpf(g, w); trn = (fold != g) & good if f is None else (fold != g) & (fold != f) & good; s = (fold == g) & ok
        num += np.sum(np.abs(sap(lp[s], sfit(lp, trn)) - y[s]) / y[s]); den += s.sum()
    return num / den * 100


def descend(f):
    w = np.zeros(K); cur = obj(w, f)
    for _ in range(2):
        for k in range(K):
            bv, bw = cur, w[k]
            for gv in GR:
                if gv == w[k]: continue
                t = w.copy(); t[k] = gv; s = obj(t, f)
                if s < bv - 1e-9: bv, bw = s, gv
            w[k] = bw; cur = bv
    return w


P = np.full(len(y), np.nan); W = {}; PST = {}
for f in range(5):
    W[f] = descend(f); lp = lpf(f, W[f]); PST[f] = sfit(lp, (fold != f) & good); P[fold == f] = sap(lp[fold == f], PST[f])
ape = lambda p, r: float(np.mean(np.abs(p[r] - y[r]) / y[r]) * 100)
out['b_strict_cv_here'] = round(ape(P, ok), 5); out['b_strict_cv_stack_file'] = round(ape(st['P'][:, 1], ok), 5)
out['b_P_max_abs_diff_vs_stack_file'] = float(np.nanmax(np.abs(P - st['P'][:, 1])))
out['b_weights_per_fold_equal'] = {f: bool(np.allclose(W[f], st[f'w{f}'])) for f in range(5)}
out['b_per_fold_cv_here'] = {f: round(ape(P, ok & (fold == f)), 4) for f in range(5)}
wall = descend(None); out['c_no_outer_fold_weights_here'] = dict(zip(names, map(float, wall))); out['c_equal_stack_w_all'] = bool(np.allclose(wall, st['w_all']))
if DIR:
    D_ = CV / DIR; PRM = json.load(open(D_ / 'cv_deploy_params.json')); wd = np.array([PRM['weights'][n if not n.startswith('graph:') else ('graph:plain' if 'tta_' in n else 'graph:order')] for n in names])
    out['c_deploy_weights'] = wd.tolist(); out['c_L5_weights_equal_deploy'] = bool(np.allclose(wall, wd))
    M = np.load(D_ / 'cv_deploy_insample_members.npz'); G, RID = M['G'][:, 1], M['RID'][:, 1]
    lp99 = np.log(BP.H[:, 1]) + G + RID
    for k, n in enumerate(names):
        if n.startswith('graph:tta_'): lp99 = lp99 + wd[k] * (M['QP'][:, 1] - G)
        elif n.startswith('graph:gord_'): lp99 = lp99 + wd[k] * (M['QO'][:, 1] - G)
        else:
            assert all(np.array_equal(COMP[k, 0], COMP[k, f]) for f in range(5)), n; lp99 = lp99 + wd[k] * COMP[k, 0]
    s99 = sfit(lp99, good); out['c_L5_stretch_here'] = [float(v) for v in s99]; out['c_L5_stretch_deploy'] = PRM['stretch_cv']
    out['c_L5_stretch_max_abs_diff'] = float(np.max(np.abs(np.array(s99) - np.array(PRM['stretch_cv']))))
    Z = np.load(D_ / 'cv_inputs_oof.npz'); out['c_per_fold_stretch_vs_inputs_max_abs'] = float(max(np.max(np.abs(np.array(PST[f]) - Z[f'stretch{f}'])) for f in range(5)))
    F = np.load(D_ / 'cv_final_oof.npz', allow_pickle=True); out['c_final_oof_vs_P_here_max_abs'] = float(np.nanmax(np.abs(F['P'][:, 1] - P)))
    out['c_final_oof_cv_value'] = round(ape(F['P'][:, 1], ok), 5)
print(json.dumps(out), flush=True); json.dump(out, open(X / f'lane_cd_20261108/reads/verify_cv_tier_{CFG}_20261112.json', 'w'), indent=1)
