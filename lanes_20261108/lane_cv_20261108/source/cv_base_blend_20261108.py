"""CV lane: is 0.5 recipe A + 0.5 Phase-1 the right base for CV? Base CV = blend of A and P1 (geometric) at weight w on P1, then the best
pipeline's CV correction (graph + simulation) and the stretch. w chosen nested (on the other four folds) and also reported per w."""
import sys, json
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
E = BP.E; b = E.load_base(); A, P1, B, y, fold = b['A'], b['P1'], b['B'], BP.y, BP.fold
print('B arithmetic?', np.abs(B - 0.5 * (A + P1)).max(), 'geometric?', np.abs(B - np.sqrt(A * P1)).max(), flush=True)
def pipe_with_base_cv(cv):
    base = BP.H.copy(); base[:, 1] = cv; return BP.pipe(BP.BEST, base)
res = {}
for w in np.round(np.arange(0.0, 1.01, 0.1), 2):
    cv = np.exp((1 - w) * np.log(A[:, 1]) + w * np.log(P1[:, 1])); p = pipe_with_base_cv(cv); a = E.ape(p, y)[:, 1]
    res[w] = {f: float(np.nanmean(a[fold == f])) for f in range(5)}; print(w, 'CV all', round(float(np.nanmean(a)), 4), 'folds', [round(res[w][f], 3) for f in range(5)], flush=True)
ref = E.ape(BP.REF_BEST, y)[:, 1]; print('REF_BEST CV', round(float(np.nanmean(ref)), 4))
# nested choice: w minimising CV over the other folds (row-weighted)
n = {f: int(np.sum((fold == f) & np.isfinite(ref))) for f in range(5)}; P = np.full(len(y), np.nan); ch = {}
for f in range(5):
    wbest = min(res, key=lambda w: sum(res[w][g] * n[g] for g in range(5) if g != f)); ch[f] = float(wbest)
    cv = np.exp((1 - wbest) * np.log(A[:, 1]) + wbest * np.log(P1[:, 1])); P[fold == f] = E.ape(pipe_with_base_cv(cv), y)[fold == f, 1]
print('nested w per fold', ch, 'CV all', round(float(np.nanmean(P)), 4), 'vs', round(float(np.nanmean(ref)), 4), '| folds 0+4', round(float(np.nanmean(P[np.isin(fold, [0, 4])])), 4), 'vs', round(float(np.nanmean(ref[np.isin(fold, [0, 4])])), 4))
