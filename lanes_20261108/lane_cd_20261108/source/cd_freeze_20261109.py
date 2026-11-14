"""Build the CD freeze candidate files (2026-11-09): nested five-fold OOF with all member inputs, per-fold params (for reproduction) and
deploy params (same search on all five folds). Recipe = cd_stack_20261109 with members DINOv3 s46 TTA + ViT-L s46 (tt) and the seg_loose mask."""
import os, json, itertools, numpy as np
os.environ['A2SNAP'] = '1'
import cd_stack_20261109 as K
import cd_read_20261108 as R
from cd_common import *
import cd_combine_20261110 as CB
D = R.member('dinov3_s46_tta'); V = R.member(os.environ.get('VKEY', 'vitl_s4647_tt')); P1 = R.P1; A = R.A2
mask_cd = np.where(np.isfinite(K.F[:, K.nm['cd_mm2']]), K.F[:, K.nm['cd_mm2']], np.nan); mask_n = K.F[:, K.nm['n_int']]
assert np.isfinite(D).all() and np.isfinite(V).all() and np.isfinite(A).all() and np.isfinite(P1).all()
Lc = [np.log(P1), np.log(A), np.log(D), np.log(V)]; ly = K.ly; have = K.have; lm = K.lm
grids = []
for a in np.round(np.arange(0.25, 0.751, 0.05), 3):
    for mw in itertools.product(np.round(np.arange(0, 0.401, 0.05), 3), repeat=2):
        r = 1 - a - sum(mw)
        if r >= 0.099: grids.append((round(r, 3), a) + tuple(mw))
def fit(tr):
    best = None
    for wt in grids:
        q = np.exp(sum(w * l for w, l in zip(wt, Lc))); e = cd(apply1(q, fit_stretch1(q, tr & good)), tr)
        if best is None or e < best[0]: best = (e, wt, q)
    _, wt, q = best; trm = tr & have & (yc > 0) & good; co = np.polyfit(lm[trm], ly[trm], 1); Mc = np.polyval(co, lm); b2 = None
    for wmk in (0, 0.05, 0.1, 0.15, 0.2, 0.25):
        qq = np.where(have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
        if b2 is None or e < b2[0]: b2 = (e, wmk, p2)
    _, wmk, (g, c, mu) = b2
    return dict(weights=dict(P1=float(wt[0]), A=float(wt[1]), D=float(wt[2]), V=float(wt[3])), mask_min_cells=50, mask_a=float(co[1]), mask_b=float(co[0]),
                mask_w=float(wmk), stretch_g=float(g), stretch_c=float(c), stretch_mu=float(mu))
per = {}; oof = np.full(len(yc), np.nan)
for f in range(5):
    per[str(f)] = fit(fold != f); m = fold == f
    oof[m] = CB.combine(P1[m], A[m], D[m], V[m], mask_cd[m], mask_n[m], per[str(f)])
v = cd(oof); print('five-fold nested CD', round(v, 4), 'baseline', round(cd(REF), 4), flush=True)
rep('freeze candidate', oof, REF, rows=yc > 0)
dep = fit(np.ones(len(yc), bool)); print('deploy params', json.dumps(dep), flush=True)
sc_fold = json.load(open(X / 'recipeA_analysis_20261028/hybrid_oof.json'))['recipeA_lofo_scalars']
out = dict(created='2026-11-09', metric='CD', five_fold_cd=round(v, 4), baseline_cd=round(cd(REF), 4), baseline='best pipeline (best_pipeline_20261106 REF_BEST, CD column)',
           per_fold=per, deploy=dep, recipeA_cd_scalar_per_fold={k: vv[0] for k, vv in sc_fold.items()},
           recipeA_cd_scalar_deploy=float(np.mean([vv[0] for vv in sc_fold.values()])),
           note='OOF A column already includes the per-fold recipe A scalar; at deploy multiply the full-data recipe A seed mean by recipeA_cd_scalar_deploy.')
os.makedirs(LANE / 'reads', exist_ok=True)
json.dump(out, open(LANE / 'reads/freeze_cd_params_20261109.json', 'w'), indent=1)
np.savez_compressed(LANE / 'reads/freeze_cd_oof_20261109.npz', idx=idx, ids=ids, fold=fold, y_cd=yc, P1=P1, A=A, D=D, V=V, mask_cd=mask_cd, mask_n_int=mask_n, oof_cd=oof, ref_cd=REF)
print('saved', flush=True)
