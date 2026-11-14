"""Does a frozen crop-SSL read add to the best pipeline? (lane GPU 1, 2026-11-07). The end-to-end members land on the recipe A solution (residual r 0.98-0.99), so test the frozen ridge prediction itself as a member: log correction w * (log frozen - log recipe A TTA), rows of folds 0 and 4 only, stretch fit as in best_pipeline. Ridge is fit on the fold's train rows only.
No GPU."""
import json, sys
import numpy as np
sys.path.insert(0, 'experiments/lane_gpu1_20261106/source'); sys.path.insert(0, 'experiments/meta_ceiling_20261103/source')
import frozen_read_20261106 as F
import best_pipeline_20261106 as BP
E = F.E; K = F.K; A = BP._b['A']
assert np.array_equal(BP.idx, F.B['idx'])
out = []
for arms, tag in ((('cropopt_both_f4', 'cropopt_both_f0'), 'crop_optics_both'), (('ssl5_f4', 'ssl5_f0'), 'ssl5_reference')):
    for pool in ('both', 'std'):
        rows_all, lg = [], []
        for f, a in zip((4, 0), arms):
            p, rows, _ = F.read(a, f, pool); rows_all.append(rows); lg.append(np.log(np.maximum(p, 1e-6)))
        rows = np.concatenate(rows_all); lg = np.concatenate(lg)
        res = np.log(BP.y[rows].clip(1e-6)) - np.log(A[rows]); r2 = np.log(BP.y[rows].clip(1e-6)) - lg
        ok = (BP.y[rows] > 0).all(1)
        rr = [round(float(np.corrcoef(res[ok, k], r2[ok, k])[0, 1]), 3) for k in range(3)]
        for w in (0.125, 0.25, 0.5):
            c = np.zeros_like(BP.H); c[rows] = w * (lg - np.log(A[rows]))
            P = BP.pipe(BP.add(BP.BEST, {ff: c for ff in range(5)}))
            eb, er = E.ape(P[rows], BP.y[rows]), E.ape(BP.REF_BEST[rows], BP.y[rows])
            pb = E.paired_bootstrap(eb, er, np.unique(BP.donors[rows], return_inverse=True)[1]); s1, s0 = E.official(eb), E.official(er)
            r = dict(arm=tag, pool=pool, w=w, delta=round(pb['delta'], 4), ci95=[round(v, 4) for v in pb['ci95']], per={k: round(s1[k] - s0[k], 4) for k in K}, residual_r_with_recipeA=rr, best_on_rows=round(s0['mean'], 4))
            print(json.dumps(r), flush=True); out.append(r)
open('experiments/lane_gpu1_20261106/reads/frozen_added_two_fold.json', 'w').write(json.dumps(out, indent=1))
