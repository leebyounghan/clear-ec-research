"""Cell-route lane for the backbone lane (2026-11-09): teacher tables per outer fold f, npz {indices (cache rows), pred (N,3)} covering every row
of folds != f, the format bb_runner --teacher reads.
Option A ('stack refit without f', CPU): best-pipeline recipe with every stack-level part refit without fold f AND without the row's own fold g:
  hex6 blend a, b refit by OLS of log HEX on the hex6 feature on folds not in {f, g} (w kept 0.28); graph+simulation correction = BP.BEST[f]
  (all-row correction from models fit without fold f); stretch fit on folds not in {f, g}. Member predictions (hybrid B = recipe A + Phase-1 OOF)
  are used as stored: a fold-g row's member prediction comes from the fold-g model, which trained on fold f (second-order leak, not removable
  from stored predictions). For the hybrid-B teacher nothing is refittable, so it equals the existing OOF table.
Also writes the plain OOF best-pipeline table per fold (the current teacher) for comparison, and prints how far A moves from it."""
import os, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import stretch_nested_20261103 as N
OUT = '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/teachers/'; os.makedirs(OUT, exist_ok=True)
y, B, fold, idx, good, f6, have = BP.y, BP.B, BP.fold, BP.idx.astype(int), BP.good, BP.f6, BP._have
lg = np.log
for f in range(5):
    T = np.full_like(B, np.nan)
    for g in [k for k in range(5) if k != f]:
        inner = (fold != f) & (fold != g) & good; m = inner & have & (y[:, 2] > 0)
        b_, a_ = np.polyfit(f6[m], lg(y[m, 2]), 1)  # log HEX ~ a + b f6
        H = B.copy(); H[have, 2] = np.exp((1 - 0.28) * lg(B[have, 2]) + 0.28 * (a_ + b_ * f6[have]))
        Cc = H * np.exp(BP.BEST[f]); st = N.fit_stretch(Cc, y, inner); sel = fold == g; T[sel] = N.apply_stretch(Cc[sel], st)
    rows = (fold != f) & np.isfinite(T).all(1) & (T > 0).all(1)
    np.savez_compressed(OUT + f'teacher_A_stackrefit_f{f}.npz', indices=idx[rows], pred=T[rows], note=__doc__)
    R = BP.REF_BEST; rr = (fold != f) & np.isfinite(R).all(1) & (R > 0).all(1)
    np.savez_compressed(OUT + f'teacher_oof_bestpipe_f{f}.npz', indices=idx[rr], pred=R[rr], note='plain OOF best pipeline (current teacher), rows of folds != f')
    mape = lambda P, r: [round(float(np.nanmean(np.abs(P[r, j] / y[r, j] - 1)[y[r, j] > 0]) * 100), 3) for j in range(3)]
    d = np.abs(lg(T[rows]) - lg(R[rows])).mean(0) * 100
    print(f'outer fold {f}: rows {int(rows.sum())} (of {int((fold != f).sum())}); MAPE vs label on these rows A {mape(T, rows)} | OOF {mape(R, rows)}; mean |log A - log OOF| x100 {d.round(3)}', flush=True)
