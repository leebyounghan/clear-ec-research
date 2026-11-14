"""Distillation teachers for the backbone lane (2026-11-08), npz {indices (cache rows), pred (N,3)}:
best pipeline OOF (REF_BEST) and hybrid B OOF. Every row's prediction comes from fold models that did not see that row,
but those models did see the other folds: a student trained on fold k's training rows gets targets shaped partly by
fold k's labels (second-order leak, see REPORT). Also reports coverage of each split's training rows."""
import sys, numpy as np
from pathlib import Path
X = Path('/workspace/clear-ec-research/experiments'); out = X / 'lane_backbone_20261108/data'; out.mkdir(exist_ok=True)
sys.path.insert(0, str(X / 'meta_ceiling_20261103/source'))
import best_pipeline_20261106 as BP
for name, P in (('teacher_best_pipeline', BP.REF_BEST), ('teacher_hybridB', BP.B)):
    ok = np.isfinite(P).all(1) & (P > 0).all(1)
    np.savez_compressed(out / f'{name}_20261108.npz', indices=BP.idx[ok], pred=P[ok])
    print(name, int(ok.sum()), 'rows; MAPE vs labels', [round(float(np.nanmean(np.abs(P[ok, j] / BP.y[ok, j] - 1) * 100)), 3) for j in range(3)])
sys.path.insert(0, str(X / 'lane_backbone_20261108/source'))
cov = set(BP.idx.tolist())
for f, p in {0: 'experiments/ssl_screen_20260911/preflight/fold0.npz', 4: 'experiments/target_fold4_20260912/preflight/fold4.npz'}.items():
    with np.load('/workspace/clear-ec-research/' + p) as z:
        tr = z['train']
    print('fold', f, 'train rows', len(tr), 'missing from teacher', [int(i) for i in tr if int(i) not in cov][:10])
