"""Teacher file for the ensemble-distillation student (coordinator 11-11, KG idea-ensemble-distillation): the integrated candidate's
OOF predictions (integration_rehearsal_20261110/out/integrated_oof.npz, P_main) as npz {indices (cache rows), pred (N,3)} for
bb_runner --teacher (target = exp(0.5 log y + 0.5 log t)). Caveat: a row's teacher value comes from models that saw the other folds'
labels, so the student's held-out-fold read is slightly optimistic."""
import numpy as np
X = '/workspace/clear-ec-research/experiments/'
z = np.load(X + 'integration_rehearsal_20261110/out/integrated_oof.npz', allow_pickle=True)
idx, p = z['idx'].astype(np.int64), z['P_main'].astype(np.float64)
assert len(np.unique(idx)) == len(idx) and np.isfinite(p).all() and (p > 0).all()
np.savez_compressed(X + 'lane_backbone_20261108/data/teacher_integrated_20261111.npz', indices=idx, pred=p.astype(np.float32))
for f in (0, 4):
    sp = np.load(X + ('ssl_screen_20260911/preflight/fold%d.npz' % f if f == 0 else 'target_fold4_20260912/preflight/fold4.npz'))
    tr = sp['train'].astype(int); print(f, 'train rows', len(tr), 'covered', int(np.isin(tr, idx).sum()))
print('rows', len(idx), 'pred min', p.min(0).round(3).tolist(), 'max', p.max(0).round(3).tolist())
