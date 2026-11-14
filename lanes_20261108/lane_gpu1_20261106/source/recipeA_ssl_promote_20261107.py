"""Promotion runs from this lane's SSL recipe (GPU 1, 2026-11-07): folds 1, 3, 2 and the all-data fit. recipeA_ssl_arm_20261106 on top of the frozen std-pooling runner, plus the zero-target handling recipe A used for fold 2 and the final fit:
- arrays(): training rows stay strictly positive; the validation rows may hold only the two zero-HEX rows of donor 2019-0064.
- metrics(): the official rule, a zero truth is unscored for that metric, and a column with no scored row reads 0.0, as in recipeA_fulldata_regression_20261028. Without zeros it equals the frozen metric.
- All-data fit: --fold 99, fulldata_regression_split.npz (8952 train, the two zero rows as a placeholder validation). The SSL covered all 8954 frames, so its train count differs on purpose."""
import sys
from pathlib import Path

import numpy as np

import recipeA_ssl_arm_20261106 as A

R, b, prior = A.R, A.R.b, A.prior
ZERO = ['2019-0064-ODCN', '2019-0064-OSCN']
FROZEN_METRICS = b.metrics
_SOURCES = R.source_files


def official_metrics(pred, y):
    if pred.shape != y.shape or not np.isfinite(pred).all() or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Invalid predictions/targets')
    scored = y > 0
    if scored.all():
        return FROZEN_METRICS(pred, y)
    err = np.where(scored, np.abs(pred - y) / np.where(scored, y, 1.), np.nan) * 100
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            v = np.nanmean(err, 0)
    v = np.where(np.isfinite(v), v, 0.0)
    return dict(CD=float(v[0]), CV=float(v[1]), HEX=float(v[2]), mean=float(v.mean()), n=len(y), zero_targets_unscored=dict(zip(('CD', 'CV', 'HEX'), (~scored).sum(0).tolist())))


def arrays():
    with np.load(b.META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    with np.load(R.SPLIT) as z:
        tr, va = z['train'].copy(), z['validation'].copy()
    assert ids.ndim == 1 and len(np.unique(ids)) == len(ids) and y.shape == (len(ids), 3)
    for ix in (tr, va):
        assert ix.ndim == 1 and np.issubdtype(ix.dtype, np.integer) and len(ix) > 0 and len(np.unique(ix)) == len(ix) and ix.min() >= 0 and ix.max() < len(ids)
    assert not np.intersect1d(tr, va).size
    assert np.isfinite(y[np.r_[tr, va]]).all() and (y[tr] > 0).all() and (y[va] >= 0).all()
    zero = sorted(ids[va][(y[va] == 0).any(1)].tolist())
    assert zero in ([], ZERO), zero
    return ids, y, tr, va


def ssl_evidence(ids, tr, va):
    sm = R.read(R.SSL / 'manifest.json'); assert sm['fold'] == R.ARGS.fold
    if R.ARGS.fold == 99:
        assert sm['config']['train_n'] == len(ids) and sm['config']['val_n'] == 0
    else:
        assert sm['config']['train_n'] == len(tr) and sm['config']['val_n'] == len(va)
    sr = prior.validate_result(R.SSL / 'run', A.sha(R.SSL / 'manifest.json'))
    assert sr['validation_images_used'] == 0
    import torch
    c = torch.load(R.SSL / 'run/last.pt', map_location='cpu', weights_only=True); s = prior.state_sha(c['encoder']); del c
    return sm, sr, s


R.arrays = arrays
R.ssl_evidence = ssl_evidence
R.source_files = lambda: _SOURCES() + [Path(__file__).resolve()]
if __name__ == '__main__':
    if sys.argv[1] in ('cpu-tests', 'cpu_tests'):
        y0 = np.array([[1053., .03, 0.], [520., .11, 0.]]); p0 = np.array([[1000., .03, .5], [600., .22, .4]])
        m = official_metrics(p0, y0); assert m['HEX'] == 0.0 and abs(m['CV'] - 50.0) < 1e-9
        rng = np.random.default_rng(5); y1 = rng.uniform(.2, 3000., (64, 3)); p1 = y1 * rng.uniform(.8, 1.2, y1.shape)
        assert official_metrics(p1, y1) == FROZEN_METRICS(p1, y1)
    elif sys.argv[1] != 'make_input':
        b.metrics = official_metrics
    if sys.argv[1] == 'make_input':
        A.make_input(sys.argv[2])
    else:
        R.main()
