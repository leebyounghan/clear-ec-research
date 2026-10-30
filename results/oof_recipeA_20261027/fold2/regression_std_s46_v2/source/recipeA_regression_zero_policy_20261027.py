"""Recipe A regression on a fold whose validation holds zero targets: official per-metric NaN rule.

Everything is the frozen phase2_backbone_ssl_regression_20261020 runner except two
functions. arrays() admits zero targets only for the validation rows bound by the
addendum; training stays strictly positive, so the log-target loss and train-only
normalisation are unchanged. metrics() scores a zero ground truth as NaN for that
metric and averages the rest, following the public CLEAR-EC evaluate.py; without
zeros it equals the frozen metric exactly. The frozen zero-rejection CPU test still
runs against the original metric.
"""
import json
from pathlib import Path
import sys

import numpy as np

import phase2_backbone_ssl_regression_20261020 as R
import phase2_foundation_train_20261002 as b

ADDENDUM = Path('experiments/oof_recipeA_20261027/fold2_zero_policy_addendum.json')
FROZEN_METRICS = b.metrics
FROZEN_SOURCES = R.source_files


def official_metrics(pred, y):
    if pred.shape != y.shape or not np.isfinite(pred).all() or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Invalid predictions/targets')
    scored = y > 0
    if scored.all():
        return FROZEN_METRICS(pred, y)
    err = np.where(scored, np.abs(pred-y)/np.where(scored, y, 1.), np.nan)*100
    v = np.nanmean(err, 0)
    out = dict(CD=float(v[0]), CV=float(v[1]), HEX=float(v[2]), mean=float(v.mean()), n=len(y))
    if not scored.all():
        out['zero_targets_unscored'] = dict(zip(('CD', 'CV', 'HEX'), (~scored).sum(0).tolist()))
    return out


def arrays():
    addendum = json.loads(ADDENDUM.read_text())
    with np.load(b.META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    with np.load(R.SPLIT) as z:
        tr, va = z['train'].copy(), z['validation'].copy()
        if 'primary_validation' in z:
            historical = z['primary_validation']
            assert len(np.unique(historical)) == len(historical)
            assert np.isin(historical, va).all(), 'Historical subset must be in full validation'
    assert ids.ndim == 1 and len(np.unique(ids)) == len(ids)
    assert y.shape == (len(ids), 3)
    for name, ix in (('train', tr), ('validation', va)):
        assert ix.ndim == 1 and np.issubdtype(ix.dtype, np.integer), name
        assert len(ix) > 0 and len(np.unique(ix)) == len(ix), name
        assert ix.min() >= 0 and ix.max() < len(ids), name
    assert not np.intersect1d(tr, va).size, 'Train/validation overlap'
    assert addendum['split_sha256'] == R.digest(R.SPLIT) and addendum['fold'] == R.ARGS.fold
    if not np.isfinite(y[np.r_[tr, va]]).all() or not (y[tr] > 0).all() or (y[va] < 0).any():
        raise ValueError('Training targets must be positive; validation targets nonnegative')
    zero = sorted(ids[va][(y[va] == 0).any(1)].tolist())
    if zero != sorted(addendum['zero_validation_ids']):
        raise ValueError('Validation zero targets differ from the addendum')
    return ids, y, tr, va


def zero_policy_tests(root):
    rng = np.random.default_rng(3)
    y = rng.uniform(.2, 3000., (64, 3))
    pred = y*rng.uniform(.8, 1.2, y.shape)
    assert official_metrics(pred, y) == FROZEN_METRICS(pred, y), 'Must equal the frozen metric without zeros'
    y0 = np.array([[100., .3, .5], [200., .4, 0.], [50., .1, .25]])
    p0 = np.array([[110., .3, .6], [180., .5, .2], [50., .2, .25]])
    m = official_metrics(p0, y0)
    assert abs(m['CD']-np.mean([10., 10., 0.])) < 1e-9 and abs(m['CV']-np.mean([0., 25., 100.])) < 1e-9
    assert abs(m['HEX']-np.mean([20., 0.])) < 1e-9 and m['zero_targets_unscored'] == dict(CD=0, CV=0, HEX=1)
    for bad in (np.array([[1., 1., -1.]]),):
        try:
            official_metrics(np.ones_like(bad), bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Negative targets must be rejected')
    try:
        FROZEN_METRICS(p0, y0)
    except ValueError:
        pass
    else:
        raise AssertionError('Frozen metric must still reject zeros')
    ids, y, tr, va = arrays()
    report = dict(status='passed', manifest_sha=R.digest(root/'manifest.json'),
                  addendum_sha=R.digest(ADDENDUM), zero_validation_ids=sorted(ids[va][(y[va] == 0).any(1)].tolist()),
                  checks=['equals_frozen_metric_without_zeros', 'per_metric_nan_for_zero_truth',
                          'negative_targets_rejected', 'frozen_metric_still_rejects_zeros',
                          'training_targets_strictly_positive', 'zero_rows_match_addendum'])
    R.save(root/'zero_policy_tests.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    R.source_files = lambda: FROZEN_SOURCES()+[Path(__file__).resolve()]
    R.arrays = arrays
    command = sys.argv[1]
    if command in ('cpu-tests', 'cpu_tests'):
        root = Path(sys.argv[sys.argv.index('--root')+1])
        if (root/'zero_policy_tests.json').exists():
            raise FileExistsError('Zero-policy test evidence already exists')
        R.SPLIT = Path(sys.argv[sys.argv.index('--split')+1])
        R.ARGS = type('A', (), dict(fold=int(sys.argv[sys.argv.index('--fold')+1])))
        zero_policy_tests(root)
    else:
        b.metrics = official_metrics
    R.main()
