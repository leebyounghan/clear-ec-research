"""Recipe A full-data regression for the submission: all audited rows except the two zero-HEX rows.

The frozen phase2_backbone_ssl_regression_20261020 runner is reused unchanged except for
three documented deviations, each bound to fulldata_regression_addendum.json:
1. arrays(): the "validation" partition is exactly the two zero-HEX rows (decision 가). They are
   never trained on and they are not an evaluation set; the per-epoch scores the runner writes
   for them are placeholders (HEX unscored, CD/CV over two images).
2. ssl_evidence(): the encoder is the all-data SSL5 (trained on all 8954 images, no validation),
   so the SSL split is a superset of the regression split instead of being identical to it.
3. metrics(): official zero rule; a column with no scored row is reported as 0.0 so the frozen
   audit's exact-equality checks hold. Without zero targets it returns the frozen metric verbatim.
Training targets stay strictly positive; log mean/std come from the 8952 training rows.
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch

import phase2_backbone_ssl_regression_20261020 as R
import phase2_foundation_train_20261002 as b
import phase2_lejepa_regression_20261003 as prior

ADDENDUM = Path('experiments/fulldata_recipeA_20261028/fulldata_regression_addendum.json')
FROZEN_METRICS = b.metrics
FROZEN_SOURCES = R.source_files


def addendum():
    return json.loads(ADDENDUM.read_text())


def placeholder_metrics(pred, y):
    if pred.shape != y.shape or not np.isfinite(pred).all() or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Invalid predictions/targets')
    scored = y > 0
    if scored.all():
        return FROZEN_METRICS(pred, y)
    err = np.where(scored, np.abs(pred-y)/np.where(scored, y, 1.), np.nan)*100
    with np.errstate(all='ignore'):
        v = np.nanmean(err, 0)
    v = np.where(np.isfinite(v), v, 0.0)
    out = dict(CD=float(v[0]), CV=float(v[1]), HEX=float(v[2]), mean=float(v.mean()), n=len(y))
    out['placeholder_not_an_evaluation'] = dict(zip(('CD', 'CV', 'HEX'), (~scored).sum(0).tolist()))
    return out


def arrays():
    a = addendum()
    with np.load(b.META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    with np.load(R.SPLIT) as z:
        tr, va = z['train'].copy(), z['validation'].copy()
    assert a['split_sha256'] == R.digest(R.SPLIT) and a['fold'] == R.ARGS.fold
    assert ids.ndim == 1 and len(np.unique(ids)) == len(ids) and y.shape == (len(ids), 3)
    for name, ix in (('train', tr), ('validation', va)):
        assert ix.ndim == 1 and np.issubdtype(ix.dtype, np.integer), name
        assert len(ix) > 0 and len(np.unique(ix)) == len(ix) and ix.min() >= 0 and ix.max() < len(ids), name
    assert not np.intersect1d(tr, va).size
    assert np.array_equal(np.sort(np.r_[tr, va]), np.arange(len(ids))), 'Train and the two held rows must cover every audited image'
    assert np.isfinite(y).all() and (y[tr] > 0).all() and (y[va] >= 0).any()
    assert sorted(ids[va].tolist()) == sorted(a['zero_validation_ids']) and len(va) == 2
    assert (y[va] == 0).any(1).all(), 'Held rows must be exactly the zero-target rows'
    return ids, y, tr, va


def ssl_evidence(ids, tr, va):
    a = addendum()
    sm = R.read(R.SSL/'manifest.json')
    for path, expected in sm['hashes'].items():
        assert R.digest(path) == expected, path
    assert str(R.SSL) == a['ssl_root'] and sm['split_sha256'] == a['ssl_split_sha256']
    assert sm['fold'] == 99 and sm['train_n'] == len(ids) and sm['val_n'] == 0 and sm['total_images'] == len(ids)
    assert sm['train_ids_sha'] == R.ids_sha(ids[np.arange(len(ids))]), 'All-data SSL must have trained on every audited image in index order'
    sr = prior.validate_result(R.SSL/'run', R.digest(R.SSL/'manifest.json'))
    assert sr['validation_images_used'] == 0 and sr['epochs'] == R.ARGS.ssl_epochs and len(sr['epochs_detail']) == R.ARGS.ssl_epochs
    for epoch, item in enumerate(sr['epochs_detail'], 1):
        assert item['epoch'] == epoch and item['validation_images_used'] == 0
        assert item['unique_train_images'] == len(ids) and item['train_images_seen'] == len(ids)
    checkpoint = torch.load(R.SSL/'run/last.pt', map_location='cpu', weights_only=True)
    assert checkpoint['epoch'] == R.ARGS.ssl_epochs and checkpoint['manifest_sha'] == R.digest(R.SSL/'manifest.json')
    assert checkpoint['step'] == sr['steps']
    encoder_sha = prior.state_sha(checkpoint['encoder'])
    del checkpoint
    return sm, sr, encoder_sha


def fulldata_tests(root):
    rng = np.random.default_rng(5)
    y = rng.uniform(.2, 3000., (64, 3))
    pred = y*rng.uniform(.8, 1.2, y.shape)
    assert placeholder_metrics(pred, y) == FROZEN_METRICS(pred, y)
    y0 = np.array([[1053., .03, 0.], [520., .11, 0.]])
    p0 = np.array([[1000., .03, .5], [600., .22, .4]])
    m = placeholder_metrics(p0, y0)
    assert m['HEX'] == 0.0 and abs(m['CD']-np.mean([100*53/1053, 100*80/520])) < 1e-9 and abs(m['CV']-50.0) < 1e-9
    assert m['placeholder_not_an_evaluation'] == dict(CD=0, CV=0, HEX=2)
    try:
        FROZEN_METRICS(p0, y0)
    except ValueError:
        pass
    else:
        raise AssertionError('Frozen metric must still reject zeros')
    ids, y, tr, va = arrays()
    a = addendum()
    report = dict(status='passed', manifest_sha=R.digest(root/'manifest.json'), addendum_sha=R.digest(ADDENDUM),
                  train_n=int(len(tr)), held_zero_rows=sorted(ids[va].tolist()), ssl_root=a['ssl_root'],
                  checks=['equals_frozen_metric_without_zeros', 'all_unscored_column_reported_as_zero',
                          'frozen_metric_still_rejects_zeros', 'train_plus_held_rows_cover_all_images',
                          'training_targets_strictly_positive', 'held_rows_are_the_two_zero_rows'])
    R.save(root/'fulldata_tests.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    R.source_files = lambda: FROZEN_SOURCES()+[Path(__file__).resolve()]
    R.arrays = arrays
    R.ssl_evidence = ssl_evidence
    command = sys.argv[1]
    if command in ('cpu-tests', 'cpu_tests'):
        root = Path(sys.argv[sys.argv.index('--root')+1])
        if (root/'fulldata_tests.json').exists():
            raise FileExistsError('Full-data test evidence already exists')
        R.SPLIT = Path(sys.argv[sys.argv.index('--split')+1])
        R.SSL = Path(sys.argv[sys.argv.index('--ssl-root')+1])
        R.ARGS = type('A', (), dict(fold=int(sys.argv[sys.argv.index('--fold')+1]), ssl_epochs=5))
        fulldata_tests(root)
    else:
        b.metrics = placeholder_metrics
    R.main()
