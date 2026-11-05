"""Recipe A labelled-data learning curve (ceiling meta-analysis, 2026-11-03).

Everything is the frozen phase2_pooling_regression_20261019 runner with std pooling (the recipe A member runner),
except that training uses a donor-grouped, nested subsample of the fold's train rows: donors (slide_id) are shuffled
once with a fixed generator and taken in that order until the kept images reach LABEL_FRACTION x train rows, so the
25 % subset is contained in the 50 % subset. Validation still scores every validation image. The SSL encoder is the
fold's train-only SSL5 (label-free, all fold-train images), so the curve measures the value of labels, not of images.
LABEL_FRACTION comes from the environment and is pinned in <root>/label_fraction.json at the first call; every later
step (cpu_tests, smoke, train, audit) must see the same value.
Why: decide whether the remaining error is limited by the amount of labelled data (curve still falling at 100 %)
or by label noise / missing information (curve flat), which decides between "more data-like training ideas" and
"stop training-side work".
"""
import csv, json, os, re
from pathlib import Path

import numpy as np

import phase2_pooling_regression_20261019 as R

WRAPPER = Path(__file__).resolve()
IDS_CSV = Path('/workspace/clear-ec-research/datasets/clear_ec_official/final_train_ids.csv')
ORDER_SEED = 20261103
FROZEN_ARRAYS = R.arrays
FROZEN_SSL_EVIDENCE = R.ssl_evidence
FROZEN_SOURCES = R.source_files
FROZEN_CPU_TESTS = R.cpu_tests


def fraction():
    env = os.environ.get('LABEL_FRACTION')
    pin = R.ROOT / 'label_fraction.json'
    if pin.exists():
        f = json.loads(pin.read_text())['label_fraction']
        assert env is None or abs(float(env) - f) < 1e-12, f'LABEL_FRACTION {env} differs from pinned {f}'
        return f
    assert env is not None, 'set LABEL_FRACTION for the first call'
    f = float(env)
    assert 0 < f < 1
    R.ROOT.mkdir(parents=True, exist_ok=True)
    pin.write_text(json.dumps(dict(label_fraction=f, order_seed=ORDER_SEED)))
    return f


def donor_of(ids):
    gt = {re.sub(r'\s+', ' ', r['ID'].strip()): r['slide_id'].strip() for r in csv.DictReader(IDS_CSV.open())}
    return np.array([gt.get(re.sub(r'\s+', ' ', str(i).strip()), str(i)) for i in ids])


def subsample(ids, tr, frac):
    d = donor_of(ids[tr])
    order = np.random.default_rng(ORDER_SEED).permutation(np.unique(d))
    counts = {k: v for k, v in zip(*np.unique(d, return_counts=True))}
    keep, n = set(), 0
    for g in order:
        if n >= frac * len(tr):
            break
        keep.add(g); n += counts[g]
    return tr[np.array([g in keep for g in d])]


def arrays():
    ids, y, tr, va = FROZEN_ARRAYS()
    return ids, y, subsample(ids, tr, fraction()), va


def ssl_evidence(ids, tr, va):
    _, _, tr_full, va_full = FROZEN_ARRAYS()
    assert np.isin(tr, tr_full).all() and np.array_equal(va, va_full)
    return FROZEN_SSL_EVIDENCE(ids, tr_full, va_full)


def source_files():
    return FROZEN_SOURCES() + [WRAPPER, IDS_CSV]


def fraction_tests():
    ids, y, tr, va = arrays()
    _, _, tr_full, _ = FROZEN_ARRAYS()
    f = fraction()
    d_full, d_sub = donor_of(ids[tr_full]), donor_of(ids[tr])
    assert np.isin(tr, tr_full).all() and np.array_equal(tr, tr_full[np.isin(tr_full, tr)])     # order preserved
    assert not np.isin(d_full[~np.isin(tr_full, tr)], d_sub).any(), 'donor split across kept/dropped'
    small = subsample(ids, tr_full, f / 2)
    assert np.isin(small, tr).all(), 'subsets are not nested'
    return dict(train_full=int(len(tr_full)), train_kept=int(len(tr)), kept_fraction=round(len(tr) / len(tr_full), 4),
                donors_kept=int(len(np.unique(d_sub))), donors_full=int(len(np.unique(d_full))), label_fraction=f)


def cpu_tests():
    print({'fraction_tests': fraction_tests()}, flush=True)
    return FROZEN_CPU_TESTS()


R.arrays = arrays
R.ssl_evidence = ssl_evidence
R.source_files = source_files
R.cpu_tests = cpu_tests

if __name__ == '__main__':
    R.main()
