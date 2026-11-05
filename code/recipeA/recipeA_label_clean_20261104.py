"""Recipe A trained without the rows whose labels disagree most with the image (2026-11-04).

Everything is the frozen phase2_pooling_regression_20261019 runner with std pooling, except that the fold's train rows listed
in experiments/meta_ceiling_20261103/label_clean/drop_rows.npz are removed (the top 3 % of all rows by the largest
MAD-scaled |log(y / OOF B)| over the three metrics). Validation still scores every validation image, unfiltered. The SSL encoder
is the fold's train-only SSL5, unchanged. Why: with more freedom the model memorises label noise (encoder LR x5: train 7.26,
validation +0.33), and the top 1 % of CD errors are mostly label-image disagreement, so those rows may teach the wrong mapping.
Risk: the list also removes real pathology, which could worsen rare cases. Read: member and hybrid on fold 0, then fold 4.
"""
from pathlib import Path

import numpy as np

import phase2_pooling_regression_20261019 as R

WRAPPER = Path(__file__).resolve()
DROP = Path('/workspace/clear-ec-research/experiments/meta_ceiling_20261103/label_clean/drop_rows.npz')
FROZEN_ARRAYS = R.arrays
FROZEN_SSL_EVIDENCE = R.ssl_evidence
FROZEN_SOURCES = R.source_files
FROZEN_CPU_TESTS = R.cpu_tests


def keep_mask(tr):
    with np.load(DROP) as z:
        return ~np.isin(tr, z['indices'])


def arrays():
    ids, y, tr, va = FROZEN_ARRAYS()
    return ids, y, tr[keep_mask(tr)], va


def ssl_evidence(ids, tr, va):
    _, _, tr_full, va_full = FROZEN_ARRAYS()
    assert np.isin(tr, tr_full).all() and np.array_equal(va, va_full)
    return FROZEN_SSL_EVIDENCE(ids, tr_full, va_full)


def source_files():
    return FROZEN_SOURCES() + [WRAPPER, DROP]


def clean_tests():
    ids, y, tr, va = arrays()
    _, _, tr_full, va_full = FROZEN_ARRAYS()
    with np.load(DROP) as z:
        drop = z['indices']
    assert np.array_equal(va, va_full) and not np.isin(tr, drop).any()
    assert np.array_equal(tr, tr_full[~np.isin(tr_full, drop)]) and 0.9 < len(tr) / len(tr_full) < 1.0
    return dict(train_full=int(len(tr_full)), train_kept=int(len(tr)), dropped=int(len(tr_full) - len(tr)))


def cpu_tests():
    print({'clean_tests': clean_tests()}, flush=True)
    return FROZEN_CPU_TESTS()


R.arrays = arrays
R.ssl_evidence = ssl_evidence
R.source_files = source_files
R.cpu_tests = cpu_tests

if __name__ == '__main__':
    R.main()
