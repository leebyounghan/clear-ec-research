"""Recipe A framed-style specialist (2026-11-02).

Everything is the frozen phase2_pooling_regression_20261019 runner with std pooling (the recipe A member runner),
except that training uses only the fold's train rows with the framed acquisition style (black_frac > 0.01, the
label-free image statistic of eval_harness_20261031: fraction of raw pixels < 5 on the 2x-subsampled frame).
Validation still scores every validation image; the specialist is meant to replace the generalist on framed images
only. The SSL encoder is the fold's train-only SSL5 (trained on all fold-train images, both styles), so the SSL
evidence is checked against the full train split while the regression sees the framed subset.
Why: framed images (48 % of rows) are a different acquisition style (bright centre, vignetted periphery falling
below 5, darker, CV ~0.41, HEX ~0.52); recipe A beats Phase-1 by 0.21 on unframed images and loses by 0.04-0.17 on
framed ones, also within 2019-2022, and masked std pooling did not change that. Hypothesis: one model compromises
between the two styles; a style specialist does not have to.
"""
from pathlib import Path

import numpy as np

import phase2_pooling_regression_20261019 as R

WRAPPER = Path(__file__).resolve()
STATS = Path('/workspace/clear-ec-research/experiments/eval_harness_20261031/image_stats.npz')
THRESH = 0.01
FROZEN_ARRAYS = R.arrays
FROZEN_SSL_EVIDENCE = R.ssl_evidence
FROZEN_SOURCES = R.source_files
FROZEN_CPU_TESTS = R.cpu_tests


def framed_mask(indices):
    with np.load(STATS) as z:
        idx, bf = z['idx'], z['black_frac']
    pos = {int(i): k for k, i in enumerate(idx)}
    missing = [int(i) for i in indices if int(i) not in pos]
    assert not missing, f'{len(missing)} indices have no image statistics'
    return np.array([bf[pos[int(i)]] > THRESH for i in indices])


def arrays():
    ids, y, tr, va = FROZEN_ARRAYS()
    return ids, y, tr[framed_mask(tr)], va


def ssl_evidence(ids, tr, va):
    _, _, tr_full, va_full = FROZEN_ARRAYS()
    assert np.isin(tr, tr_full).all() and np.array_equal(va, va_full)
    return FROZEN_SSL_EVIDENCE(ids, tr_full, va_full)


def source_files():
    return FROZEN_SOURCES() + [WRAPPER, STATS]


def specialist_tests():
    ids, y, tr, va = arrays()
    _, _, tr_full, _ = FROZEN_ARRAYS()
    assert framed_mask(tr).all() and len(tr) > 1000
    assert not framed_mask(np.setdiff1d(tr_full, tr)).any()
    assert np.array_equal(tr, tr_full[framed_mask(tr_full)])          # order preserved
    return dict(train_full=int(len(tr_full)), train_framed=int(len(tr)), framed_fraction=round(len(tr) / len(tr_full), 4))


def cpu_tests():
    print({'specialist_tests': specialist_tests()}, flush=True)
    return FROZEN_CPU_TESTS()


R.arrays = arrays
R.ssl_evidence = ssl_evidence
R.source_files = source_files
R.cpu_tests = cpu_tests

if __name__ == '__main__':
    R.main()
