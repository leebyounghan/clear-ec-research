import unittest
import numpy as np

from audit_submission_ensembles import crossfit_scale, metrics
from run_submission_sweep import build_jobs


class SubmissionSweepTests(unittest.TestCase):
    def test_job_manifest(self):
        jobs = build_jobs()
        self.assertEqual(len(jobs), 20)
        self.assertEqual(len({j['key'] for j in jobs}), 20)
        self.assertEqual(len({j['run'] for j in jobs}), 20)
        for group in {j['group'] for j in jobs}:
            self.assertEqual({j['fold'] for j in jobs if j['group'] == group}, set(range(5)))
        for j in jobs:
            self.assertEqual(j['env']['EPOCHS'], 25)
            self.assertEqual(j['env']['MAX_TRAIN_STEPS'], 0)
            self.assertEqual(j['env']['ENCODER_INIT'], '')

    def test_zero_labels_and_calibration_exclusion(self):
        folds = np.repeat(np.arange(5), 4)
        pred = np.ones((20, 3)) * 2
        target = np.ones((20, 3))
        target[4, 2] = 0
        target[5, 2] = np.nan
        scaled, _ = crossfit_scale(pred, target, folds)
        np.testing.assert_allclose(metrics(scaled, target), 0)
        changed = target.copy(); changed[folds == 0] = 100
        updated, _ = crossfit_scale(pred, changed, folds)
        np.testing.assert_allclose(updated[folds == 0], scaled[folds == 0])
        with self.assertRaises(ValueError):
            metrics(np.full((20, 3), np.nan), target)


if __name__ == '__main__':
    unittest.main()
