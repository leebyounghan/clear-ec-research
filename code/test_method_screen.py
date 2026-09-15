import unittest

import numpy as np

from audit_method_screen import aligned, gate, METHODS
from run_method_screen import make_job


class ScreenTests(unittest.TestCase):
    def test_finite_job_design(self):
        jobs = [make_job(m, f) for m in METHODS for f in (0, 1)]
        self.assertEqual(len({j["key"] for j in jobs}), 8)
        for job in jobs:
            env = job["env"]
            self.assertEqual(env["EPOCHS"], 35)
            self.assertEqual(env["FEATURE_AGG"], 0)
            self.assertEqual(env["ENCODER_INIT"], "")
            self.assertEqual(env["SEED"], 42)
            self.assertEqual(env["MAX_VAL_IMAGES"], 0)
            self.assertEqual(env["FINAL_D8"], 1)
        self.assertEqual(make_job("predbag2_35", 0)["env"]["K_CROPS"], 2)
        self.assertEqual(make_job("sam35", 0)["env"]["SAM_RHO"], 0.02)

    def test_gate_rejects_noise_and_bad_fold(self):
        example = dict(standalone_delta=-0.001, standalone_fold_deltas=[-0.001, -0.001],
                       blend_delta=-0.001, blend_fold_deltas=[-0.001, -0.001])
        self.assertFalse(gate(example))
        example.update(blend_delta=-0.02, blend_fold_deltas=[-0.03, -0.01])
        self.assertTrue(gate(example))
        example.update(blend_delta=-0.1, blend_fold_deltas=[-0.25, 0.05])
        self.assertFalse(gate(example))

    def test_alignment_rejects_training_leak_or_id_mismatch(self):
        data = dict(indices=np.array([0, 1]), ids=np.array(["a", "b"]),
                    target=np.ones((2, 3)), train=np.array([2, 3]))
        aligned(data, data)
        for key in data:
            changed = {k: v.copy() for k, v in data.items()}
            changed[key] = changed[key][::-1] if key != "target" else changed[key] * 2
            with self.assertRaises(ValueError):
                aligned(data, changed)


if __name__ == "__main__":
    unittest.main()
