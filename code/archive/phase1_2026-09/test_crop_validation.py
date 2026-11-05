"""Regression tests for optimizer coverage and zero-label-safe crop validation."""
import ast
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

CODE = Path(__file__).resolve().parent


class CropValidationTests(unittest.TestCase):
    def test_normalization_is_optimized_and_cleared(self):
        source = (CODE / "train_oof_h200.py").read_text()
        module = ast.parse(source)
        node = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "ScalarNet")
        ns = dict(nn=nn, torch=torch, BAG_POOL="mean", POOL="mean", ENC_LR=1e-3, HEAD_LR=1e-2, WEIGHT_DECAY=0)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "ScalarNet", "exec"), ns)
        net = ns["ScalarNet"](nn.Conv2d(3, 256, 1), "mean")
        ns["net"] = net
        start = source.index("head_params =")
        end = source.index("steps_per_epoch =", start)
        exec(source[start:end], ns)
        opt = ns["optimizer"]
        initial = net.norm.weight.detach().clone()
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            assert net.norm.weight.grad is None
            loss = (net(torch.randn(2, 3, 8, 8)) - 1).square().mean()
            loss.backward(); opt.step()
        self.assertFalse(torch.equal(initial, net.norm.weight.detach()))
        opt.zero_grad(set_to_none=True)
        self.assertIsNone(net.norm.weight.grad)

    def test_reports_handle_two_zero_hex_targets(self):
        rng = np.random.default_rng(123)
        n = 100
        lm = np.array([7.919919967651367, -.9824771285057068, -.6211431622505188])
        ls = np.array([.18971863389015198, .18813571333885193, .2748945653438568])
        z = rng.normal(0, .4, (n, 5, 3))
        y = np.exp(z.mean(1) * ls + lm + rng.normal(0, .05, (n, 3)))
        y[[0, 45], 2] = 0
        ids = np.array([f"image{i}" for i in range(n)])
        fold = np.arange(n) // 20
        with tempfile.TemporaryDirectory(prefix="clear-ec-test-") as temporary:
            root = Path(temporary)
            def tiles(path, indices):
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, indices=indices, ids=ids[indices], target=y[indices],
                                    z_tile=z[indices], quality_tile=rng.uniform(.01, 1, (len(indices), 5, 4)), lm=lm, ls=ls)
            base = root / "experiments/oof_h200"
            for f in range(5):
                tiles(base / f"baseline_f{f}_seed42_c256k1_mean_multitask_slidesmooth/crop_tiles.npz", np.where(fold == f)[0])
            env = os.environ.copy(); env.update(OOF_ROOT=str(base), TAG="baseline", SEED="42")
            subprocess.run([sys.executable, str(CODE / "summarize_crop_oof.py")], cwd=root, env=env, check=True, capture_output=True)
            report = json.loads((base / "crop_audit_baseline.json").read_text())
            self.assertEqual(report["invalid_targets_by_metric"], [0, 0, 2])
            self.assertTrue(all(np.isfinite(v["overall"]) for v in report["methods"].values()))
            predicted = np.exp(z.mean(1) * ls + lm)
            expected = [np.mean(np.abs(predicted[y[:, m] > 0, m] - y[y[:, m] > 0, m]) / y[y[:, m] > 0, m]) * 100 for m in range(3)]
            np.testing.assert_allclose(report["methods"]["current_geomean"]["metrics"], expected)
            followup = root / "experiments/followup_20260902"
            for outer in range(5):
                cal = (outer + 1) % 5
                run = followup / "runs" / f"nested_o{outer}_f{cal}_seed42_c256k1_mean_multitask_slidesmooth"
                tiles(run / "crop_tiles.npz", np.where(fold == cal)[0])
                tiles(run / "crop_holdout.npz", np.where(fold == outer)[0])
                np.savez(run / "split_indices.npz", train=np.where((fold != cal) & (fold != outer))[0], validation=np.where(fold == cal)[0], holdout=np.where(fold == outer)[0])
                (run / "result.json").write_text(json.dumps(dict(select_mode="last", exclude_fold=outer)))
            data = root / "datasets/clear_ec_official"
            data.mkdir(parents=True)
            with (data / "final_train_ids.csv").open("w", newline="") as stream:
                writer = csv.writer(stream); writer.writerow(["ID", "slide_id"])
                writer.writerows((value, str(i // 2)) for i, value in enumerate(ids))
            subprocess.run([sys.executable, str(CODE / "validate_crop_holdouts.py")], cwd=root, check=True, capture_output=True)
            report = json.loads((followup / "nested_holdout_report.json").read_text())
            self.assertEqual(report["invalid_by_metric"], [0, 0, 2])
            self.assertEqual(report["n"], n)
            self.assertTrue(all(np.isfinite(v["overall"]) for v in report["methods"].values()))


if __name__ == "__main__":
    unittest.main()
