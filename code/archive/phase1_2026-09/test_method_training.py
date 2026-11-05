"""CPU tests; GPU integration is exercised by each recipe's separate smoke run."""
import ast
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

from method_training_core import SAMPerturbation, cosine_multiplier, regression_loss


class MethodTests(unittest.TestCase):
    def test_schedule(self):
        values = [cosine_multiplier(i, 5215) for i in range(5216)]
        self.assertAlmostEqual(values[0], 0.005)
        self.assertEqual(values[199], 1)
        self.assertEqual(values[200], 1)
        self.assertAlmostEqual(values[-1], 0.05)
        self.assertTrue(all(a >= b for a, b in zip(values[200:], values[201:])))

    def test_mape_identity_and_gradient(self):
        z = torch.tensor([[0.1, -0.3, 0.4]], requires_grad=True)
        target = torch.tensor([[0.0, 0.1, -0.2]])
        scale = torch.tensor([0.19, 0.188, 0.275])
        predicted = torch.exp(z * scale)
        truth = torch.exp(target * scale)
        direct = ((predicted - truth).abs() / truth).mean()
        expected = 0.5 * (z - target).abs().mean() + 2.5 * direct
        actual = regression_loss(z, target, scale, "mixed_mape")
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_sam_radius_restore_and_descent(self):
        p = nn.Parameter(torch.tensor([1.0, 2.0]))
        original = p.detach().clone()
        optimizer = torch.optim.SGD([p], lr=0.1)
        p.square().sum().backward()
        with SAMPerturbation([p], 0.02):
            torch.testing.assert_close((p - original).norm(), torch.tensor(0.02))
            optimizer.zero_grad(set_to_none=True)
            p.square().sum().backward()
            perturbed_grad = p.grad.clone()
        self.assertTrue(torch.equal(p, original))
        optimizer.step()
        torch.testing.assert_close(p, original - 0.1 * perturbed_grad)

    def test_sam_exception_restore(self):
        p = nn.Parameter(torch.ones(3))
        original = p.detach().clone()
        p.sum().backward()
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with SAMPerturbation([p], 0.02):
                raise RuntimeError("deliberate")
        self.assertTrue(torch.equal(p, original))

    def test_sam_zero_gradients(self):
        p = nn.Parameter(torch.ones(3))
        p.grad = torch.zeros_like(p)
        with SAMPerturbation([p], 0.02):
            self.assertTrue(torch.equal(p, torch.ones(3)))

    def test_prediction_bag_and_state_compatibility(self):
        path = Path(__file__).parent
        classes = []
        for filename in ("train_oof_h200.py", "train_method_h200.py"):
            tree = ast.parse((path / filename).read_text())
            node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "ScalarNet")
            context = dict(torch=torch, nn=nn, BAG_POOL="mean")
            exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), context)
            classes.append(context["ScalarNet"])
        encoder = nn.Conv2d(3, 256, 1)
        original = classes[0](encoder, "mean").eval()
        updated = classes[1](nn.Conv2d(3, 256, 1), "mean").eval()
        # Nonzero heads make averaging order observable.
        for head in (original.cd, original.cv, original.hex):
            nn.init.normal_(head[-1].weight)
        updated.load_state_dict(original.state_dict(), strict=True)
        x = torch.randn(2, 2, 3, 8, 8)
        torch.testing.assert_close(original(x[:, 0]), updated(x[:, 0]))
        expected = updated(x.flatten(0, 1)).reshape(2, 2, 3).mean(1)
        torch.testing.assert_close(updated(x), expected)
        updated.train()
        updated(x).sum().backward()
        self.assertTrue(all(p.grad is not None for p in updated.parameters()))

    def test_unchanged_data_and_evaluation(self):
        path = Path(__file__).parent
        trees = [ast.parse((path / name).read_text()) for name in
                 ("train_oof_h200.py", "train_method_h200.py")]
        for name in ("make_group_folds", "clean_id", "normalize", "random_view", "train_batch", "evaluate"):
            nodes = [next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name) for tree in trees]
            self.assertEqual(ast.dump(nodes[0]), ast.dump(nodes[1]), name)


if __name__ == "__main__":
    unittest.main()
