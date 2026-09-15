"""Synthetic-only CPU tests; no challenge images or labels."""
import unittest
import numpy as np
import torch
from ssl_native_core import penalties, normalized_tokens, feature_diagnostics, require_healthy, mask_batch


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)
        np.random.seed(12)

    def test_regularization_gradients(self):
        x = torch.randn(8,256,32,32,requires_grad=True)
        variance,covariance=penalties(x)
        (variance+covariance).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()),0)
        self.assertEqual(normalized_tokens(x).shape,(8,32,32,256))

    def test_collapsed_rejected(self):
        healthy=feature_diagnostics(torch.randn(8,256,32,32))
        require_healthy(healthy,healthy)
        collapsed=feature_diagnostics(torch.zeros(8,256,32,32))
        with self.assertRaises(FloatingPointError):
            require_healthy(collapsed,healthy)
        variance,covariance=penalties(torch.zeros(8,256,32,32))
        self.assertGreater(float(variance),.6)
        self.assertEqual(float(covariance),0)

    def test_masks(self):
        masks=mask_batch(32)
        self.assertEqual(masks.shape,(32,1,32,32))
        self.assertTrue(set(np.unique(masks))<={0,1})
        self.assertTrue((masks.mean((1,2,3))>=.39).all())
        self.assertTrue((masks.mean((1,2,3))<.71).all())


if __name__=='__main__':
    unittest.main()
