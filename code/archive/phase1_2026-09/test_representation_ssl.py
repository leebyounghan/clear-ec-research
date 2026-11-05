"""Small CPU checks, run in the K8s preflight Pod."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path('experiments/representation_ssl_20260913/source')))
sys.path.insert(0,'experiments/ssl_screen_20260911/source')
from representation_ssl_core import spatial_relation_loss, VARIANTS
from run_representation_ssl_20260913 import gate


class RepresentationTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42);torch.set_num_threads(2)

    def test_identical_features_zero(self):
        x=torch.randn(2,16,32,32,requires_grad=True)
        loss=spatial_relation_loss(x,x,torch.zeros(2,1,32,32))
        self.assertLess(float(loss),1e-12)

    def test_anchor_has_no_gradient(self):
        x=torch.randn(2,16,32,32,requires_grad=True)
        ref=torch.randn_like(x,requires_grad=True)
        loss=spatial_relation_loss(x,ref,torch.zeros(2,1,32,32));loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(float(x.grad.abs().sum()),0)
        self.assertIsNone(ref.grad)

    def test_no_visible_pair_is_safe(self):
        x=torch.randn(2,16,32,32,requires_grad=True)
        loss=spatial_relation_loss(x,torch.randn_like(x),torch.ones(2,1,32,32));loss.backward()
        self.assertEqual(float(loss),0);self.assertTrue(torch.isfinite(x.grad).all())

    def test_hidden_reference_does_not_affect_loss(self):
        x=torch.randn(2,16,32,32);ref=torch.randn_like(x);mask=torch.zeros(2,1,32,32);mask[:,:,0:16]=1
        first=spatial_relation_loss(x,ref,mask);ref[:,:,0:16]+=torch.randn_like(ref[:,:,0:16])*10
        torch.testing.assert_close(first,spatial_relation_loss(x,ref,mask))

    def test_only_one_objective_change_per_variant(self):
        self.assertEqual(VARIANTS['low_cov'],{'covariance_weight':.001,'anchor_weight':0.})
        self.assertEqual(VARIANTS['spatial_anchor'],{'covariance_weight':.01,'anchor_weight':.1})

    def test_gate_requires_both_fold_and_magnitude(self):
        self.assertTrue(gate(9.,9.04,{'2':{'candidate':9.,'reference':9.1},'3':{'candidate':9.,'reference':9.1}}))
        self.assertFalse(gate(9.,9.02,{'2':{'candidate':9.,'reference':9.1}}))
        self.assertFalse(gate(9.,9.1,{'2':{'candidate':9.2,'reference':9.1}}))


if __name__=='__main__':unittest.main(verbosity=2)
