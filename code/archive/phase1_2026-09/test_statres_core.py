"""CPU invariants; no data, credentials, GPUs, or checkpoint access."""
import unittest
import torch
from torch import nn
from statres_core import StatisticsResidualNet


class StatisticsTests(unittest.TestCase):
    def test_zero_residual_and_capacity(self):
        net = StatisticsResidualNet(nn.Identity()).eval()
        # Nonzero reference heads simulate a trained common checkpoint.
        for h in (net.cd,net.cv,net.hex): nn.init.normal_(h[-1].weight)
        x = torch.randn(3,256,4,4)
        reference = net.mean_output(x)
        params = sum(p.numel() for p in net.parameters())
        for mode in ("mean","std"):
            net.statistic = mode
            torch.testing.assert_close(net(x),reference,rtol=0,atol=0)
            self.assertEqual(params,sum(p.numel() for p in net.parameters()))
        self.assertEqual(params-sum(p.numel() for n,p in net.named_parameters()
            if n.startswith(("cd.","cv.","hex.","norm."))),33666)

    def test_std_has_information_beyond_mean(self):
        net = StatisticsResidualNet(nn.Identity(),"std")
        x = torch.zeros(2,256,2,2)
        x[1,:,:,0] = -1; x[1,:,:,1] = 1
        torch.testing.assert_close(x[0].mean((1,2)),x[1].mean((1,2)))
        self.assertTrue((net.descriptor(x)[1] > net.descriptor(x)[0]).all())

    def test_constant_input_gradient_is_finite(self):
        net = StatisticsResidualNet(nn.Identity(),"std")
        x = torch.ones(2,256,2,2,requires_grad=True)
        net.descriptor(x).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_residual_affects_only_cv_hex(self):
        net = StatisticsResidualNet(nn.Identity(),"std").eval()
        x = torch.randn(2,256,3,3)
        reference = net.mean_output(x)
        nn.init.constant_(net.residual[-1].bias,.5)
        result = net(x)
        torch.testing.assert_close(result[:,0],reference[:,0],rtol=0,atol=0)
        torch.testing.assert_close(result[:,1:],reference[:,1:]+.5)


if __name__ == "__main__": unittest.main()
