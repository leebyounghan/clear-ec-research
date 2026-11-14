#!/usr/bin/env python3
"""Distribution head (HL-Gauss / DLDL) on the family runner (CD lane, 2026-11-09). KG: gao2017dldl, farebrother2024hlgauss; related closed
exp-gpu2-frozen-pooling-heads (quantile head on a frozen encoder, different). Only the last layer and the loss change:
- last layer Linear(128, 3 x K) -> per target a softmax over K = 64 bins of the standardized log target z in [-4, 4];
- target = Gaussian over the bin centres around the true z (sigma = 0.75 bin width; constant in log space = width proportional to the label);
- loss = cross-entropy, summed over the three targets; the network output used everywhere else (metrics, predictions, TTA) is the expected z.
Same SSL, seed, schedule, flips and split as the recipe member it is compared with. usage: as family_cd_20261108.py (imports and patches it)."""
import os, sys
import torch
from torch import nn
sys.path.insert(0, 'experiments/lane_cd_20261108/source')
import family_cd_20261108 as Fm
K, LO, HI = 64, -4.0, 4.0
WIDTH = (HI - LO) / (K - 1); SIG = 0.75 * WIDTH
HOLD = {}
class DistOut(nn.Linear):
    def __init__(self, d):
        super().__init__(d, 3 * K)
    def forward(self, x):
        l = super().forward(x).view(-1, 3, K); HOLD['logits'] = l
        c = torch.linspace(LO, HI, K, device=l.device, dtype=torch.float32)
        return (l.float().softmax(-1) * c).sum(-1)
_orig_build = Fm.build_net
def build_net(a, seed, ssl_path):
    net = _orig_build(a, seed, ssl_path)
    torch.manual_seed(seed); d = DistOut(net[-1].in_features); nn.init.normal_(d.weight, std=1e-3); nn.init.zeros_(d.bias)
    net[-1] = d.cuda(); return net
def dist_loss(z, targets, mu, sigma):
    zt = ((targets.log() - mu) / sigma).float().clamp(LO, HI)                     # B x 3
    c = torch.linspace(LO, HI, K, device=zt.device, dtype=torch.float32)
    q = torch.softmax(-(c[None, None, :] - zt[..., None]) ** 2 / (2 * SIG ** 2), -1)
    lp = torch.log_softmax(HOLD['logits'].float(), -1)
    return -(q * lp).sum(-1).sum(-1).mean()
Fm.build_net = build_net
Fm.lane_loss = dist_loss
Fm.CD_LOSS = 'dist_hlgauss_K64'
if __name__ == '__main__':
    Fm.main()
