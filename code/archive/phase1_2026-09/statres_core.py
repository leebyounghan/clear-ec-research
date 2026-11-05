"""Zero-initialized CV/HEX residuals: matched mean vs spatial-standard-deviation.

Spatial feature statistics are not cell-area measurements or ROI ground truth.
The pretrained mean path and checkpoint keys remain unchanged.
"""
import torch
from torch import nn
from final_candidate_core import CandidateNet


class StatisticsResidualNet(CandidateNet):
    def __init__(self, encoder, statistic="mean"):
        super().__init__(encoder)
        self.statistic = statistic
        self.residual_norm = nn.LayerNorm(256)
        self.residual = nn.Sequential(nn.Linear(256,128), nn.GELU(), nn.Linear(128,2))
        self.reset_residual()

    def reset_residual(self):
        self.residual_norm.reset_parameters()
        self.residual[0].reset_parameters()
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def mean_output(self, features):
        g = self.norm(self.pool(features).flatten(1))
        return torch.cat([self.cd(g), self.cv(g), self.hex(g)], 1)

    def descriptor(self, features):
        flat = features.float().flatten(2)
        if self.statistic == "mean":
            return flat.mean(-1)
        if self.statistic == "std":
            # Population variance, float32 accumulation, epsilon-stable gradient.
            return (flat.var(-1, correction=0) + 1e-6).sqrt()
        raise ValueError(self.statistic)

    def forward_features(self, features):
        output = self.mean_output(features)
        correction = self.residual(self.residual_norm(self.descriptor(features)))
        return output + torch.cat([torch.zeros_like(correction[:, :1]), correction], 1)

    def forward(self, x):
        return self.forward_features(self.encoder(x))
