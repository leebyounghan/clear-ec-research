"""Recipe A regression with black-frame-masked std pooling (probe, 2026-10-31).

Everything is the frozen phase2_backbone_ssl_regression_20261020 runner (the runner that produced the fold-0 seed-46
recipe A member) except the pooling mode 'std_masked': patch tokens whose 8x8 input patch is mostly black
(more than half of its pixels <= 0.02 after CLAHE + 1/99 percentile scaling, which recovers 93 % of the raw-black
patches) are excluded from the std; with no such token it equals the frozen 'std' pooling exactly; if every
token is black all tokens are used. Applied identically in training and evaluation.
Why: recipe A beats Phase-1 by 0.21 on frames without a black border but loses by 0.04-0.17 on any framed image
(not in proportion to the frame size; the percentile window changes only 4 %); the std over all tokens mixes
background tokens into the cell statistics. An inference-only mask hurt (12.56 vs 9.43 on framed fold-4 images)
because the head was trained on the unmasked statistic, so the hypothesis needs a retrained head and encoder.
"""
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if not (HERE / 'phase2_backbone_ssl_regression_20261020.py').exists():
    sys.path.insert(0, '/workspace/clear-ec-research/experiments/phase2_pooling_replication_20261026/f0_s46_std/source')
import phase2_pooling_20261019 as pooling
import phase2_backbone_ssl_regression_20261020 as R

WRAPPER = Path(__file__).resolve()
DARK_X, DARK_FRAC = 0.02, 0.5
pooling.MODES['std_masked'] = ('std',)
FROZEN_FORWARD = pooling.SpatialPoolEncoder.forward
FROZEN_SOURCES = R.source_files
FROZEN_CPU_TESTS = R.cpu_tests


def masked_std(tokens, x):
    B, N, C = tokens.shape
    H, W = x.shape[-2:]
    p = int(round((H * W / N) ** 0.5))
    assert (H // p) * (W // p) == N, (H, W, N, p)
    dark = F.avg_pool2d((x[:, :1].float() <= DARK_X).float(), p).flatten(1) > DARK_FRAC
    keep = (~dark).float()
    keep[keep.sum(1) == 0] = 1.0
    w = keep[..., None]
    t = tokens.float()
    n = w.sum(1)
    mean = (t * w).sum(1) / n
    var = (((t - mean[:, None]) ** 2) * w).sum(1) / n
    return (var + pooling.STD_EPS).sqrt().to(tokens.dtype)


def forward(self, x):
    if self.mode != 'std_masked':
        return FROZEN_FORWARD(self, x)
    return masked_std(self.original.forward_features(x)['x_norm_patchtokens'], x)


pooling.SpatialPoolEncoder.forward = forward


def source_files():
    return FROZEN_SOURCES() + [WRAPPER]


def masked_pool_tests():
    g = torch.Generator().manual_seed(0)
    tok = torch.randn(3, 12, 5, generator=g, dtype=torch.float64)
    x = torch.rand(3, 3, 6, 8, generator=g, dtype=torch.float64) * 0.5 + 0.3       # no dark pixels, p = 2
    ref = pooling.pool_tokens(tok, 'std')
    torch.testing.assert_close(masked_std(tok, x), ref)                                  # identity without dark
    x2 = x.clone(); x2[1, :, 0:2, 0:4] = 0.0                                             # two dark patches in image 1
    keep = torch.ones(12, dtype=torch.bool); keep[[0, 1]] = False
    exp1 = (tok[1][keep].var(0, unbiased=False) + pooling.STD_EPS).sqrt()
    out = masked_std(tok, x2)
    torch.testing.assert_close(out[1], exp1); torch.testing.assert_close(out[[0, 2]], ref[[0, 2]])
    x3 = torch.zeros_like(x)                                                             # all dark -> all tokens
    torch.testing.assert_close(masked_std(tok, x3), ref)
    t = tok.clone().requires_grad_(True)
    masked_std(t, x2).sum().backward()
    assert torch.isfinite(t.grad).all() and (t.grad[1, :2] == 0).all()
    return dict(identity_without_dark=True, excludes_dark_tokens=True, all_dark_fallback=True, finite_gradients=True)


def cpu_tests():
    print({'masked_pool_tests': masked_pool_tests()}, flush=True)
    return FROZEN_CPU_TESTS()


R.source_files = source_files
R.cpu_tests = cpu_tests

if __name__ == '__main__':
    R.main()
