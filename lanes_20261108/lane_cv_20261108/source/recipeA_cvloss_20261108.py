"""CV-weighted recipe A on one GPU (CV lane, 2026-11-08). Same frozen runner and one-process wrapper as lane GPU 0's arms; only the loss changes:
the per-metric recipe A loss (0.5 |z - zt| + 2.5 |expm1((z - zt) sigma)|) is weighted per column by LOSS_W from the environment (CD,CV,HEX), normalised to
mean 1, e.g. LOSS_W=0.25,2.5,0.25 (CV-weighted) or 0,3,0 (CV only). Everything else (SSL5 init, std pooling, schedule, flips, validation) unchanged."""
import os
from pathlib import Path
import torch
import phase2_foundation_train_20261002 as b
import phase2_pooling_regression_20261019 as R
import recipeA_single_gpu_20261106 as SG

WRAPPER = Path(__file__).resolve()
_S = R.source_files
R.source_files = lambda: _S() + [WRAPPER]
_W = [float(v) for v in os.environ['LOSS_W'].split(',')]; assert len(_W) == 3 and sum(_W) > 0
_W = [3 * v / sum(_W) for v in _W]


def loss_w(z, targets, mu, sigma):
    zt = (targets.log() - mu) / sigma; d = z.float() - zt
    per = 0.5 * d.abs() + 2.5 * torch.expm1(d * sigma).abs()
    return (per * torch.tensor(_W, device=per.device, dtype=per.dtype)).mean()


b.loss_fn = loss_w

if __name__ == '__main__':
    print('LOSS_W normalised', _W, flush=True)
    R.main()
