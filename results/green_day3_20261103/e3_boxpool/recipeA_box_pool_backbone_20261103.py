"""Recipe A regression with std pooling restricted to the detector cluster window (green day 3 / E3, research, 2026-11-03).
Identical to the recipe A member (phase2_backbone_ssl_regression_20261020 + std pooling) except the pooling mode
'std_box': the std statistic is taken over the patch tokens whose 8x8 input patch lies (>= 50 %) inside the
538x408 window centred on the compact stride-2 detector's cluster (experiments/green_day3_20261103/e3_boxpool/
cluster_boxes.npz, built from green day-2 D1; rows without a window use the full frame).  The box is carried as a
second input channel by the frame dataset and split off again in b.inputs, so the encoder input is unchanged.
Applied identically in training and evaluation.  Based on recipeA_masked_pool_backbone_20261031.py.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if not (HERE / 'phase2_backbone_ssl_regression_20261020.py').exists():
    sys.path.insert(0, '/workspace/clear-ec-research/experiments/phase2_pooling_replication_20261026/f0_s46_std/source')
import phase2_pooling_20261019 as pooling
import phase2_foundation_train_20261002 as b
import phase2_backbone_ssl_regression_20261020 as R

WRAPPER = Path(__file__).resolve()
BOX_TABLE = Path('/workspace/clear-ec-research/experiments/green_day3_20261103/e3_boxpool/cluster_boxes.npz')
BOX_FRAC = 0.5
pooling.MODES['std_box'] = ('std',)
FROZEN_FORWARD = pooling.SpatialPoolEncoder.forward
FROZEN_SOURCES = R.source_files
FROZEN_CPU_TESTS = R.cpu_tests
FROZEN_GETITEM = b.Frames.__getitem__
FROZEN_INPUTS = b.inputs
_BOXES = None
_CURRENT = {'mask': None}


def boxes():
    global _BOXES
    if _BOXES is None:
        _BOXES = np.load(BOX_TABLE)['box'].astype(np.float32)
    return _BOXES


def box_channel(index, h, w):
    x0, y0, x1, y1 = boxes()[int(index)]
    m = torch.zeros(1, h, w)
    m[:, int(max(0, y0)):int(min(h, y1)), int(max(0, x0)):int(min(w, x1))] = 1.0
    return m


def frames_getitem(self, row):
    x, y, index = FROZEN_GETITEM(self, row)
    return torch.cat([x, box_channel(index, x.shape[-2], x.shape[-1])], 0), y, index


def inputs(x, architecture):
    if x.shape[1] == 2:
        img, box = x[:, :1], x[:, 1:2]
        _CURRENT['mask'] = F.pad(box.cuda(non_blocking=True).float(), (0, 16, 0, 20), mode='constant', value=0.0)
    else:
        img = x; _CURRENT['mask'] = None
    return FROZEN_INPUTS(img, architecture)


def box_std(tokens, mask):
    B, N, C = tokens.shape
    t = tokens.float()
    if mask is None:
        return pooling.pool_tokens(tokens, 'std')
    H, W = mask.shape[-2:]
    p = int(round((H * W / N) ** 0.5))
    assert (H // p) * (W // p) == N, (H, W, N, p)
    keep = (F.avg_pool2d(mask[:, :1], p).flatten(1) >= BOX_FRAC).float()
    keep[keep.sum(1) == 0] = 1.0
    w = keep[..., None]
    n = w.sum(1)
    mean = (t * w).sum(1) / n
    var = (((t - mean[:, None]) ** 2) * w).sum(1) / n
    return (var + pooling.STD_EPS).sqrt().to(tokens.dtype)


def forward(self, x):
    if self.mode != 'std_box':
        return FROZEN_FORWARD(self, x)
    return box_std(self.original.forward_features(x)['x_norm_patchtokens'], _CURRENT['mask'])


pooling.SpatialPoolEncoder.forward = forward
b.Frames.__getitem__ = frames_getitem
b.inputs = inputs


def source_files():
    return FROZEN_SOURCES() + [WRAPPER]


def box_pool_tests():
    g = torch.Generator().manual_seed(0)
    tok = torch.randn(3, 12, 5, generator=g, dtype=torch.float64)
    ref = pooling.pool_tokens(tok, 'std')
    torch.testing.assert_close(box_std(tok, None), ref)                                   # no box -> identity
    full = torch.ones(3, 1, 6, 8, dtype=torch.float64)
    torch.testing.assert_close(box_std(tok, full), ref)                                   # full-frame box -> identity
    m = torch.ones(3, 1, 6, 8, dtype=torch.float64); m[1, :, 0:2, 0:4] = 0.0              # image 1: two patches outside
    keep = torch.ones(12, dtype=torch.bool); keep[[0, 1]] = False
    exp1 = (tok[1][keep].var(0, unbiased=False) + pooling.STD_EPS).sqrt()
    out = box_std(tok, m)
    torch.testing.assert_close(out[1], exp1); torch.testing.assert_close(out[[0, 2]], ref[[0, 2]])
    empty = torch.zeros(3, 1, 6, 8, dtype=torch.float64)
    torch.testing.assert_close(box_std(tok, empty), ref)                                  # empty box -> all tokens
    t = tok.clone().requires_grad_(True)
    box_std(t, m).sum().backward()
    assert torch.isfinite(t.grad).all() and (t.grad[1, :2] == 0).all()
    bx = boxes(); assert bx.shape == (8954, 4) and np.isfinite(bx).all()
    ch = box_channel(0, 972, 1296); assert ch.shape == (1, 972, 1296) and 0 < ch.mean() <= 1
    return dict(identity_without_box=True, identity_full_box=True, excludes_outside_tokens=True, empty_box_fallback=True, finite_gradients=True, box_table_rows=int(bx.shape[0]))


def cpu_tests():
    print({'box_pool_tests': box_pool_tests()}, flush=True)
    return FROZEN_CPU_TESTS()


R.source_files = source_files
R.cpu_tests = cpu_tests

if __name__ == '__main__':
    R.main()
