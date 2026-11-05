"""Recipe A with cell maps as input (2026-11-05). The three input channels have always been copies of one grey frame; nothing told the encoder where cells are, and its predictions move with blur and noise while cells do not. Two label-free maps now enter: dark ridges (cell borders) and rendered pre-only peaks (cell centres).
The frozen phase2_pooling_regression_20261019 runner with std pooling, except: the patch projection gains a zero-initialised Conv2d for the two maps, same kernel and stride, added to the grey projection, so training starts exactly at recipe A and the maps enter only as far as they help. The new layer trains at 1e-4 (constant), the encoder keeps 1e-5. Flips and padding apply to all channels. Targets, validation, the SSL encoder, the schedule and the loss are unchanged.
Read: terminal fold 0 against the matched seed-46 member, and error correlation with recipe A. Protocol in cellmap/."""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import phase2_pooling_regression_20261019 as R

WRAPPER = Path(__file__).resolve()
MAPS = Path('/workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep/cellmaps_u8.dat')
H, W, H2, W2 = 972, 1296, 486, 648
EXTRA_LR = 1e-4
FROZEN_SETUP = R.setup
FROZEN_SOURCES = R.source_files
FROZEN_CPU_TESTS = R.cpu_tests
FROZEN_SAVE = R.save
FROZEN_GET = R.b.Frames.__getitem__


class CellProj(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.extra = torch.nn.Conv2d(2, base.out_channels, kernel_size=base.kernel_size, stride=base.stride, padding=base.padding)
        torch.nn.init.zeros_(self.extra.weight)
        torch.nn.init.zeros_(self.extra.bias)

    def forward(self, x):
        return self.base(x[:, :3]) + self.extra(x[:, 3:])


def getitem(self, row):
    x, t, idx = FROZEN_GET(self, row)
    if not hasattr(self, 'maps'):
        self.maps = np.memmap(MAPS, dtype=np.uint8, mode='r', shape=(8954, 2, H2, W2))
    m = np.stack([cv2.resize(self.maps[idx, c].astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_LINEAR) for c in range(2)])
    return torch.cat([x, torch.from_numpy(m)], 0), t, idx


def inputs(x, architecture):
    x = x.cuda(non_blocking=True)
    extra = x[:, 1:] if x.shape[1] == 3 else torch.zeros_like(x).repeat(1, 2, 1, 1)
    return F.pad(torch.cat([x[:, :1].expand(-1, 3, -1, -1), extra], 1), (0, 16, 0, 20), mode='replicate').contiguous()


def setup(m):
    net, opt, mu, sigma, details = FROZEN_SETUP(m)
    pe = net[0].original.patch_embed
    pe.proj = CellProj(pe.proj).cuda()
    opt.add_param_group({'params': list(pe.proj.extra.parameters()), 'lr': EXTRA_LR})
    details['cellmap'] = dict(extra_lr=EXTRA_LR, channels=['dark ridge', 'rendered pre-only peaks'], init='zero')
    return net, opt, mu, sigma, details


def source_files():
    return FROZEN_SOURCES() + [WRAPPER]


def save(path, obj):
    if Path(path).name == 'manifest.json' and isinstance(obj, dict) and 'recipe' in obj:
        obj['recipe']['cellmap'] = dict(extra_lr=EXTRA_LR, channels=['dark ridge', 'rendered pre-only peaks'], init='zero', maps=str(MAPS))
    return FROZEN_SAVE(path, obj)


def cellmap_tests():
    base = torch.nn.Conv2d(3, 8, kernel_size=16, stride=8, padding=4)
    p = CellProj(base)
    x = torch.randn(2, 5, 992, 1312)
    assert p(x).shape == (2, 8, 124, 164)
    torch.testing.assert_close(p(x), base(x[:, :3]))                 # zero-initialised: starts at recipe A
    assert np.memmap(MAPS, dtype=np.uint8, mode='r', shape=(8954, 2, H2, W2))[0, 0].max() > 0
    ids, y, tr, va = R.arrays()
    v, t, i = R.Frames(tr[:2], y, len(ids), 3, True, 46)[0]
    assert v.shape == (3, H, W) and 0 <= float(v.min()) and float(v.max()) <= 1 and float(v[1:].max()) > 0
    return dict(frame_channels=int(v.shape[0]))


def cpu_tests():
    print({'cellmap_tests': cellmap_tests()}, flush=True)
    return FROZEN_CPU_TESTS()


R.b.Frames.__getitem__ = getitem
R.b.inputs = inputs
R.setup = setup
R.source_files = source_files
R.save = save
R.cpu_tests = cpu_tests

if __name__ == '__main__':
    R.main()
