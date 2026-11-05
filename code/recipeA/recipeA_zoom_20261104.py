"""Recipe A with a physics-labelled zoom augmentation (2026-11-04). Zooming a frame by s makes each cell s times wider, so CD becomes CD / s^2 and CV, HEX do not change. The label
transform is exact and needs no annotation, so each frame teaches what cell size means for CD and that CV and HEX do not depend on it. Training has used flips only.
The frozen phase2_pooling_regression_20261019 runner with std pooling, except: during training each frame gets s log-uniform in [1 / ZMAX, ZMAX], applied to the raw frame about its centre
(linear interpolation, reflected border) before CLAHE and the percentile scaling, deterministic in (seed, epoch, row); the row's CD target is divided by s^2. Flips as before.
Validation, the SSL encoder, the schedule and the loss are unchanged. Read: terminal fold 0 against the matched seed-46 member at s = 1, protocol in zoom/."""
import math
from pathlib import Path

import cv2
import numpy as np
import torch

import phase2_pooling_regression_20261019 as R

WRAPPER = Path(__file__).resolve()
ZMAX = 1.25
H, W = 972, 1296
FROZEN_SOURCES = R.source_files
FROZEN_CPU_TESTS = R.cpu_tests
FROZEN_SAVE = R.save
FrozenFrames = R.Frames


def zoom(a, s):
    M = np.array([[s, 0, (1 - s) * (W - 1) / 2], [0, s, (1 - s) * (H - 1) / 2]], np.float32)
    return cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def scale_for(seed, epoch, index):
    return math.exp(np.random.default_rng(np.random.SeedSequence([seed, 77, epoch, index])).uniform(-math.log(ZMAX), math.log(ZMAX)))


class Frames(FrozenFrames):
    def __getitem__(self, row):
        if not self.use_augmentation:
            return super().__getitem__(row)
        cv2.setNumThreads(1)
        if self.raw is None:
            self.raw = np.memmap(R.b.RAW, mode='r', dtype=np.uint8, shape=(self.total, H, W))
        index = int(self.indices[row])
        s = scale_for(self.seed, self.epoch, index)
        a = cv2.createCLAHE(2.0, (W // 32, H // 32)).apply(zoom(np.asarray(self.raw[index]), s))
        lo, hi = np.percentile(a, (1, 99))
        a = np.clip((a.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
        x = torch.from_numpy(np.ascontiguousarray(a[None]))
        horizontal, vertical = R.flip_decisions(self.seed, self.epoch, index)
        if horizontal:
            x = torch.flip(x, (-1,))
        if vertical:
            x = torch.flip(x, (-2,))
        y = np.array(self.targets[index], copy=True)
        y[0] = y[0] / (s * s)
        return x, torch.from_numpy(y), index


def source_files():
    return FROZEN_SOURCES() + [WRAPPER]


def save(path, obj):
    if Path(path).name == 'manifest.json' and isinstance(obj, dict) and 'recipe' in obj:
        obj['recipe']['zoom'] = dict(zmax=ZMAX, label='CD / s^2; CV, HEX unchanged', interpolation='linear', border='reflect101', applied='raw frame, before CLAHE')
    return FROZEN_SAVE(path, obj)


def zoom_tests():
    a = np.tile(np.arange(W, dtype=np.uint8), (H, 1))
    assert np.array_equal(zoom(a, 1.0), a)
    dot = np.zeros((H, W), np.uint8)
    dot[H // 2 - 100:H // 2 + 100, W // 2 - 100:W // 2 + 100] = 255
    area0, area1 = (dot > 127).sum(), (zoom(dot, 1.25) > 127).sum()
    assert abs(area1 / area0 - 1.25 ** 2) < 0.03, (area0, area1)                 # zoom s widens by s, area by s^2
    v = [scale_for(46, 3, i) for i in range(2000)]
    assert min(v) >= 1 / ZMAX - 1e-6 and max(v) <= ZMAX + 1e-6 and abs(np.mean(np.log(v))) < 0.02
    assert scale_for(46, 3, 5) == scale_for(46, 3, 5) and scale_for(46, 3, 5) != scale_for(46, 4, 5)
    ids, y, tr, va = R.arrays()
    ds = Frames(tr[:2], y, len(ids), 3, True, 46)
    x, t, index = ds[0]
    s = scale_for(46, 3, index)
    assert x.shape == (1, H, W) and abs(float(t[0]) - float(y[index, 0]) / s ** 2) < 1e-2 and float(t[1]) == float(y[index, 1]) and float(t[2]) == float(y[index, 2])
    plain = Frames(tr[:2], y, len(ids), 3, False, 46)
    assert float(plain[0][1][0]) == float(y[index, 0])
    return dict(zmax=ZMAX, area_ratio_at_1p25=round(float(area1 / area0), 3))


def cpu_tests():
    print({'zoom_tests': zoom_tests()}, flush=True)
    return FROZEN_CPU_TESTS()


R.Frames = Frames
R.source_files = source_files
R.save = save
R.cpu_tests = cpu_tests

if __name__ == '__main__':
    R.main()
