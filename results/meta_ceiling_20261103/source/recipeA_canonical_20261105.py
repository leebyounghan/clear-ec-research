"""Recipe A with a canonical cell size (2026-11-05). Cellpose normalises every image to one cell diameter before its network sees it; nobody did that here, although the encoder was pretrained that way and cells range from about 17 to 40 px. Token size relative to the cell changes with density, so topology is read at a different resolution in every frame, and cell size is mixed into everything the model reads (zoom response of CD -1.42, physics -2).
The frozen phase2_pooling_regression_20261019 runner with std pooling, except: each frame, train and validation, is zoomed about its centre on the raw frame before CLAHE by s = D0 / d, d = median nearest-neighbour distance of its pre-only peaks >= 0.35 (label-free, available at test time), clipped to [0.7, 1.45], so cells get one spacing. The CD target becomes CD / s^2, the label of the zoomed frame; CV and HEX unchanged. MAPE is unchanged by that rescaling, since prediction and target move together. Decoded CD = prediction x s^2.
Read: terminal fold 0 against the matched seed-46 member. Protocol in canonical/."""
from pathlib import Path

import cv2
import numpy as np
import torch

import phase2_pooling_regression_20261019 as R

WRAPPER = Path(__file__).resolve()
SPACING = Path('/workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep/spacing_t35.npz')
H, W = 972, 1296
with np.load(SPACING) as z:
    SCALE = z['scale'].astype(np.float64)
    D0 = float(z['d0'])
FROZEN_ARRAYS = R.arrays
FROZEN_SOURCES = R.source_files
FROZEN_CPU_TESTS = R.cpu_tests
FROZEN_SAVE = R.save


def zoom(a, s):
    M = np.array([[s, 0, (1 - s) * (W - 1) / 2], [0, s, (1 - s) * (H - 1) / 2]], np.float32)
    return cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def arrays():
    ids, y, tr, va = FROZEN_ARRAYS()
    y = y.copy()
    y[:, 0] = (y[:, 0] / SCALE ** 2).astype(y.dtype)
    return ids, y, tr, va


def getitem(self, row):
    cv2.setNumThreads(1)
    if self.raw is None:
        self.raw = np.memmap(R.b.RAW, mode='r', dtype=np.uint8, shape=(self.total, H, W))
    idx = int(self.indices[row])
    a = cv2.createCLAHE(2.0, (W // 32, H // 32)).apply(zoom(np.asarray(self.raw[idx]), SCALE[idx]))
    lo, hi = np.percentile(a, (1, 99))
    a = np.clip((a.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
    return torch.from_numpy(np.ascontiguousarray(a[None])), torch.from_numpy(self.targets[idx]), idx


def source_files():
    return FROZEN_SOURCES() + [WRAPPER, SPACING]


def save(path, obj):
    if Path(path).name == 'manifest.json' and isinstance(obj, dict) and 'recipe' in obj:
        obj['recipe']['canonical'] = dict(d0=D0, clip=[0.7, 1.45], spacing='median NN of pre-only peaks >= 0.35', decode='CD = prediction x s^2')
    return FROZEN_SAVE(path, obj)


def canonical_tests():
    assert SCALE.shape == (8954,) and SCALE.min() >= 0.7 - 1e-9 and SCALE.max() <= 1.45 + 1e-9 and np.isfinite(SCALE).all()
    ids0, y0, _, _ = FROZEN_ARRAYS()
    ids, y, tr, va = arrays()
    s = y0[:, 0] > 0
    assert np.allclose(y[s, 0] * SCALE[s] ** 2, y0[s, 0], rtol=1e-4) and np.array_equal(y[:, 1:], y0[:, 1:])
    x, t, i = R.Frames(tr[:2], y, len(ids), 3, True, 46)[0]
    assert x.shape == (1, H, W) and abs(float(t[0]) - float(y[i, 0])) < 1e-3 and 0 <= float(x.min()) and float(x.max()) <= 1
    p = np.array([[1000.0, 0.3, 0.5]]); q = np.array([[1100.0, 0.3, 0.5]]); c = SCALE[0] ** 2
    assert abs(abs(p[0, 0] / c - q[0, 0] / c) / (q[0, 0] / c) - abs(p[0, 0] - q[0, 0]) / q[0, 0]) < 1e-12
    return dict(d0=D0, scale_p5_p50_p95=[round(float(v), 3) for v in np.percentile(SCALE, (5, 50, 95))], log_sd_cd_before=round(float(np.log(y0[s, 0]).std()), 4), log_sd_cd_after=round(float(np.log(y[s, 0]).std()), 4))


def cpu_tests():
    print({'canonical_tests': canonical_tests()}, flush=True)
    return FROZEN_CPU_TESTS()


R.b.Frames.__getitem__ = getitem
R.arrays = arrays
R.source_files = source_files
R.save = save
R.cpu_tests = cpu_tests

if __name__ == '__main__':
    R.main()
