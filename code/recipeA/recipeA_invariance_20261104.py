"""Recipe A with label-preserving perturbations in training (2026-11-04). Blur, exposure, noise and small rotations do not change the cells, so they cannot change the labels, yet a trained member's predictions move a lot:
blur sigma 1 px: CV +6.8 %, HEX -5.8 %; sigma 2.5 px: CV +12 %, HEX -10 %; noise sigma 6: CD -6.8 % (SD 7.9 %). The label SD of log CV is 0.16. Training has used flips only, so sharpness and noise are free shortcuts.
The frozen phase2_pooling_regression_20261019 runner with std pooling, except: during training each frame gets, on the raw frame before CLAHE and the percentile scaling, deterministic in (seed, epoch, row):
rotation uniform +-10 deg (reflected border); gamma log-uniform in [0.8, 1.25]; Gaussian blur sigma uniform [0.3, 2.0] with probability 0.5; Gaussian noise sigma uniform [1, 7] with probability 0.5. Flips as before. Targets, validation, the SSL encoder, the schedule and the loss are unchanged.
Read: terminal fold 0 against the matched seed-46 member, and the trained member's response to the same perturbations, protocol in invariance/."""
from pathlib import Path

import cv2
import numpy as np
import torch

import phase2_pooling_regression_20261019 as R

WRAPPER = Path(__file__).resolve()
H, W = 972, 1296
FROZEN_SOURCES = R.source_files
FROZEN_CPU_TESTS = R.cpu_tests
FROZEN_SAVE = R.save
FrozenFrames = R.Frames
CONFIG = dict(rotation_deg=10.0, gamma=(0.8, 1.25), blur=dict(p=0.5, sigma=(0.3, 2.0)), noise=dict(p=0.5, sigma=(1.0, 7.0)))


def perturb(a, rng):
    angle = rng.uniform(-CONFIG['rotation_deg'], CONFIG['rotation_deg'])
    a = cv2.warpAffine(a, cv2.getRotationMatrix2D(((W - 1) / 2, (H - 1) / 2), angle, 1.0), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    g = np.exp(rng.uniform(np.log(CONFIG['gamma'][0]), np.log(CONFIG['gamma'][1])))
    a = cv2.LUT(a, np.clip(255 * (np.arange(256) / 255.0) ** g, 0, 255).astype(np.uint8))
    if rng.random() < CONFIG['blur']['p']:
        a = cv2.GaussianBlur(a, (0, 0), rng.uniform(*CONFIG['blur']['sigma']))
    if rng.random() < CONFIG['noise']['p']:
        a = np.clip(a.astype(np.float32) + rng.normal(0, rng.uniform(*CONFIG['noise']['sigma']), a.shape).astype(np.float32), 0, 255).astype(np.uint8)
    return a


class Frames(FrozenFrames):
    def __getitem__(self, row):
        if not self.use_augmentation:
            return super().__getitem__(row)
        cv2.setNumThreads(1)
        if self.raw is None:
            self.raw = np.memmap(R.b.RAW, mode='r', dtype=np.uint8, shape=(self.total, H, W))
        index = int(self.indices[row])
        a = perturb(np.asarray(self.raw[index]), np.random.default_rng(np.random.SeedSequence([self.seed, 88, self.epoch, index])))
        a = cv2.createCLAHE(2.0, (W // 32, H // 32)).apply(a)
        lo, hi = np.percentile(a, (1, 99))
        a = np.clip((a.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
        x = torch.from_numpy(np.ascontiguousarray(a[None]))
        horizontal, vertical = R.flip_decisions(self.seed, self.epoch, index)
        if horizontal:
            x = torch.flip(x, (-1,))
        if vertical:
            x = torch.flip(x, (-2,))
        return x, torch.from_numpy(self.targets[index]), index


def source_files():
    return FROZEN_SOURCES() + [WRAPPER]


def save(path, obj):
    if Path(path).name == 'manifest.json' and isinstance(obj, dict) and 'recipe' in obj:
        obj['recipe']['invariance'] = CONFIG
    return FROZEN_SAVE(path, obj)


def invariance_tests():
    a = (np.random.default_rng(0).random((H, W)) * 255).astype(np.uint8)
    assert perturb(a, np.random.default_rng(1)).shape == (H, W) and perturb(a, np.random.default_rng(1)).dtype == np.uint8
    assert np.array_equal(perturb(a, np.random.default_rng(1)), perturb(a, np.random.default_rng(1)))
    assert not np.array_equal(perturb(a, np.random.default_rng(1)), perturb(a, np.random.default_rng(2)))
    ids, y, tr, va = R.arrays()
    x, t, index = Frames(tr[:2], y, len(ids), 3, True, 46)[0]
    assert x.shape == (1, H, W) and np.array_equal(t.numpy(), y[index]) and 0 <= float(x.min()) and float(x.max()) <= 1
    x0, _, _ = Frames(tr[:2], y, len(ids), 4, True, 46)[0]
    assert not torch.equal(x, x0)
    plain, plain2 = FrozenFrames(tr[:2], y, len(ids), 3, False, 46)[0], Frames(tr[:2], y, len(ids), 3, False, 46)[0]
    assert torch.equal(plain[0], plain2[0])
    return dict(config=CONFIG)


def cpu_tests():
    print({'invariance_tests': invariance_tests()}, flush=True)
    return FROZEN_CPU_TESTS()


R.Frames = Frames
R.source_files = source_files
R.save = save
R.cpu_tests = cpu_tests

if __name__ == '__main__':
    R.main()
