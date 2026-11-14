"""Recipe A with augmentation drawn from the measured noise, focus and exposure of real frames (lane GPU 0, 2026-11-06).

Measured on all 8954 raw frames before CLAHE (frame_stats.npz, calibrated against known blur and noise on 120 frames):
- White noise floor sigma (spectrum, 0.27-0.40 cycles/px): source A 2.59-4.02, source B 2.14-2.80 (p5-p95). Added noise reads back within 0.02.
- A frame with no grain never occurs: minimum sigma 2.0. Gaussian blur alone takes the floor to 0.46 at sigma 1, so blur without refill leaves the real distribution.
- Focus (spectral roll-off, 0.07-0.14 cycles/px): same range in both sources; p5-p95 equals added blur of about sigma 1.7, IQR about 1.2.
- Exposure: median and p99, by source. Noise grows with intensity: exponent 0.4 in A, 0.95 in B.

For each training frame with probability p: pick a donor frame from the same source among this fold's training rows, then, in acquisition order:
blur up to the donor's focus if the donor is softer (no sharpening); match the donor's median and p99 linearly, gain within [0.6, 1.6] and capped so the frame's own grain stays within the larger of the two floors (the uncapped version overshot: A p95 5.57 against a real 4.02); add signal-dependent white noise up to the donor's floor, counting what blur and gain did to the frame's own noise.
Deterministic in (seed, epoch, row). Flips as before. No rotation. Targets, validation, SSL encoder, schedule and loss are unchanged, except encoder_lr_mult when the arm config sets it.
The arm config comes from ARM_CONFIG (JSON) at the first call and is pinned in <root>/arm_config.json."""
import json, os
from pathlib import Path

import cv2
import numpy as np
import torch

import phase2_pooling_regression_20261019 as R
import recipeA_invariance_20261104 as I
import recipeA_single_gpu_20261106 as SG

WRAPPER = Path(__file__).resolve()
STATS = Path('/workspace/clear-ec-research/experiments/lane_gpu0_20261106/frame_aug_stats.npz')
H, W = 972, 1296
KAPPA = (0.4, 0.95)
GAIN = (0.6, 1.6)
FOCUS_MAP = ([0.25, 0.64, 1.88, 3.63], [0.75, 1.0, 1.5, 2.0])  # focus difference to added blur sigma, measured
_SOURCES = R.source_files
_SAVE = R.save
_CPU = I.FROZEN_CPU_TESTS
_S = None


def stats():
    global _S
    if _S is None:
        with np.load(STATS, allow_pickle=True) as z:
            _S = {k: z[k] for k in ('src', 'sigma', 'focus', 'p50', 'p99')}
    return _S


def arm():
    pin = R.ROOT / 'arm_config.json'
    if pin.exists():
        return json.loads(pin.read_text())
    cfg = json.loads(os.environ['ARM_CONFIG'])
    assert set(cfg) <= {'p', 'encoder_lr_mult', 'parts'} and 0 <= cfg['p'] <= 1
    cfg.setdefault('encoder_lr_mult', 1.0)
    cfg.setdefault('parts', ['focus', 'exposure', 'noise'])
    R.ROOT.mkdir(parents=True, exist_ok=True)
    pin.write_text(json.dumps(cfg))
    return cfg


def transfer(a, own, donor, src, rng, parts=('focus', 'exposure', 'noise')):
    """a: uint8 frame. own and donor: indices into the statistics. Returns uint8 frame."""
    s = stats()
    x = a.astype(np.float32)
    sig = float(s['sigma'][own])
    if 'focus' in parts:
        d = float(s['focus'][donor] - s['focus'][own])
        if d > FOCUS_MAP[0][0]:
            b = float(np.interp(d, *FOCUS_MAP))
            x = cv2.GaussianBlur(x, (0, 0), b)
            k = cv2.getGaussianKernel(int(2 * np.ceil(4 * b) + 1), b)
            sig *= float((k ** 2).sum())
    if 'exposure' in parts:
        g = float(np.clip((s['p99'][donor] - s['p50'][donor]) / max(float(s['p99'][own] - s['p50'][own]), 1.0), *GAIN))
        g = min(g, max(float(s['sigma'][donor]), float(s['sigma'][own])) / max(sig, 1e-3))  # gain also scales the frame's own grain: stay inside the measured range
        x = g * x + (float(s['p50'][donor]) - g * float(s['p50'][own]))
        sig *= g
    if 'noise' in parts:
        v = float(s['sigma'][donor]) ** 2 - sig ** 2
        if v > 0:
            amp = (np.maximum(x, 1.0) / max(float(x.mean()), 1.0)) ** KAPPA[src]
            amp /= float(np.sqrt((amp[H // 4:3 * H // 4, W // 4:3 * W // 4] ** 2).mean()))  # the floor is read on the bright centre; normalise there, or vignetted frames overshoot
            x = x + rng.standard_normal(x.shape, dtype=np.float32) * amp * float(np.sqrt(v))
    x[a == 0] = 0  # clipped black stays black: no lifted floor, no grain where real frames have none
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


class Frames(I.Frames):
    def __init__(self, indices, targets, total, epoch=0, augment=False, seed=42):
        super().__init__(indices, targets, total, epoch, augment, seed)
        self.by_source = None

    def __getitem__(self, row):
        if not self.use_augmentation:
            return I.FrozenFrames.__getitem__(self, row)
        cv2.setNumThreads(1)
        if self.raw is None:
            self.raw = np.memmap(R.b.RAW, mode='r', dtype=np.uint8, shape=(self.total, H, W))
        s = stats()
        if self.by_source is None:
            ix = np.sort(np.asarray(self.indices))
            self.by_source = {v: ix[s['src'][ix] == v] for v in (0, 1)}
        index = int(self.indices[row])
        a = np.asarray(self.raw[index])
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 89, self.epoch, index]))
        cfg = ARM
        if rng.random() < cfg['p']:
            src = int(s['src'][index])
            donor = int(rng.choice(self.by_source[src]))
            a = transfer(a, index, donor, src, rng, cfg['parts'])
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


ARM = None


def source_files():
    return _SOURCES() + [WRAPPER, STATS]


def save(path, obj):
    if Path(path).name == 'manifest.json' and isinstance(obj, dict) and 'recipe' in obj:
        cfg = arm()
        I.CONFIG = dict(kind='measured transfer', stats=STATS.name, kappa=KAPPA, gain=GAIN, focus_map=FOCUS_MAP, **cfg)  # I.save writes this into the recipe
        obj['recipe']['encoder_lr'] = 1e-5 * cfg['encoder_lr_mult']
        obj['recipe']['augmentation'] = 'measured focus, exposure and noise transfer from a same-source training frame, then independent deterministic H/V flips'
    return _SAVE(path, obj)


def measured_tests():
    a = (np.random.default_rng(0).random((H, W)) * 255).astype(np.uint8)
    out = transfer(a, 0, 1, int(stats()['src'][0]), np.random.default_rng(1))
    assert out.shape == (H, W) and out.dtype == np.uint8 and np.array_equal(out, transfer(a, 0, 1, int(stats()['src'][0]), np.random.default_rng(1)))
    ids, y, tr, va = R.arrays()
    assert len(stats()['src']) == len(ids)
    with np.load(STATS, allow_pickle=True) as z:
        assert np.array_equal(z['ids'].astype(str), ids), 'statistics follow the cache order'
    # Without augmentation the frames match the frozen transform; with it the targets are untouched, frames differ and stay in range.
    plain, frozen = Frames(tr[:2], y, len(ids), 3, False, 46)[0], I.FrozenFrames(tr[:2], y, len(ids), 3, False, 46)[0]
    assert torch.equal(plain[0], frozen[0])
    x0, t0, i0 = Frames(tr[:8], y, len(ids), 3, True, 46)[0]
    x1, t1, _ = Frames(tr[:8], y, len(ids), 3, True, 46)[0]
    assert torch.equal(x0, x1) and x0.shape == (1, H, W) and 0 <= float(x0.min()) and float(x0.max()) <= 1 and np.array_equal(t0.numpy(), y[i0])
    return dict(arm=arm(), kappa=KAPPA, gain=GAIN)


def cpu_tests():
    print({'measured_tests': measured_tests()}, flush=True)
    return _CPU()


class _Arm(dict):
    """Filled on first read, because R.ROOT is known only inside main()."""
    def __getitem__(self, k):
        if not len(self):
            self.update(arm())
        return dict.__getitem__(self, k)


ARM = _Arm()
R.Frames = Frames
R.source_files = source_files
R.save = save
R.cpu_tests = cpu_tests

if __name__ == '__main__':
    R.main()
