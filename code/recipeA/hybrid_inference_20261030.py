"""Submission-1 candidate: hybrid of the Phase-1 six-member package and three full-data recipe A members.

Per image (uint8 grayscale, native resolution):
  phase1  = Phase-1 app predict(): six crop-based CPSAM members, 40 views each, equal mean, fixed Phase-1 scalars
  recipeA = mean over three whole-frame CPDINO members (seeds 46/47/48, all-data SSL5 encoder, std pooling) of
            the four-flip TTA prediction (flips averaged in standardized-log space, decoded with each member's
            training log mean/std), then per-target scalars fitted on the five-fold OOF
  output  = 0.5 * phase1 + 0.5 * recipeA   (equal weight, pre-registered)
Attention runs through PyTorch SDPA with the flash and cuDNN backends disabled so the same kernels are used on
sm75 (T4); the math-only backend would need 18.5 GiB and is refused at start-up.
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = Path(__file__).resolve().parent
PHASE1_APP = HERE / 'phase1_app'
RECIPEA_SRC = HERE / 'recipea_src'
for p in (str(PHASE1_APP), str(RECIPEA_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

RECIPEA_SCALARS = np.array([0.9908461289079755, 0.9869076334979047, 0.9929561461961562])
HYBRID_WEIGHT = 0.5
FLIPS = ((False, False), (True, False), (False, True), (True, True))
BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
PAD = (0, 16, 0, 20)


def assert_memory_efficient_sdpa():
    """Refuse to run if the memory-efficient kernel cannot serve the whole-frame attention."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    q = torch.randn(1, 12, 20336, 64, device='cuda', dtype=torch.float16)
    try:
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            F.scaled_dot_product_attention(q, q, q)
    except RuntimeError as e:
        raise RuntimeError(f'memory-efficient SDPA unavailable for 20336 tokens: {e}')
    del q
    torch.cuda.empty_cache()


def preprocess_frame(image):
    """Recipe A input: CLAHE(2.0, grid 1296//32 x 972//32) + 1/99 percentile normalisation, 1 x 972 x 1296 float32."""
    import cv2
    cv2.setNumThreads(1)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(f'Expected uint8 grayscale, got {image.shape} {image.dtype}')
    if image.shape != (972, 1296):
        raise ValueError(f'Recipe A was trained on 972x1296 frames only; got {image.shape}')
    a = cv2.createCLAHE(2.0, (1296 // 32, 972 // 32)).apply(np.ascontiguousarray(image))
    lo, hi = np.percentile(a, (1, 99))
    a = np.clip((a.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
    return torch.from_numpy(np.ascontiguousarray(a[None, None]))  # 1 x 1 x H x W (batch, channel)


def recipea_inputs(x):
    x = x.cuda(non_blocking=True).expand(-1, 3, -1, -1)
    return F.pad(x, PAD, mode='replicate').contiguous()


class RecipeAMember:
    def __init__(self, checkpoint, device='cuda'):
        import phase2_foundation_preflight_20261002 as preflight
        import phase2_pooling_20261019 as pooling
        ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
        net, _, _ = preflight.build('cpdino_vitb', False, native_convnext_norm=True)
        net[0] = pooling.SpatialPoolEncoder(net[0].original, 'std')
        net.load_state_dict(ck['model'], strict=True)
        self.net = net.to(device).eval().requires_grad_(False)
        self.log_mean = torch.tensor(ck['log_mean'], dtype=torch.float32, device=device)
        self.log_std = torch.tensor(ck['log_std'], dtype=torch.float32, device=device)
        self.epoch = ck['epoch']

    @torch.no_grad()
    def predict(self, x, dtype):
        zs = []
        for h, v in FLIPS:
            xf = x
            if h:
                xf = torch.flip(xf, (-1,))
            if v:
                xf = torch.flip(xf, (-2,))
            with sdpa_kernel(BACKENDS), torch.autocast('cuda', dtype=dtype):
                zs.append(self.net(recipea_inputs(xf)).float())
        z = torch.stack(zs).mean(0)
        return torch.exp(z * self.log_std + self.log_mean)[0].cpu().numpy()


def predict_recipea(image, member_paths, dtype=torch.float16):
    x = preprocess_frame(image)
    values = []
    for path in member_paths:
        member = RecipeAMember(path)
        values.append(member.predict(x, dtype))
        del member
        torch.cuda.empty_cache()
    raw = np.mean(values, 0)
    return raw * RECIPEA_SCALARS, np.asarray(values)


def predict_hybrid(image, phase1_model_dir, phase1_config, recipea_members, dtype=torch.float16, return_parts=False):
    import inference as phase1
    p1, p1_members = phase1.predict(image, Path(phase1_model_dir), phase1_config, dtype=dtype, return_members=True)
    ra, ra_members = predict_recipea(image, recipea_members, dtype)
    out = HYBRID_WEIGHT * ra + (1 - HYBRID_WEIGHT) * p1
    if not np.isfinite(out).all() or not (out > 0).all():
        raise ValueError('Invalid hybrid output')
    if return_parts:
        return out, dict(phase1=p1, phase1_members=p1_members, recipea=ra, recipea_members=ra_members)
    return out
