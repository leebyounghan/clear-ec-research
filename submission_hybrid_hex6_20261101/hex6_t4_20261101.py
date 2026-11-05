"""T4-compatible wrapper around the green-track Hex6Corrector (submission 2, 2026-11-01).

The verified module (hex6/deploy/hex6_correction.py, ten ACTIVE detectors, a/b/w frozen) is not edited. This
subclass changes only how each detector encoder is run: autocast dtype is a parameter (fp16 on the T4, which has
no bf16 tensor cores; the green track validated bf16) and SDPA is restricted to the memory-efficient kernel with
math as the last resort, exactly as the hybrid does, because the 20,336-token attention needs 18.5 GiB with the
math kernel. Peak finding, bounded Voronoi, the log-hex6 feature and the blend are inherited unchanged.
With dtype=torch.bfloat16 the override must reproduce the parent class exactly (checked in the pod test).
"""
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = Path(__file__).resolve().parent
DEPLOY = HERE / 'hex6' / 'deploy'
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))
import hex6_correction as H6

BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


class Hex6CorrectorT4(H6.Hex6Corrector):
    def __init__(self, deploy_dir=DEPLOY, device='cuda', dtype=torch.float16):
        super().__init__(deploy_dir, device)
        self.dtype = dtype

    @torch.no_grad()
    def hex6_per_detector(self, image_u8):
        x = H6.preprocess(image_u8)
        out = {}
        for name, kind, features, enc, head in self.detectors:
            with sdpa_kernel(BACKENDS), torch.autocast('cuda', dtype=self.dtype):
                tok = H6.S8.tokens_of(enc, x, features)
            if kind == 'sg2':
                p = head(tok).sigmoid()[0].float().cpu().numpy()
                pk = H6.SG.peaks2(p, self.support2, 0.35)
                pts = pk[:, :2].astype(np.float64)
            else:
                p = head(tok.float()).squeeze(-1).sigmoid().reshape(*H6.GRID8).cpu().numpy()
                pts = H6.peaks_s8(p)
            out[name] = H6.hex6_all(pts)
        return out
