"""Small, import-safe training changes for the finite September 4 method screen."""
from __future__ import annotations

import math

import torch


def cosine_multiplier(step, total_steps, warmup_steps=200, floor=0.05):
    if total_steps <= 0 or warmup_steps <= 0 or not 0 <= floor <= 1:
        raise ValueError("invalid schedule")
    warmup_steps = min(warmup_steps, max(1, total_steps // 10))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def regression_loss(output, target, log_scale, objective):
    # Float32 exponentiation avoids reduced-precision loss quantization. Training
    # targets are strictly positive; zero-label exclusion remains in the trainer.
    delta = output.float() - target.float()
    log_l1 = delta.abs().mean()
    if objective == "log_l1":
        return log_l1
    if objective != "mixed_mape":
        raise ValueError(objective)
    relative_error = torch.expm1(delta * log_scale).abs().mean()
    # Predeclared common scale, NOT a fitted or per-metric weight. MAPE remains
    # equally weighted across CD/CV/HEX; log-L1 supplies a stabilizing term.
    return 0.5 * log_l1 + 0.5 * relative_error / 0.2


class SAMPerturbation:
    """SAM's unscaled global L2 ascent, with exact restoration on exceptions.

    AdamW performs the descent outside this context. No batch normalization is
    used in the scalar model. The caller replays dropout RNG for the second pass.
    """

    def __init__(self, parameters, rho):
        if rho <= 0:
            raise ValueError("rho must be positive")
        self.parameters = [p for p in parameters if p.requires_grad and p.grad is not None]
        self.rho = rho
        self.saved = []

    @torch.no_grad()
    def __enter__(self):
        if not self.parameters:
            raise RuntimeError("SAM requires first-pass gradients")
        norm = torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in self.parameters]))
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite SAM gradient norm")
        scale = self.rho / (norm + 1e-12)
        try:
            for parameter in self.parameters:
                self.saved.append((parameter, parameter.detach().clone()))
                parameter.add_(parameter.grad * scale)
        except BaseException:
            self.restore()
            raise
        return self

    @torch.no_grad()
    def restore(self):
        for parameter, original in self.saved:
            parameter.copy_(original)
        self.saved.clear()

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore()
        return False
