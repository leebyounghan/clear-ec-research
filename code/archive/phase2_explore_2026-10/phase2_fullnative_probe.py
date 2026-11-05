#!/usr/bin/env python3
"""Bounded feasibility probe for a truly uncropped CLEAR-EC CPSAM input.

One native 972x1296 image is padded by four rows (never resized or tiled),
and its entire 61x81 patch grid enters each original CPSAM attention block.
This is a memory/timing probe, not a validation score or candidate release.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path("datasets/clear_ec_fullframe_20260904")
OUT = Path("experiments/phase2_fullnative_20260929/probe.json")


class FullNativeEncoder(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.encoder.patch_embed(image)
        height, width = x.shape[1:3]
        position = self.encoder.pos_embed
        if position is not None:
            if position.shape[1:3] != (height, width):
                position = F.interpolate(
                    position.permute(0, 3, 1, 2), size=(height, width),
                    mode="bicubic", align_corners=False,
                ).permute(0, 2, 3, 1)
            x = x + position
        for block in self.encoder.blocks:
            x = checkpoint(block, x, use_reentrant=False)
        return self.encoder.neck(x.permute(0, 3, 1, 2))


class FullNativeRegressor(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = FullNativeEncoder(encoder)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, 3)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self.pool(self.encoder(image)).flatten(1)))


def image_tensor() -> torch.Tensor:
    with np.load(ROOT / "fullframe_meta.npz", allow_pickle=True) as data:
        n = int(data["N"])
    raw = np.memmap(ROOT / "img_full_raw.dat", mode="r", dtype=np.uint8,
                    shape=(n, 972, 1296))[0]
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(np.ascontiguousarray(raw))
    low, high = np.percentile(clahe, (1, 99))
    normalized = np.clip((clahe.astype(np.float32) - low) / (high - low + 1e-6), 0, 1)
    padded = np.pad(normalized, ((0, 4), (0, 0)), mode="edge")
    if padded.shape != (976, 1296):
        raise ValueError("Unexpected source image geometry")
    return torch.from_numpy(np.stack([padded] * 3)[None]).cuda()


def main() -> None:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Probe requires one isolated GPU")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from cellpose import models

    original = models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
    encoder = original.net.encoder.float()
    del original
    net = FullNativeRegressor(encoder).cuda().train()
    image = image_tensor()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        predicted = net(image)
        loss = (predicted.float() - torch.tensor([[0.2, -0.1, 0.3]], device="cuda")).square().mean()
    torch.cuda.synchronize()
    forward_s = time.monotonic() - start
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    result = {
        "geometry": [972, 1296],
        "model_input": [976, 1296],
        "resized": False,
        "cropped_or_tiled": False,
        "patch_grid": [61, 81],
        "attention_blocks": len(encoder.blocks),
        "forward_seconds": forward_s,
        "full_step_seconds": time.monotonic() - start,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "loss": float(loss.detach()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
