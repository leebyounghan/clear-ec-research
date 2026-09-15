#!/usr/bin/env python3
"""Extract held-out predictions for each C5 crop instead of averaging them.

The checkpoint and validation rows are taken from an existing honest OOF run.  This
lets us test crop aggregation without retraining or leaking a model onto its own rows.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from cellpose import models


ROOT = Path(os.environ.get("ROOT", "datasets/clear_ec_official"))
RUN_DIR = Path(os.environ["RUN_DIR"])
BS = int(os.environ.get("EVAL_IMAGE_BS", "4"))
D8 = int(os.environ.get("D8", "1"))
SPLIT = os.environ.get("EXTRACT_SPLIT", "validation")
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "0"))
DEVICE = "cuda"
SZ = 512
CROP = 256
C5 = [(0, 0), (0, 256), (256, 0), (256, 256), (128, 128)]
LM = np.array([7.919919967651367, -0.9824771285057068, -0.6211431622505188], np.float32)
LS = np.array([0.18971863389015198, 0.18813571333885193, 0.2748945653438568], np.float32)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
meta = np.load(ROOT / "cache_native_meta_512.npz", allow_pickle=True)
images = np.memmap(ROOT / "img_native_512.dat", dtype=np.uint8, mode="r", shape=(int(meta["N"]), SZ, SZ))
oof = np.load(RUN_DIR / "best_oof.npz", allow_pickle=True)
indices = oof["indices"].astype(np.int64)
targets = oof["target"].astype(np.float32)
if SPLIT not in ("validation", "holdout"):
    raise ValueError(SPLIT)
if SPLIT == "holdout":
    splits = np.load(RUN_DIR / "split_indices.npz")
    indices = splits["holdout"]
    assert len(indices) and not np.intersect1d(indices, splits["train"]).size
    assert not np.intersect1d(indices, splits["validation"]).size
    targets = meta["Y"][indices].astype(np.float32)
if MAX_IMAGES:
    indices, targets = indices[:MAX_IMAGES], targets[:MAX_IMAGES]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def normalize(image: np.ndarray) -> np.ndarray:
    image = clahe.apply(np.ascontiguousarray(image))
    lo, hi = np.percentile(image, (1, 99))
    return np.clip((image.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)


def dihedral(image: np.ndarray, variant: int) -> np.ndarray:
    out = np.rot90(image, variant % 4)
    if variant >= 4:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def quality(raw: np.ndarray) -> np.ndarray:
    x = raw.astype(np.float32) / 255.0
    lap = cv2.Laplacian(x, cv2.CV_32F)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    return np.array([x.std(), lap.var(), np.hypot(gx, gy).mean(), np.mean((x < .02) | (x > .98))], np.float32)


class ScalarNet(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(256)

        def head() -> nn.Module:
            return nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(.2), nn.Linear(128, 1))

        self.cd, self.cv, self.hex = head(), head(), head()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(self.encoder(inputs)).flatten(1)
        pooled = self.norm(pooled)
        return torch.cat((self.cd(pooled), self.cv(pooled), self.hex(pooled)), dim=1)


base = models.CellposeModel(gpu=True, pretrained_model=os.environ.get("CPSAM_MODEL", "cpsam_models/cpsam_v2"))
net = ScalarNet(base.net.encoder.float()).to(DEVICE)
payload = torch.load(RUN_DIR / "best.pt", map_location=DEVICE, weights_only=True)
config = payload["config"]
LM = np.asarray(config.get("lm", LM), np.float32)
LS = np.asarray(config.get("ls", LS), np.float32)
if SPLIT == "holdout":
    assert config.get("select_mode") == "last", "nested validation requires fixed-epoch checkpoint"
net.load_state_dict(payload["model"])
net.eval()

all_z, all_q = [], []
variants = range(8) if D8 else range(1)
with torch.inference_mode():
    for start in range(0, len(indices), BS):
        batch_rows = indices[start : start + BS]
        views, qualities = [], []
        for index in batch_rows:
            image_views, image_quality = [], []
            for row, col in C5:
                raw = np.asarray(images[index, row : row + CROP, col : col + CROP])
                norm = normalize(raw)
                image_views.append([
                    np.stack((dihedral(norm, variant),) * 3, axis=0) for variant in variants
                ])
                image_quality.append(quality(raw))
            views.append(image_views)
            qualities.append(image_quality)
        array = np.asarray(views, np.float32)
        b, tiles, augmentations = array.shape[:3]
        inputs = torch.from_numpy(array.reshape(-1, 3, CROP, CROP)).to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = net(inputs).float().reshape(b, tiles, augmentations, 3).mean(dim=2)
        all_z.append(z.cpu().numpy())
        all_q.append(np.asarray(qualities, np.float32))
        print(f"{min(start + BS, len(indices))}/{len(indices)}", flush=True)

z_tile = np.concatenate(all_z)
quality_tile = np.concatenate(all_q)
pred_mean = np.exp(z_tile.mean(axis=1) * LS + LM)
relative_delta = 0.0
if SPLIT == "validation":
    reference = oof["pred"][:len(indices)].astype(np.float32)
    relative_delta = np.max(np.abs(pred_mean - reference) / np.maximum(reference, 1e-6))
metric = np.array([
    np.mean(np.abs(pred_mean[targets[:, m] > 0, m] - targets[targets[:, m] > 0, m])
            / targets[targets[:, m] > 0, m]) * 100 for m in range(3)
])
assert np.isfinite(z_tile).all() and np.isfinite(metric).all()
output_name = "crop_tiles.npz" if SPLIT == "validation" else "crop_holdout.npz"
if MAX_IMAGES:
    output_name = "smoke_" + output_name
np.savez_compressed(
    RUN_DIR / output_name,
    indices=indices,
    ids=meta["ids"][indices],
    target=targets,
    z_tile=z_tile,
    quality_tile=quality_tile,
    pred_mean=pred_mean,
    metric_mean=metric,
    reference_relative_delta=relative_delta,
    lm=LM,
    ls=LS,
)
print(f"saved {RUN_DIR / output_name} metric={metric.tolist()} reference_delta={relative_delta:.6g}")
