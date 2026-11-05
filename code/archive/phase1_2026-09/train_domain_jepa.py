#!/usr/bin/env python3
"""Domain-adapt the Cellpose-SAM encoder with a masked latent prediction objective.

The online encoder receives a block-masked view and predicts the EMA target
encoder's features at the hidden positions.  Sampling stays anchored on all
9,000 CLEAR-EC images while adding provenance-aware public corneal-endothelium
data without allowing a large adjacent-modality collection to dominate.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from cellpose import models
from torch.nn.parallel import DistributedDataParallel as DDP


ROOT = Path(os.environ.get("ROOT", "datasets/clear_ec_official"))
EXTERNAL_MANIFEST = Path(os.environ.get("EXTERNAL_MANIFEST", "datasets/external/ssl_manifest.jsonl"))
EXTERNAL_SOURCES = {
    value.strip()
    for value in os.environ.get("EXTERNAL_SOURCES", "").split(",")
    if value.strip()
}
EXTERNAL_SOURCE_WEIGHTS = json.loads(os.environ.get("EXTERNAL_SOURCE_WEIGHTS", "{}"))
OUT = Path(os.environ.get("OUT", "experiments/domain_jepa"))
INITIAL_ENCODER = os.environ.get("INITIAL_ENCODER", "")
RESUME = os.environ.get("RESUME", "")
SEED = int(os.environ.get("SEED", "2026"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
BS = int(os.environ.get("BS", "24"))
STEPS_PER_EPOCH = int(os.environ.get("STEPS_PER_EPOCH", "0"))
LR = float(os.environ.get("LR", "2e-5"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.05"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "100"))
EXTERNAL_FRACTION = float(os.environ.get("EXTERNAL_FRACTION", "0.30"))
MASK_RATIO = float(os.environ.get("MASK_RATIO", "0.55"))
EMA_START = float(os.environ.get("EMA_START", "0.996"))
EMA_END = float(os.environ.get("EMA_END", "0.9999"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "5"))
SAVE_FULL_STATE = int(os.environ.get("SAVE_FULL_STATE", "1"))
CROP = 256
FEATURE_SIZE = 32

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
if WORLD_SIZE > 1:
    dist.init_process_group("nccl")
torch.cuda.set_device(LOCAL_RANK)
DEVICE = torch.device("cuda", LOCAL_RANK)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED + RANK * 100003)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

meta = np.load(ROOT / "cache_native_meta_512.npz", allow_pickle=True)
N = int(meta["N"])
CLEAR_IMAGES = np.memmap(ROOT / "img_native_512.dat", dtype=np.uint8, mode="r", shape=(N, 512, 512))

external_entries: list[dict] = []
if EXTERNAL_MANIFEST.exists():
    with EXTERNAL_MANIFEST.open(encoding="utf-8") as stream:
        external_entries = [json.loads(line) for line in stream if line.strip()]
external_by_source: dict[str, list[dict]] = defaultdict(list)
for entry in external_entries:
    if not EXTERNAL_SOURCES or entry["source"] in EXTERNAL_SOURCES:
        external_by_source[entry["source"]].append(entry)

# Source-level weights deliberately do not depend on file count.  This keeps
# the 100 GiB adjacent-modality collection useful but subordinate to specular
# microscopy, and quarantines the dataset with undocumented provenance.
default_source_weights = {
    "srep_topcon_sp3000": 0.55,
    "rodrep_confocal_endothelium": 0.15,
    "github_cornea_cells": 0.05,
    "zenodo_fecd_flatmount": 0.25,
}
source_names = sorted(external_by_source)
source_weights = np.array(
    [EXTERNAL_SOURCE_WEIGHTS.get(name, default_source_weights.get(name, 0.05)) for name in source_names],
    np.float64,
)
if len(source_weights):
    if np.any(source_weights < 0) or source_weights.sum() <= 0:
        raise ValueError(f"invalid external source weights: {dict(zip(source_names, source_weights))}")
    source_weights /= source_weights.sum()
if EXTERNAL_FRACTION > 0 and not source_names:
    raise ValueError(
        f"EXTERNAL_FRACTION={EXTERNAL_FRACTION} but no manifest entries match "
        f"EXTERNAL_SOURCES={sorted(EXTERNAL_SOURCES)}"
    )

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def decode(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"cannot decode {path}")
    image = np.squeeze(image)
    if image.ndim == 3:
        if image.shape[-1] >= 3:
            image = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)
        else:
            image = image[..., 0]
    if image.ndim != 2:
        raise RuntimeError(f"unsupported image shape {image.shape}: {path}")
    if image.dtype != np.uint8:
        array = image.astype(np.float32)
        lo, hi = np.percentile(array[np.isfinite(array)], (1, 99))
        image = np.clip((array - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
    return image


def geometry(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    minimum = min(height, width)
    if minimum < CROP:
        scale = CROP / minimum
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_CUBIC)
        height, width = image.shape
    # Mostly preserve native scale; occasional wider crops teach robustness to
    # device magnification and the pre-generated flat-mount scales.
    crop_size = min(min(height, width), int(round(CROP * np.random.uniform(1.0, 1.35))))
    row = np.random.randint(0, height - crop_size + 1)
    col = np.random.randint(0, width - crop_size + 1)
    output = image[row : row + crop_size, col : col + crop_size]
    if crop_size != CROP:
        output = cv2.resize(output, (CROP, CROP), interpolation=cv2.INTER_AREA)
    if np.random.rand() < 0.5:
        output = output[:, ::-1]
    if np.random.rand() < 0.5:
        output = output[::-1]
    output = np.rot90(output, np.random.randint(4))
    return np.ascontiguousarray(output)


def normalize(image: np.ndarray) -> np.ndarray:
    image = clahe.apply(np.ascontiguousarray(image.astype(np.uint8))).astype(np.float32)
    lo, hi = np.percentile(image, (1, 99))
    return np.clip((image - lo) / (hi - lo + 1e-6), 0, 1)


def photometric(image: np.ndarray, strong: bool) -> np.ndarray:
    output = image.copy()
    gamma = np.random.uniform(0.72, 1.38) if strong else np.random.uniform(0.9, 1.1)
    output = np.power(np.clip(output, 0, 1), gamma)
    gain = np.random.uniform(0.78, 1.22) if strong else np.random.uniform(0.94, 1.06)
    bias = np.random.uniform(-0.12, 0.12) if strong else np.random.uniform(-0.03, 0.03)
    output = output * gain + bias
    if strong and np.random.rand() < 0.35:
        sigma = float(np.random.uniform(0.2, 1.2))
        output = cv2.GaussianBlur(output, (0, 0), sigma)
    noise = np.random.uniform(0.0, 0.045 if strong else 0.015)
    if noise:
        output += np.random.normal(0, noise, output.shape).astype(np.float32)
    return np.clip(output, 0, 1).astype(np.float32)


def feature_mask() -> np.ndarray:
    mask = np.zeros((FEATURE_SIZE, FEATURE_SIZE), np.float32)
    target = int(round(MASK_RATIO * mask.size))
    attempts = 0
    while int(mask.sum()) < target and attempts < 100:
        height = np.random.randint(4, 11)
        width = np.random.randint(4, 11)
        row = np.random.randint(0, FEATURE_SIZE - height + 1)
        col = np.random.randint(0, FEATURE_SIZE - width + 1)
        mask[row : row + height, col : col + width] = 1
        attempts += 1
    return mask


def sample_external() -> tuple[np.ndarray, str]:
    source_index = int(np.random.choice(len(source_names), p=source_weights))
    source = source_names[source_index]
    entry = random.choice(external_by_source[source])
    return decode(entry["path"]), source


def sample_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    contexts, targets, masks = [], [], []
    counts: dict[str, int] = defaultdict(int)
    for _ in range(BS):
        if source_names and np.random.rand() < EXTERNAL_FRACTION:
            raw, source = sample_external()
        else:
            raw, source = CLEAR_IMAGES[np.random.randint(N)], "clear_ec_official"
        counts[source] += 1
        base = normalize(geometry(raw))
        context = photometric(base, strong=True)
        target = photometric(base, strong=False)
        mask = feature_mask()
        pixel_mask = cv2.resize(mask, (CROP, CROP), interpolation=cv2.INTER_NEAREST)
        # A neutral value is less revealing than a hard zero after normalization.
        context = context * (1 - pixel_mask) + 0.5 * pixel_mask
        contexts.append(np.stack((context, context, context)))
        targets.append(np.stack((target, target, target)))
        masks.append(mask)
    return (
        torch.from_numpy(np.stack(contexts)).to(DEVICE, non_blocking=True),
        torch.from_numpy(np.stack(targets)).to(DEVICE, non_blocking=True),
        torch.from_numpy(np.stack(masks)[:, None]).to(DEVICE, non_blocking=True),
        counts,
    )


class Predictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(256, 512, 1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.Conv2d(512, 256, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class OnlineModel(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = Predictor()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.encoder(inputs))


base = models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
online = OnlineModel(base.net.encoder.float()).to(DEVICE)
if INITIAL_ENCODER:
    payload = torch.load(INITIAL_ENCODER, map_location="cpu", weights_only=True)
    state = payload.get("teacher_encoder", payload.get("encoder", payload))
    message = online.encoder.load_state_dict(state, strict=False)
    if RANK == 0:
        print(f"loaded INITIAL_ENCODER={INITIAL_ENCODER}: {message}", flush=True)
teacher = copy.deepcopy(online.encoder).to(DEVICE).eval()
for parameter in teacher.parameters():
    parameter.requires_grad_(False)

import torch.utils.checkpoint as checkpoint

for block in online.encoder.blocks:
    original_forward = block.forward
    block.forward = (lambda function: lambda x: checkpoint.checkpoint(
        function, x, use_reentrant=False
    ))(original_forward)

if WORLD_SIZE > 1:
    online = DDP(online, device_ids=[LOCAL_RANK], broadcast_buffers=False)
raw_online = online.module if isinstance(online, DDP) else online
optimizer = torch.optim.AdamW(online.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
scaler = None  # bfloat16 does not require loss scaling

global_batch = BS * WORLD_SIZE
if not STEPS_PER_EPOCH:
    STEPS_PER_EPOCH = math.ceil(N / global_batch)
total_steps = EPOCHS * STEPS_PER_EPOCH
start_epoch = 0
global_step = 0
if RESUME:
    payload = torch.load(RESUME, map_location="cpu", weights_only=True)
    raw_online.encoder.load_state_dict(payload["online_encoder"])
    raw_online.predictor.load_state_dict(payload["predictor"])
    teacher.load_state_dict(payload["teacher_encoder"])
    optimizer.load_state_dict(payload["optimizer"])
    start_epoch = int(payload["epoch"]) + 1
    global_step = int(payload["global_step"])

if RANK == 0:
    OUT.mkdir(parents=True, exist_ok=True)
    print(
        f"domain JEPA world={WORLD_SIZE} global_batch={global_batch} epochs={EPOCHS} "
        f"steps/epoch={STEPS_PER_EPOCH} external={len(external_entries)} ({source_names})",
        flush=True,
    )


def schedule(step: int) -> tuple[float, float]:
    if step < WARMUP_STEPS:
        lr = LR * (step + 1) / max(WARMUP_STEPS, 1)
    else:
        progress = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
        lr = LR * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    progress = min(step / max(total_steps - 1, 1), 1.0)
    ema = EMA_END - (EMA_END - EMA_START) * 0.5 * (1 + math.cos(math.pi * progress))
    return lr, ema


def save(epoch: int, filename: str) -> None:
    if RANK != 0:
        return
    if SAVE_FULL_STATE:
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "online_encoder": raw_online.encoder.state_dict(),
                "teacher_encoder": teacher.state_dict(),
                "predictor": raw_online.predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": {
                    "world_size": WORLD_SIZE,
                    "global_batch": global_batch,
                    "external_fraction": EXTERNAL_FRACTION,
                    "external_manifest": str(EXTERNAL_MANIFEST),
                    "external_sources": source_names,
                    "external_source_weights": source_weights.tolist(),
                    "mask_ratio": MASK_RATIO,
                },
            },
            OUT / filename,
        )
    torch.save(teacher.state_dict(), OUT / "encoder_latest.pt")


for epoch in range(start_epoch, EPOCHS):
    online.train()
    started = time.time()
    loss_sum = 0.0
    source_counts: dict[str, int] = defaultdict(int)
    for step in range(STEPS_PER_EPOCH):
        contexts, targets, mask, counts = sample_batch()
        for name, count in counts.items():
            source_counts[name] += count
        lr, ema = schedule(global_step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            target_features = teacher(targets)
            target_features = F.layer_norm(target_features.float().permute(0, 2, 3, 1), (256,)).permute(0, 3, 1, 2)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = online(contexts)
        predictions = F.layer_norm(predictions.float().permute(0, 2, 3, 1), (256,)).permute(0, 3, 1, 2)
        loss_map = F.smooth_l1_loss(predictions, target_features, reduction="none").mean(dim=1, keepdim=True)
        loss = (loss_map * mask).sum() / (mask.sum() + 1e-6)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            for target_parameter, online_parameter in zip(teacher.parameters(), raw_online.encoder.parameters()):
                target_parameter.mul_(ema).add_(online_parameter.detach(), alpha=1 - ema)
        loss_sum += float(loss.item())
        global_step += 1
    reduced = torch.tensor([loss_sum], device=DEVICE)
    if WORLD_SIZE > 1:
        dist.all_reduce(reduced)
    if RANK == 0:
        print(
            f"epoch={epoch:03d} loss={reduced.item() / (STEPS_PER_EPOCH * WORLD_SIZE):.6f} "
            f"lr={lr:.2e} ema={ema:.6f} seconds={time.time() - started:.1f} sources={dict(source_counts)}",
            flush=True,
        )
    if (epoch + 1) % SAVE_EVERY == 0 or epoch + 1 == EPOCHS:
        save(epoch, f"checkpoint_epoch{epoch + 1:03d}.pt")

save(EPOCHS - 1, "checkpoint_final.pt")
if WORLD_SIZE > 1:
    dist.barrier()
    dist.destroy_process_group()
