"""Isolated snapshot of train_oof_h200 with cosine, metric loss, prediction bags and SAM.

One process owns one GPU. Launch several processes with different CUDA_VISIBLE_DEVICES,
FOLD, SEED, CROP, and LOSS_MODE values. Folds are grouped by slide_id to avoid
same-slide leakage. Validation predictions and targets are written for honest OOF
model/calibration selection.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from method_training_core import cosine_multiplier, regression_loss, SAMPerturbation

import cv2
import numpy as np
import torch
import torch.nn as nn
from cellpose import models


ROOT = Path(os.environ.get("ROOT", "datasets/clear_ec_official"))
OUT_ROOT = Path(os.environ.get("OUT_ROOT", "experiments/oof_h200"))
TAG = os.environ.get("TAG", "s2")
FOLD = int(os.environ.get("FOLD", "0"))
EXCLUDE_FOLD = int(os.environ.get("EXCLUDE_FOLD", "-1"))
SELECT_MODE = os.environ.get("SELECT_MODE", "best")
NORM_MODE = os.environ.get("NORM_MODE", "fixed")
NFOLDS = int(os.environ.get("NFOLDS", "5"))
FOLD_SEED = int(os.environ.get("FOLD_SEED", "2026"))
SEED = int(os.environ.get("SEED", "42"))
CROP = int(os.environ.get("CROP", "256"))
K_CROPS = int(os.environ.get("K_CROPS", "1"))
CROP_MODE = os.environ.get("CROP_MODE", "random")  # random | cover4 | cover5
BAG_POOL = os.environ.get("BAG_POOL", "mean")  # mean | meanstd | attn
POOL = os.environ.get("POOL", "mean")
LOSS_MODE = os.environ.get("LOSS_MODE", "multitask")  # multitask | cd
YSRC = os.environ.get("YSRC", "slidesmooth")  # raw | slidesmooth
EPOCHS = int(os.environ.get("EPOCHS", "25"))
BS = int(os.environ.get("BS", "48"))
EVAL_IMAGE_BS = int(os.environ.get("EVAL_IMAGE_BS", "4"))
EVAL_D8 = int(os.environ.get("EVAL_D8", "1"))
FINAL_D8 = int(os.environ.get("FINAL_D8", "1"))
FEATURE_AGG = int(os.environ.get("FEATURE_AGG", "1" if K_CROPS > 1 else "0"))
MAX_TRAIN_STEPS = int(os.environ.get("MAX_TRAIN_STEPS", "0"))
MAX_VAL_IMAGES = int(os.environ.get("MAX_VAL_IMAGES", "0"))
SAVE_CHECKPOINT = int(os.environ.get("SAVE_CHECKPOINT", "1"))
ENC_LR = float(os.environ.get("ENC_LR", "1e-5"))
HEAD_LR = float(os.environ.get("HEAD_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
OBJECTIVE = os.environ.get("OBJECTIVE", "log_l1")
SAM_RHO = float(os.environ.get("SAM_RHO", "0"))
DEVICE = "cuda"
SZ = 512
REGION_POLICY = os.environ.get("REGION_POLICY", "central")
if REGION_POLICY not in ("central", "full_uniform", "full_quality", "anchor_quality"):
    raise ValueError("Unknown region policy")

if POOL != "mean" or BAG_POOL != "mean" or FEATURE_AGG or LOSS_MODE != "multitask":
    raise ValueError("method screen requires mean single-view heads and prediction-level inference")
if OBJECTIVE not in ("log_l1", "mixed_mape") or SAM_RHO < 0:
    raise ValueError("invalid method configuration")
if encoder_override := os.environ.get("ENCODER_INIT", ""):
    permitted = Path("experiments/region_search_20260905/pretrain/encoder_final.pt")
    if Path(encoder_override) != permitted:
        raise ValueError("Only the isolated external-only encoder is allowed; never all-data weights")
    expected_hash = os.environ.get("ENCODER_SHA256", "")
    if not expected_hash or hashlib.sha256(permitted.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("External encoder provenance hash mismatch")

if CROP != 256:
    raise ValueError("cpsam_v2 positional embeddings require CROP=256; use K_CROPS for wider coverage")
if K_CROPS < 1:
    raise ValueError(f"K_CROPS must be positive, got {K_CROPS}")
if CROP_MODE not in ("random", "cover4", "cover5"):
    raise ValueError(f"unknown CROP_MODE={CROP_MODE}")
if BAG_POOL not in ("mean", "meanstd", "attn"):
    raise ValueError(f"unknown BAG_POOL={BAG_POOL}")
if CROP_MODE == "cover4" and K_CROPS != 4:
    raise ValueError("CROP_MODE=cover4 requires K_CROPS=4")
if CROP_MODE == "cover5" and K_CROPS != 5:
    raise ValueError("CROP_MODE=cover5 requires K_CROPS=5")
if BAG_POOL != "mean" and K_CROPS == 1:
    raise ValueError(f"BAG_POOL={BAG_POOL} requires K_CROPS>1")
if not 0 <= FOLD < NFOLDS:
    raise ValueError(f"FOLD={FOLD} outside [0,{NFOLDS})")
if LOSS_MODE not in ("multitask", "cd"):
    raise ValueError(f"unknown LOSS_MODE={LOSS_MODE}")
if EXCLUDE_FOLD >= 0 and (EXCLUDE_FOLD == FOLD or EXCLUDE_FOLD >= NFOLDS):
    raise ValueError("EXCLUDE_FOLD must be a different valid fold")
if SELECT_MODE not in ("best", "last") or NORM_MODE not in ("fixed", "train"):
    raise ValueError("invalid selection or normalization mode")
if BAG_POOL != "mean" and not FEATURE_AGG:
    raise ValueError("bag pooling requires FEATURE_AGG=1")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

meta = np.load(ROOT / f"cache_native_meta_{SZ}.npz", allow_pickle=True)
Y = meta["Y"].astype(np.float32)
IDS = [str(x) for x in meta["ids"]]
N = int(meta["N"])
IMGS = np.memmap(ROOT / f"img_native_{SZ}.dat", dtype=np.uint8, mode="r", shape=(N, SZ, SZ))
region_views = None
if REGION_POLICY != "central":
    from native_region_views import NativeRegionViews
    region_views = NativeRegionViews(IMGS, IDS, REGION_POLICY)


def clean_id(value: str) -> str:
    value = str(value)
    for suffix in (".bmp", ".mha", ".tif", ".tiff"):
        if value.lower().endswith(suffix):
            return value[: -len(suffix)]
    return value


with open(ROOT / "final_train_ids.csv", newline="") as f:
    rows = list(csv.DictReader(f))
id_to_slide = {clean_id(r["ID"]): str(r["slide_id"]) for r in rows}
slide_for_row = [id_to_slide.get(clean_id(image_id), f"missing:{clean_id(image_id)}") for image_id in IDS]


def make_group_folds() -> np.ndarray:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, slide in enumerate(slide_for_row):
        groups[slide].append(i)
    rng = np.random.default_rng(FOLD_SEED)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: len(kv[1]), reverse=True)
    fold_sizes = np.zeros(NFOLDS, dtype=np.int64)
    assignment = np.empty(N, dtype=np.int16)
    for _, indices in items:
        fold = int(np.argmin(fold_sizes))
        assignment[indices] = fold
        fold_sizes[fold] += len(indices)
    print(f"fold sizes: {fold_sizes.tolist()}", flush=True)
    return assignment


fold_of = make_group_folds()
val_idx = np.where(fold_of == FOLD)[0]
train_idx = np.where(fold_of != FOLD)[0]
holdout_idx = np.where(fold_of == EXCLUDE_FOLD)[0] if EXCLUDE_FOLD >= 0 else np.array([], dtype=np.int64)
train_idx = train_idx[fold_of[train_idx] != EXCLUDE_FOLD]
train_idx = train_idx[(Y[train_idx] > 0).all(axis=1)]
if MAX_VAL_IMAGES:
    val_idx = val_idx[:MAX_VAL_IMAGES]

Y_TRAIN = Y.copy()
if YSRC == "slidesmooth":
    by_slide: dict[str, list[int]] = defaultdict(list)
    train_set = set(train_idx.tolist())
    for i, slide in enumerate(slide_for_row):
        if i in train_set:
            by_slide[slide].append(i)
    for indices in by_slide.values():
        if len(indices) >= 2:
            Y_TRAIN[indices] = Y[indices].mean(axis=0)
elif YSRC != "raw":
    raise ValueError(f"unknown YSRC={YSRC}")

# Fixed normalization matches submission inference and makes folds directly compatible.
LM = np.array([7.919919967651367, -0.9824771285057068, -0.6211431622505188], np.float32)
LS = np.array([0.18971863389015198, 0.18813571333885193, 0.2748945653438568], np.float32)
if NORM_MODE == "train":
    train_logs = np.log(Y_TRAIN[train_idx])
    LM, LS = train_logs.mean(axis=0), np.maximum(train_logs.std(axis=0), 1e-6)
Z = ((np.log(np.clip(Y_TRAIN, 1e-6, None)) - LM) / LS).astype(np.float32)
Z_T = torch.tensor(Z, device=DEVICE)
LM_T = torch.tensor(LM, device=DEVICE)
LS_T = torch.tensor(LS, device=DEVICE)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def normalize(image: np.ndarray) -> np.ndarray:
    image = clahe.apply(np.ascontiguousarray(image))
    lo, hi = np.percentile(image, (1, 99))
    return np.clip((image.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)


def random_view(index: int) -> np.ndarray:
    limit = SZ - CROP
    row = np.random.randint(0, limit + 1) if limit else 0
    col = np.random.randint(0, limit + 1) if limit else 0
    image = normalize(IMGS[index, row : row + CROP, col : col + CROP])
    if np.random.rand() < 0.5:
        image = image[:, ::-1]
    if np.random.rand() < 0.5:
        image = image[::-1, :]
    image = np.ascontiguousarray(np.rot90(image, np.random.randint(4)))
    return np.stack((image, image, image), axis=0)


def covered_views(index: int) -> np.ndarray:
    """A deterministic full-coverage bag with one shared random dihedral transform.

    This avoids assigning the image-level target independently to a random local crop.
    The four corner tiles exactly partition the cached 512x512 field; cover5 also adds
    the overlapping centre tile used by submission inference.
    """
    origins = [(0, 0), (0, 256), (256, 0), (256, 256)]
    if CROP_MODE == "cover5":
        origins.append((128, 128))
    variant = np.random.randint(8)
    views = []
    for row, col in origins:
        image = normalize(IMGS[index, row : row + CROP, col : col + CROP])
        image = dihedral(image, variant)
        views.append(np.stack((image, image, image), axis=0))
    return np.stack(views)


def train_batch(indices: np.ndarray) -> torch.Tensor:
    if region_views is not None:
        return torch.from_numpy(np.stack([region_views.train(int(i),K_CROPS) for i in indices])).to(DEVICE)
    if CROP_MODE != "random":
        array = np.stack([covered_views(int(i)) for i in indices])
    elif K_CROPS == 1:
        array = np.stack([random_view(int(i)) for i in indices])
    else:
        array = np.stack([[random_view(int(i)) for _ in range(K_CROPS)] for i in indices])
    return torch.from_numpy(array).to(DEVICE, non_blocking=True)


def eval_origins() -> list[tuple[int, int]]:
    if CROP == SZ:
        return [(0, 0)]
    limit = SZ - CROP
    mid = limit // 2
    corners = [(0, 0), (0, limit), (limit, 0), (limit, limit)]
    return corners if CROP_MODE == "cover4" else corners + [(mid, mid)]


def dihedral(image: np.ndarray, variant: int) -> np.ndarray:
    out = np.rot90(image, variant % 4)
    if variant >= 4:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def eval_views(index: int, use_d8: int) -> np.ndarray:
    if region_views is not None:
        return region_views.evaluate(index,use_d8)
    variants = range(8) if use_d8 else range(1)
    views = []
    for row, col in eval_origins():
        image = normalize(IMGS[index, row : row + CROP, col : col + CROP])
        for variant in variants:
            view = dihedral(image, variant)
            views.append(np.stack((view, view, view), axis=0))
    return np.stack(views)


class ScalarNet(nn.Module):
    def __init__(self, encoder: nn.Module, pool: str):
        super().__init__()
        self.encoder = encoder
        self.pool_name = pool
        self.pool = nn.AdaptiveAvgPool2d(1)
        bag_dim = 512 if BAG_POOL == "meanstd" else 256
        self.norm = nn.LayerNorm(bag_dim)
        if pool == "attn":
            self.query = nn.Parameter(torch.zeros(1, 1, 256))
            self.attn = nn.MultiheadAttention(256, 4, batch_first=True)
        elif pool != "mean":
            raise ValueError(f"unknown POOL={pool}")
        if BAG_POOL == "attn":
            self.bag_query = nn.Parameter(torch.zeros(1, 1, 256))
            self.bag_attn = nn.MultiheadAttention(256, 4, batch_first=True)

        def head() -> nn.Module:
            module = nn.Sequential(nn.Linear(bag_dim, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1))
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)
            return module

        self.cd = head()
        self.cv = head()
        self.hex = head()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        multiview = inputs.ndim == 5
        if multiview:
            batch, views = inputs.shape[:2]
            inputs = inputs.reshape(batch * views, *inputs.shape[2:])
        features = self.encoder(inputs)
        if self.pool_name == "attn":
            seq = features.flatten(2).transpose(1, 2)
            pooled = self.attn(self.query.expand(len(inputs), -1, -1), seq, seq)[0].squeeze(1)
        else:
            pooled = self.pool(features).flatten(1)
        # Apply the exact same per-crop head as at inference. The image-level
        # target supervises the average prediction, NOT the average feature.
        pooled = self.norm(pooled)
        output = torch.cat((self.cd(pooled), self.cv(pooled), self.hex(pooled)), dim=1)
        return output.reshape(batch, views, 3).mean(dim=1) if multiview else output



pretrained = os.environ.get("CPSAM_MODEL", "cpsam_models/cpsam_v2")
base = models.CellposeModel(gpu=True, pretrained_model=pretrained)
encoder_init = os.environ.get("ENCODER_INIT", "")
if encoder_init:
    payload = torch.load(encoder_init, map_location="cpu", weights_only=True)
    if not payload.get("config",{}).get("external_only"):
        raise ValueError("External initialization may not have seen CLEAR-EC images")
    state = payload.get("teacher_encoder", payload.get("encoder", payload))
    message = base.net.encoder.load_state_dict(state, strict=True)
    print(f"loaded ENCODER_INIT={encoder_init}: {message}", flush=True)
net = ScalarNet(base.net.encoder.float(), POOL).to(DEVICE)

import torch.utils.checkpoint as checkpoint

for block in net.encoder.blocks:
    block._clear_ec_forward = block.forward
    block.forward = (lambda module: lambda x: checkpoint.checkpoint(
        module._clear_ec_forward, x, use_reentrant=False
    ))(block)

head_params = list(net.cd.parameters()) + list(net.cv.parameters()) + list(net.hex.parameters()) + list(net.norm.parameters())
if POOL == "attn":
    head_params += [net.query] + list(net.attn.parameters())
if BAG_POOL == "attn":
    head_params += [net.bag_query] + list(net.bag_attn.parameters())
optimizer = torch.optim.AdamW(
    [
        {"params": net.encoder.parameters(), "lr": ENC_LR},
        {"params": head_params, "lr": HEAD_LR},
    ],
    weight_decay=WEIGHT_DECAY,
)
optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
missing = [name for name, p in net.named_parameters() if p.requires_grad and id(p) not in optimized]
assert not missing, f"trainable parameters missing from optimizer: {missing}"
print("optimizer coverage: all trainable parameters included (including norm)", flush=True)
steps_per_epoch = len(train_idx) // BS
if MAX_TRAIN_STEPS:
    steps_per_epoch = min(steps_per_epoch, MAX_TRAIN_STEPS)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda step: cosine_multiplier(step, EPOCHS * steps_per_epoch))

bag_suffix = "" if CROP_MODE == "random" and BAG_POOL == "mean" else f"_{CROP_MODE}_{BAG_POOL}"
run_name = f"{TAG}_f{FOLD}_seed{SEED}_c{CROP}k{K_CROPS}_{POOL}_{LOSS_MODE}_{YSRC}{bag_suffix}"
run_dir = OUT_ROOT / run_name
run_dir.mkdir(parents=True, exist_ok=True)
if (run_dir / "result.json").exists():
    raise FileExistsError(f"completed run must not be overwritten: {run_dir}")
checkpoint_path = run_dir / "best.pt"
slide_sets = [set(np.asarray(slide_for_row)[idx]) for idx in (train_idx, val_idx, holdout_idx)]
assert not (slide_sets[0] & slide_sets[1] or slide_sets[0] & slide_sets[2] or slide_sets[1] & slide_sets[2])
np.savez_compressed(run_dir / "split_indices.npz", train=train_idx, validation=val_idx, holdout=holdout_idx)


@torch.no_grad()
def evaluate(indices: np.ndarray, use_d8: int) -> tuple[np.ndarray, np.ndarray]:
    net.eval()
    predictions = []
    for start in range(0, len(indices), EVAL_IMAGE_BS):
        batch_indices = indices[start : start + EVAL_IMAGE_BS]
        arrays = [eval_views(int(i), use_d8) for i in batch_indices]
        nviews = arrays[0].shape[0]
        if FEATURE_AGG:
            inputs = torch.from_numpy(np.stack(arrays)).to(DEVICE, non_blocking=True)
            augmentations = 8 if use_d8 else 1
            # Aggregate spatial tiles within each orientation, then average predictions.
            # Rotated duplicates must not inflate between-tile statistics.
            inputs = inputs.reshape(len(batch_indices), -1, augmentations, 3, CROP, CROP)
            inputs = inputs.permute(0, 2, 1, 3, 4, 5).reshape(
                len(batch_indices) * augmentations, -1, 3, CROP, CROP
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                encoded = net(inputs).float().reshape(len(batch_indices), augmentations, 3).mean(dim=1)
        else:
            inputs = torch.from_numpy(np.concatenate(arrays)).to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                encoded = net(inputs).float().reshape(len(batch_indices), nviews, 3).mean(dim=1)
        decoded = torch.exp(encoded * LS_T + LM_T)
        predictions.append(decoded.cpu().numpy())
    pred = np.concatenate(predictions)
    target = Y[indices]
    metric_values = []
    for metric in range(3):
        valid = target[:, metric] > 0
        metric_values.append(
            (np.abs(pred[valid, metric] - target[valid, metric]) / target[valid, metric]).mean() * 100
        )
    metrics = np.asarray(metric_values, dtype=np.float32)
    return pred, metrics


print(
    f"[{run_name}] train={len(train_idx)} val={len(val_idx)} epochs={EPOCHS} "
    f"bs={BS} k_crops={K_CROPS} feature_agg={FEATURE_AGG} "
    f"crop_mode={CROP_MODE} bag_pool={BAG_POOL} "
    f"eval_views={len(eval_origins()) * (8 if EVAL_D8 else 1)}",
    flush=True,
)
best_score = float("inf")
best_epoch = -1
for epoch in range(EPOCHS):
    net.train()
    rng = np.random.default_rng(SEED * 1000 + epoch)
    shuffled = rng.permutation(train_idx)
    total_loss = 0.0
    started = time.time()
    for step in range(steps_per_epoch):
        indices = shuffled[step * BS : (step + 1) * BS]
        inputs = train_batch(indices)
        cpu_rng = torch.get_rng_state() if SAM_RHO else None
        cuda_rng = torch.cuda.get_rng_state() if SAM_RHO else None
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = regression_loss(net(inputs), Z_T[indices], LS_T, OBJECTIVE)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite first-pass loss")
        loss.backward()
        if SAM_RHO:
            with SAMPerturbation(net.parameters(), SAM_RHO):
                optimizer.zero_grad(set_to_none=True)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    perturbed_loss = regression_loss(net(inputs), Z_T[indices], LS_T, OBJECTIVE)
                if not torch.isfinite(perturbed_loss):
                    raise FloatingPointError("nonfinite SAM loss")
                perturbed_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        total_loss += loss.detach().item()

    pred, metrics = evaluate(val_idx, EVAL_D8)
    selection = float(metrics[0] if LOSS_MODE == "cd" else metrics.mean())
    print(
        f"epoch={epoch + 1:02d} loss={total_loss / max(steps_per_epoch, 1):.4f} "
        f"val=[{metrics[0]:.3f},{metrics[1]:.3f},{metrics[2]:.3f}] "
        f"overall={metrics.mean():.3f} seconds={time.time() - started:.0f}",
        flush=True,
    )
    if (SELECT_MODE == "best" and selection < best_score) or (SELECT_MODE == "last" and epoch == EPOCHS - 1):
        best_score = selection
        best_epoch = epoch + 1
        if SAVE_CHECKPOINT:
            torch.save(
                {
                    "model": net.state_dict(),
                    "config": {
                        "tag": TAG,
                        "region_policy": REGION_POLICY,
                        "encoder_init_sha256": os.environ.get("ENCODER_SHA256", ""),
                        "optimizer_version": "method_screen_v1",
                        "objective": OBJECTIVE,
                        "sam_rho": SAM_RHO,
                        "prediction_bag": K_CROPS,
                        "schedule": "cosine35_floor005_warmup200",
                        "select_mode": SELECT_MODE,
                        "exclude_fold": EXCLUDE_FOLD,
                        "norm_mode": NORM_MODE,
                        "batch_size": BS,
                        "steps_per_epoch": steps_per_epoch,
                        "epochs": EPOCHS,
                        "enc_lr": ENC_LR,
                        "head_lr": HEAD_LR,
                        "fold": FOLD,
                        "nfolds": NFOLDS,
                        "fold_seed": FOLD_SEED,
                        "seed": SEED,
                        "crop": CROP,
                        "k_crops": K_CROPS,
                        "feature_agg": FEATURE_AGG,
                        "crop_mode": CROP_MODE,
                        "bag_pool": BAG_POOL,
                        "pool": POOL,
                        "loss_mode": LOSS_MODE,
                        "ysrc": YSRC,
                        "lm": LM.tolist(),
                        "ls": LS.tolist(),
                    },
                    "epoch": best_epoch,
                    "metrics": metrics.tolist(),
                },
                checkpoint_path,
            )
        np.savez_compressed(
            run_dir / "best_oof.npz",
            indices=val_idx,
            ids=np.asarray(IDS)[val_idx],
            pred=pred,
            target=Y[val_idx],
            metrics=metrics,
        )

final_metrics = None
if SAVE_CHECKPOINT and checkpoint_path.exists():
    saved = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    net.load_state_dict(saved["model"])
    final_pred, final_metrics_array = evaluate(val_idx, FINAL_D8)
    np.savez_compressed(
        run_dir / "best_oof.npz",
        indices=val_idx,
        ids=np.asarray(IDS)[val_idx],
        pred=final_pred,
        target=Y[val_idx],
        metrics=final_metrics_array,
    )
    final_metrics = final_metrics_array.tolist()
    print(
        f"final_tta val=[{final_metrics_array[0]:.3f},{final_metrics_array[1]:.3f},"
        f"{final_metrics_array[2]:.3f}] overall={final_metrics_array.mean():.3f}",
        flush=True,
    )

result = {
    "run_name": run_name,
    "region_policy": REGION_POLICY,
    "encoder_init_sha256": os.environ.get("ENCODER_SHA256", ""),
    "best_epoch": best_epoch,
    "selection_score": best_score,
    "final_metrics": final_metrics,
    "checkpoint": str(checkpoint_path),
    "select_mode": SELECT_MODE,
    "exclude_fold": EXCLUDE_FOLD,
    "optimizer_version": "method_screen_v1",
    "objective": OBJECTIVE,
    "sam_rho": SAM_RHO,
    "epochs": EPOCHS,
    "prediction_bag": K_CROPS,
    "schedule": "cosine_floor005_warmup200",
}
with open(run_dir / "result.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2), flush=True)
