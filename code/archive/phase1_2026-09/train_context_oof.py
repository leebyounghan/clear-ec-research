"""Full-frame versus central context/local OOF training.

Snapshot of train_oof_h200.py with unchanged fold, optimizer, and loss logic;
only the image-view provider and per-bag TTA evaluation differ. Never reads ROI GT.
"""
from __future__ import annotations

import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from cellpose import models
from fullframe_views import ContextViews


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
DEVICE = "cuda"
SZ = 512
VIEW_SCOPE = os.environ.get("VIEW_SCOPE", "central")
QUALITY_MODE = os.environ.get("QUALITY_MODE", "uniform")
if K_CROPS not in (1, 2):
    raise ValueError("context experiments require one native view or global+native pair")

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
context_views = ContextViews(IMGS, IDS, scope=VIEW_SCOPE, quality=QUALITY_MODE,
    two_views=K_CROPS == 2, raw_root=os.environ.get("FULLFRAME_ROOT", "datasets/clear_ec_fullframe_20260904"))


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


def train_batch(indices: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.stack([context_views.train(int(i)) for i in indices])).to(DEVICE)


def eval_origins():
    return range(5)


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
        if multiview:
            pooled = pooled.reshape(batch, views, -1)
            if BAG_POOL == "meanstd":
                pooled = torch.cat((pooled.mean(dim=1), pooled.std(dim=1)), dim=1)
            elif BAG_POOL == "attn":
                pooled = self.bag_attn(
                    self.bag_query.expand(batch, -1, -1), pooled, pooled
                )[0].squeeze(1)
            else:
                pooled = pooled.mean(dim=1)
        pooled = self.norm(pooled)
        return torch.cat((self.cd(pooled), self.cv(pooled), self.hex(pooled)), dim=1)


pretrained = os.environ.get("CPSAM_MODEL", "cpsam_models/cpsam_v2")
base = models.CellposeModel(gpu=True, pretrained_model=pretrained)
encoder_init = os.environ.get("ENCODER_INIT", "")
if encoder_init:
    payload = torch.load(encoder_init, map_location="cpu", weights_only=True)
    state = payload.get("teacher_encoder", payload.get("encoder", payload))
    message = base.net.encoder.load_state_dict(state, strict=False)
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
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 200))

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
        batch_indices = indices[start:start + EVAL_IMAGE_BS]
        array = np.stack([context_views.evaluate(int(i), use_d8) for i in batch_indices])
        batch, bags, members = array.shape[:3]
        inputs = torch.from_numpy(array.reshape(batch * bags, members, 3, 256, 256)).to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Each bag contains the same native-local/global members at train and test.
            # TTA is averaged between bag predictions, never inside spatial statistics.
            encoded = net(inputs).float().reshape(batch, bags, 3).mean(dim=1)
        predictions.append(torch.exp(encoded * LS_T + LM_T).cpu().numpy())
    pred = np.concatenate(predictions)
    target = Y[indices]
    values = []
    for m in range(3):
        valid = (target[:,m] > 0) & np.isfinite(target[:,m])
        values.append(np.mean(np.abs(pred[valid,m] - target[valid,m]) / target[valid,m]) * 100)
    assert np.isfinite(pred).all() and np.isfinite(values).all()
    return pred, np.asarray(values, dtype=np.float32)


print(
    f"[{run_name}] train={len(train_idx)} val={len(val_idx)} epochs={EPOCHS} "
    f"bs={BS} k_crops={K_CROPS} feature_agg={FEATURE_AGG} "
    f"crop_mode={CROP_MODE} bag_pool={BAG_POOL} scope={VIEW_SCOPE} quality={QUALITY_MODE} "
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
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = net(train_batch(indices))
            element_loss = torch.abs(output - Z_T[indices])
            loss = element_loss[:, 0].mean() if LOSS_MODE == "cd" else element_loss.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
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
                        "optimizer_version": "all_trainable_v2",
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
                        "view_scope": VIEW_SCOPE,
                        "quality_mode": QUALITY_MODE,
                        "context_version": "global_native_v1",
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
    "best_epoch": best_epoch,
    "selection_score": best_score,
    "final_metrics": final_metrics,
    "checkpoint": str(checkpoint_path),
    "select_mode": SELECT_MODE,
    "exclude_fold": EXCLUDE_FOLD,
    "optimizer_version": "all_trainable_v2",
}
with open(run_dir / "result.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2), flush=True)

