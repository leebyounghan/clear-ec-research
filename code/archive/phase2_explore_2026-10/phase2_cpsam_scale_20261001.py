#!/usr/bin/env python3
"""Fold-2, one-epoch, warm-started CPSAM input-field screen.

All three arms start from the same completed native256 control checkpoint and
see the same 7153 training IDs, in the same order, once. This is a bounded
scale *screen*, not a replacement for the 35-epoch control or a Phase II score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path("experiments/phase2_cpsam_scale_20261001")
SOURCE = ROOT / "source/phase2_cpsam_scale_20261001.py"
DATA = Path("datasets/clear_ec_official")
FULL = Path("datasets/clear_ec_fullframe_20260904")
PREVIOUS = Path("experiments/phase2_metric35_ssl_20260928")
FROZEN = Path("experiments/crop_dino_screen_20260910/source/frozen")
ARMS = ("native256", "central512", "full972x1296")
FOLD = 2
SEED = 42
FOUNDATION_SHA = "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def imports():
    sys.path.insert(0, str(FROZEN))
    from final_candidate_core import build_model, LM, LS
    from method_training_core import regression_loss, cosine_multiplier
    return build_model, LM, LS, regression_loss, cosine_multiplier


class FullGridEncoder(nn.Module):
    """CPSAM encoder with interpolated position grid and checkpointed blocks."""

    def __init__(self, original: nn.Module):
        super().__init__()
        self.original = original

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.original.patch_embed(image)
        position = self.original.pos_embed
        if position is not None:
            if position.shape[1:3] != x.shape[1:3]:
                position = F.interpolate(position.permute(0, 3, 1, 2),
                                         size=x.shape[1:3], mode="bicubic",
                                         align_corners=False).permute(0, 2, 3, 1)
            x = x + position
        for block in self.original.blocks:
            x = checkpoint(block, x, use_reentrant=False)
        return self.original.neck(x.permute(0, 3, 1, 2))


def prepare() -> None:
    if (ROOT / "manifest.json").exists():
        raise FileExistsError("Scale screen already prepared")
    prior = read(PREVIOUS / "manifest.json")
    fold = prior["folds"][str(FOLD)]
    split = Path(fold["split_path"])
    checkpoint_path = PREVIOUS / f"runs/fold{FOLD}/control/selected.pt"
    if digest(split) != fold["split_sha256"] or digest(Path("cpsam_models/cpsam_v2")) != FOUNDATION_SHA:
        raise ValueError("Changed split or CPSAM foundation")
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    full_manifest = read(FULL / "fullframe_manifest.json")
    if full_manifest["cache_shape"] != [8954, 972, 1296] or not full_manifest["central_pixels_exact_match"]:
        raise ValueError("Full-frame provenance check failed")
    with np.load(DATA / "cache_native_meta_512.npz", allow_pickle=True) as d:
        ids, y = d["ids"].astype(str), d["Y"].astype(np.float32)
    with np.load(FULL / "fullframe_meta.npz", allow_pickle=True) as d:
        np.testing.assert_array_equal(ids, d["ids"].astype(str))
    with np.load(split) as d:
        train, validation = d["train"], d["validation"]
    if len(train) != 7153 or len(validation) != 1791 or np.intersect1d(train, validation).size:
        raise ValueError("Unexpected fold geometry")
    if not np.isfinite(y).all() or not (y[train] > 0).all():
        raise ValueError("Invalid training targets")
    with np.load(fold["reference"], allow_pickle=True) as d:
        mask = d["folds"] == FOLD
        np.testing.assert_array_equal(d["indices"][mask], validation)
        np.testing.assert_array_equal(d["ids"][mask].astype(str), ids[validation])
        np.testing.assert_array_equal(d["target"][mask], y[validation])
        primary = d["primary"][mask].astype(bool)
    if primary.sum() != fold["primary_rows"]:
        raise ValueError("Primary subset mismatch")
    write(ROOT / "manifest.json", dict(
        created_at=time.strftime("%FT%T%z"), source_sha256=digest(SOURCE),
        baseline_checkpoint=str(checkpoint_path), baseline_sha256=digest(checkpoint_path),
        split_path=str(split), split_sha256=digest(split), reference=str(fold["reference"]),
        foundation_sha256=FOUNDATION_SHA, full_manifest_sha256=digest(FULL / "fullframe_manifest.json"),
        central_meta_sha256=digest(DATA / "cache_native_meta_512.npz"),
        full_meta_sha256=digest(FULL / "fullframe_meta.npz"),
        train_rows=len(train), validation_rows=len(validation), primary_rows=int(primary.sum()),
        arms=list(ARMS), fold=FOLD, seed=SEED, epochs=1, batch=1,
        training="Same warm start, train-ID order, optimizer and exposure count; input field differs",
        normalization="CLAHE ~32px physical tile; per-image 1st/99th percentile; no resize",
        model="CPSAM mean-pooled encoder, existing CD/CV/HEX heads; full-grid position interpolation",
        validation="One deterministic centered view per image; raw labels; zero targets excluded from MAPE",
        gate="Feasibility and directional screen only; no model promotion without fold3 and longer learning curves",
        official_release_untouched=True))
    print(json.dumps({"prepared": True, "train_rows": len(train), "validation_rows": len(validation)}), flush=True)


def verify() -> dict:
    m = read(ROOT / "manifest.json")
    checks = ((SOURCE, "source_sha256"), (Path(m["baseline_checkpoint"]), "baseline_sha256"),
              (Path(m["split_path"]), "split_sha256"),
              (FULL / "fullframe_manifest.json", "full_manifest_sha256"),
              (DATA / "cache_native_meta_512.npz", "central_meta_sha256"),
              (FULL / "fullframe_meta.npz", "full_meta_sha256"))
    for path, key in checks:
        if digest(path) != m[key]:
            raise ValueError(f"Changed {key}: {path}")
    return m


def seed_all() -> None:
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


def load_images(arm: str, count: int):
    if arm == "full972x1296":
        return np.memmap(FULL / "img_full_raw.dat", mode="r", dtype=np.uint8,
                         shape=(count, 972, 1296))
    return np.memmap(DATA / "img_native_512.dat", mode="r", dtype=np.uint8,
                     shape=(count, 512, 512))


def input_image(raw: np.ndarray, arm: str, augmentation: int = 0) -> torch.Tensor:
    if arm == "native256":
        raw = raw[128:384, 128:384]
        grid = (8, 8)
    elif arm == "central512":
        grid = (16, 16)
    else:
        grid = (40, 30)
    normalized = cv2.createCLAHE(2.0, grid).apply(np.ascontiguousarray(raw))
    low, high = np.percentile(normalized, (1, 99))
    normalized = np.clip((normalized.astype(np.float32) - low) / (high - low + 1e-6), 0, 1)
    if augmentation & 1:
        normalized = normalized[::-1]
    if augmentation & 2:
        normalized = normalized[:, ::-1]
    if arm == "full972x1296":
        normalized = np.pad(normalized, ((0, 4), (0, 0)), mode="edge")
    return torch.from_numpy(np.stack([np.ascontiguousarray(normalized)] * 3))


def net_from_baseline(m: dict, arm: str):
    build_model, _, _, _, _ = imports()
    net = build_model("cpsam_models/cpsam_v2")
    saved = torch.load(m["baseline_checkpoint"], map_location="cpu", weights_only=True)
    net.load_state_dict(saved["model"], strict=True)
    del saved
    if arm != "native256":
        net.encoder = FullGridEncoder(net.encoder)
    else:
        for block in net.encoder.blocks:
            original = block.forward
            block.forward = lambda x, fn=original: checkpoint(fn, x, use_reentrant=False)
    return net


def score(pred: np.ndarray, target: np.ndarray) -> list[float]:
    result = []
    for j in range(3):
        valid = target[:, j] > 0
        if not valid.any() or not np.isfinite(pred[:, j]).all():
            raise ValueError(f"Invalid predictions/targets for metric {j}")
        result.append(float((np.abs(pred[valid, j] - target[valid, j]) / target[valid, j]).mean() * 100))
    return result


def worker(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(arm)
    m = verify()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one isolated GPU is required")
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    out = ROOT / "runs" / arm
    out.mkdir(parents=True, exist_ok=False)
    with np.load(m["split_path"]) as d:
        train, validation = d["train"].copy(), d["validation"].copy()
    with np.load(DATA / "cache_native_meta_512.npz", allow_pickle=True) as d:
        ids, labels = d["ids"].astype(str), d["Y"].astype(np.float32)
    sys.path.insert(0, str(PREVIOUS / "source"))
    from phase2_metric35_ssl import slide_targets
    smoothed = slide_targets(ids, labels, train, validation)
    build_model, LM, LS, regression_loss, cosine_multiplier = imports()
    targets = torch.tensor(((np.log(smoothed.clip(1e-6)) - LM) / LS).astype(np.float32), device="cuda")
    images = load_images(arm, len(ids))
    seed_all()
    net = net_from_baseline(m, arm).cuda().train()
    enc = list(net.encoder.parameters())
    enc_ids = {id(p) for p in enc}
    heads = [p for p in net.parameters() if id(p) not in enc_ids]
    optimizer = torch.optim.AdamW([dict(params=enc, lr=3e-6), dict(params=heads, lr=3e-4)],
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda i: cosine_multiplier(i, len(train)))
    scale = torch.tensor(LS, device="cuda")
    order = np.random.default_rng(SEED * 1000).permutation(train)
    order_sha = hashlib.sha256(order.tobytes()).hexdigest()
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step, index in enumerate(order):
        array = input_image(images[int(index)], arm, int(np.random.randint(4)))
        x = array[None].cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = net(x)
            loss = regression_loss(prediction, targets[int(index)][None], scale, "mixed_mape")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        if step in (1, 24) or (step + 1) % 100 == 0 or step + 1 == len(order):
            elapsed = time.perf_counter() - start
            state = dict(stage="training", arm=arm, step=step + 1, total=len(order),
                         seconds_per_step=elapsed / (step + 1), peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                         mean_loss=float(loss.detach()), updated_at=time.strftime("%FT%T%z"))
            write(out / "progress.json", state)
            print(json.dumps(state), flush=True)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    del optimizer, scheduler
    for p in net.parameters():
        p.grad = None
    checkpoint_path = out / "selected.pt"
    torch.save(dict(model=net.state_dict(), config=dict(arm=arm, fold=FOLD,
        epochs=1, warm_start_sha256=m["baseline_sha256"], train_order_sha256=order_sha,
        source_sha256=m["source_sha256"], validation_selection=False)), checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    net.load_state_dict(loaded["model"], strict=True)
    net.eval()
    predictions = []
    start_eval = time.perf_counter()
    with torch.inference_mode():
        for n, index in enumerate(validation):
            x = input_image(images[int(index)], arm)[None].cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                encoded = net(x).float()[0]
            prediction = torch.exp(encoded * torch.as_tensor(LS, device="cuda") +
                                   torch.as_tensor(LM, device="cuda"))
            predictions.append(prediction.cpu().numpy())
            if (n + 1) % 100 == 0:
                state = dict(stage="evaluation", arm=arm, images=n + 1, total=len(validation),
                             elapsed_seconds=time.perf_counter() - start_eval,
                             updated_at=time.strftime("%FT%T%z"))
                write(out / "progress.json", state)
                print(json.dumps(state), flush=True)
    predicted = np.stack(predictions)
    if not np.isfinite(predicted).all() or not (predicted > 0).all():
        raise ValueError("Invalid model predictions")
    np.savez_compressed(out / "oof.npz", indices=validation, ids=ids[validation],
                        target=labels[validation], pred=predicted)
    result = dict(arm=arm, fold=FOLD, train_images=len(order),
                  train_order_sha256=order_sha, train_seconds=train_seconds,
                  evaluation_seconds=time.perf_counter() - start_eval,
                  peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                  metrics=score(predicted, labels[validation]),
                  checkpoint_sha256=digest(checkpoint_path), oof_sha256=digest(out / "oof.npz"),
                  source_sha256=m["source_sha256"], completed_at=time.strftime("%FT%T%z"))
    write(out / "result.json", result)
    write(out / "complete.json", dict(arm=arm, fold=FOLD, result_sha256=digest(out / "result.json")))
    print(json.dumps({"complete": True, **result}), flush=True)


def audit() -> None:
    m = verify()
    with np.load(DATA / "cache_native_meta_512.npz", allow_pickle=True) as d:
        ids, labels = d["ids"].astype(str), d["Y"].astype(np.float32)
    with np.load(m["split_path"]) as d:
        validation = d["validation"]
    with np.load(m["reference"], allow_pickle=True) as d:
        primary = d["primary"][d["folds"] == FOLD].astype(bool)
    output = dict(fold=FOLD, n=int(primary.sum()), warm_start_sha256=m["baseline_sha256"],
                  arms={}, caveat="One warm-started epoch, batch1, one view, reused fold2. Not comparable directly to the 35-epoch/40-view CPSAM control or Phase II.")
    order = None
    for arm in ARMS:
        base = ROOT / "runs" / arm
        result = read(base / "result.json")
        complete = read(base / "complete.json")
        if complete["result_sha256"] != digest(base / "result.json") or \
                digest(base / "selected.pt") != result["checkpoint_sha256"] or \
                digest(base / "oof.npz") != result["oof_sha256"] or \
                result["source_sha256"] != m["source_sha256"]:
            raise ValueError(f"Changed or incomplete run: {arm}")
        if order is None:
            order = result["train_order_sha256"]
        elif order != result["train_order_sha256"]:
            raise ValueError("Train ID order differs across arms")
        with np.load(base / "oof.npz", allow_pickle=True) as d:
            np.testing.assert_array_equal(d["indices"], validation)
            np.testing.assert_array_equal(d["ids"].astype(str), ids[validation])
            np.testing.assert_array_equal(d["target"], labels[validation])
            pred = d["pred"].astype(float)
        per_metric = score(pred[primary], labels[validation][primary])
        output["arms"][arm] = dict(overall=float(np.mean(per_metric)), per_metric=per_metric,
                                  train_seconds=result["train_seconds"],
                                  evaluation_seconds=result["evaluation_seconds"],
                                  peak_gib=result["peak_gib"])
    write(ROOT / "audit.json", output)
    print(json.dumps(output, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "worker", "audit"))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "worker":
        if args.arm is None:
            parser.error("worker requires --arm")
        worker(args.arm)
    else:
        audit()


if __name__ == "__main__":
    main()
