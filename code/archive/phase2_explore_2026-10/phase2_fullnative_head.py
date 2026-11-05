#!/usr/bin/env python3
"""Fold-clean head comparison on frozen, truly full-native CPSAM features.

No crop, resize, SSL update, validation-based checkpoint selection, or official
submission change. A spatial attention head is compared with mean pooling on
identical full-frame features and fixed historical six-member OOF predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path("experiments/phase2_fullnative_20260929")
FEATURES = ROOT / "features"
DATA = Path("datasets/clear_ec_official")
METRIC35 = Path("experiments/phase2_metric35_ssl_20260928/manifest.json")
SIXTH = Path("experiments/raw_ssl_next_20260913/ssl_runs")
LM = np.array([7.919919967651367, -0.9824771285057068,
               -0.6211431622505188], dtype=np.float32)
LS = np.array([0.18971863389015198, 0.18813571333885193,
               0.2748945653438568], dtype=np.float32)
FOLDS = (2, 3)
MODES = ("mean", "spatial_attention")
EPOCHS = 35
BATCH = 128


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load_features(ids: np.ndarray) -> torch.Tensor:
    expected = hashlib.sha256("\n".join(ids.tolist()).encode()).hexdigest()
    parts = []
    for shard in (0, 1):
        record = json.loads((FEATURES / f"shard{shard}.json").read_text())
        file = FEATURES / f"shard{shard}.dat"
        if record["ids_sha256"] != expected or record["feature_sha256"] != sha(file):
            raise ValueError(f"Shard {shard} identity or checksum mismatch")
        if record["cropped_or_resized"] or record["shape"] != [4477, 24, 512]:
            raise ValueError(f"Shard {shard} is not the full-native feature tensor")
        if record["start"] != 4477 * shard or record["end"] != 4477 * (shard + 1):
            raise ValueError(f"Shard {shard} indices are not aligned")
        part = np.memmap(file, mode="r", dtype=np.float16, shape=(4477, 24, 512))
        parts.append(np.asarray(part, dtype=np.float32))
    values = np.concatenate(parts, axis=0)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite full-native features")
    return torch.from_numpy(values)


def smooth_training(ids: np.ndarray, labels: np.ndarray,
                    train: np.ndarray, validation: np.ndarray) -> np.ndarray:
    with (DATA / "final_train_ids.csv").open(newline="") as stream:
        slides_by_id = {Path(row["ID"]).stem: row["slide_id"] for row in csv.DictReader(stream)}
    slides = np.array([slides_by_id[Path(identifier).stem] for identifier in ids])
    if set(slides[train]) & set(slides[validation]):
        raise ValueError("Validation slide enters head training")
    output = labels.copy()
    for slide in np.unique(slides[train]):
        members = train[slides[train] == slide]
        if len(members) > 1:
            output[members] = labels[members].mean(axis=0)
    return output


class SpatialHead(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.project = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 128),
                                     nn.GELU(), nn.Dropout(0.1))
        if mode == "spatial_attention":
            self.position = nn.Parameter(torch.zeros(24, 128))
            self.score = nn.Linear(128, 1)
        self.regressor = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 128),
                                       nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 3))
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        x = self.project(tiles)
        if self.mode == "spatial_attention":
            weights = self.score(torch.tanh(x + self.position)).softmax(dim=1)
            x = (weights * x).sum(dim=1)
        else:
            x = x.mean(dim=1)
        return self.regressor(x)


def train_head(features: torch.Tensor, labels: np.ndarray, ids: np.ndarray,
               train: np.ndarray, validation: np.ndarray, fold: int, mode: str) -> np.ndarray:
    output = ROOT / "heads" / f"fold{fold}" / mode
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    net = SpatialHead(mode)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    smoothed = smooth_training(ids, labels, train, validation)
    targets = torch.from_numpy(((np.log(smoothed.clip(1e-6)) - LM) / LS).astype(np.float32))
    scale = torch.from_numpy(LS)
    generator = torch.Generator().manual_seed(42)
    start = time.monotonic()
    for epoch in range(EPOCHS):
        net.train()
        perm = train[torch.randperm(len(train), generator=generator).numpy()]
        losses = []
        for subset in np.array_split(perm, max(1, (len(perm) + BATCH - 1) // BATCH)):
            predicted = net(features[subset])
            delta = predicted - targets[subset]
            loss = 0.5 * delta.abs().mean() + 0.5 * torch.expm1(delta * scale).abs().mean() / 0.2
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite head loss {fold}/{mode}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(json.dumps({"fold": fold, "mode": mode, "epoch": epoch + 1,
                              "loss": float(np.mean(losses)),
                              "elapsed_seconds": time.monotonic() - start}), flush=True)
    torch.save({"model": net.state_dict(), "fold": fold, "mode": mode,
                "epochs": EPOCHS, "seed": 42}, output / "terminal.pt")
    net.eval()
    chunks = []
    with torch.inference_mode():
        for subset in np.array_split(validation, max(1, (len(validation) + 255) // 256)):
            chunks.append(net(features[subset]).numpy())
    encoded = np.concatenate(chunks)
    physical = np.exp(encoded * LS + LM)
    if not np.isfinite(physical).all() or (physical <= 0).any():
        raise ValueError("Invalid full-native predictions")
    np.savez_compressed(output / "oof.npz", indices=validation, ids=ids[validation],
                        target=labels[validation], pred=physical)
    save(output / "result.json", {"fold": fold, "mode": mode,
                                  "checkpoint_sha256": sha(output / "terminal.pt"),
                                  "oof_sha256": sha(output / "oof.npz"),
                                  "training_seconds": time.monotonic() - start})
    return physical


def mape(predicted: np.ndarray, target: np.ndarray) -> dict:
    values = []
    for metric in range(3):
        valid = (target[:, metric] > 0) & np.isfinite(target[:, metric])
        if not valid.any() or not np.isfinite(predicted[:, metric]).all():
            raise ValueError(f"Invalid metric {metric}")
        values.append(float((np.abs(predicted[valid, metric] - target[valid, metric]) /
                             target[valid, metric]).mean() * 100))
    return {"overall": float(np.mean(values)), "per_metric": values}


def load_saved_predictions(fold: int, mode: str, validation: np.ndarray,
                           ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    output = ROOT / "heads" / f"fold{fold}" / mode
    result = json.loads((output / "result.json").read_text())
    if result["fold"] != fold or result["mode"] != mode:
        raise ValueError(f"Saved head metadata mismatch: {fold}/{mode}")
    if result["checkpoint_sha256"] != sha(output / "terminal.pt") or \
            result["oof_sha256"] != sha(output / "oof.npz"):
        raise ValueError(f"Saved head checksum mismatch: {fold}/{mode}")
    with np.load(output / "oof.npz", allow_pickle=True) as saved:
        np.testing.assert_array_equal(saved["indices"], validation)
        np.testing.assert_array_equal(saved["ids"].astype(str), ids[validation])
        np.testing.assert_array_equal(saved["target"], labels[validation])
        predicted = saved["pred"].astype(float)
    if predicted.shape != (len(validation), 3) or not np.isfinite(predicted).all() or (predicted <= 0).any():
        raise ValueError(f"Invalid saved predictions: {fold}/{mode}")
    return predicted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true",
                        help="Recompute scores from checksum-verified saved OOF predictions")
    args = parser.parse_args()
    torch.set_num_threads(8)
    with np.load(DATA / "cache_native_meta_512.npz", allow_pickle=True) as metadata:
        ids = metadata["ids"].astype(str)
        labels = metadata["Y"].astype(np.float32)
    features = None if args.audit_only else load_features(ids)
    manifest = json.loads(METRIC35.read_text())
    groups = []
    for fold in FOLDS:
        prior = manifest["folds"][str(fold)]
        with np.load(prior["split_path"]) as split:
            train, validation = split["train"].copy(), split["validation"].copy()
        if len(train) != prior["train_rows"] or len(validation) != prior["validation_rows"]:
            raise ValueError(f"Fold {fold} size changed")
        predictions = {mode: (load_saved_predictions(fold, mode, validation, ids, labels)
                              if args.audit_only else
                              train_head(features, labels, ids, train, validation, fold, mode))
                       for mode in MODES}
        with np.load(prior["reference"], allow_pickle=True) as reference:
            mask = reference["folds"] == fold
            np.testing.assert_array_equal(reference["indices"][mask], validation)
            np.testing.assert_array_equal(reference["ids"][mask].astype(str), ids[validation])
            np.testing.assert_array_equal(reference["target"][mask], labels[validation])
            primary = reference["primary"][mask].astype(bool)
            current5 = reference["current5"][mask].astype(float)
        with np.load(SIXTH / f"fold{fold}/standard/oof.npz", allow_pickle=True) as sixth:
            np.testing.assert_array_equal(sixth["indices"], validation)
            np.testing.assert_array_equal(sixth["target"], labels[validation])
            historical_six = (5 * current5 + sixth["pred"].astype(float)) / 6
        group = {"fold": fold, "target": labels[validation][primary].astype(float),
                 "historical_six": historical_six[primary]}
        for mode in MODES:
            group[mode] = predictions[mode][primary].astype(float)
            group[f"add7_{mode}"] = ((6 * historical_six + predictions[mode]) / 7)[primary]
        groups.append(group)
    target = np.concatenate([group["target"] for group in groups])
    scores = {}
    for name in ("historical_six", *MODES, *(f"add7_{mode}" for mode in MODES)):
        predicted = np.concatenate([group[name] for group in groups])
        scores[name] = {**mape(predicted, target),
                        "per_fold": {str(group["fold"]): mape(group[name], group["target"])["overall"]
                                     for group in groups}}
    if abs(scores["historical_six"]["overall"] - 9.074336039812735) > 1e-5:
        raise ValueError("Historical six baseline reconstruction changed")
    gate = {mode: (scores[f"add7_{mode}"]["overall"] <= scores["historical_six"]["overall"] - 0.01
                   and all(scores[f"add7_{mode}"]["per_fold"][str(fold)] <
                           scores["historical_six"]["per_fold"][str(fold)] for fold in FOLDS))
            for mode in MODES}
    report = {"n": len(target), "folds": list(FOLDS), "scores": scores, "screening_gate": gate,
              "input": "Entire native 972x1296 frame, 4-row padding only; frozen CPSAM full-grid attention",
              "encoder_finetuned": False, "official_release_untouched": True,
              "caveat": "Historical reused folds and historical validation-selected ensemble members; not independent Phase II score."}
    save(ROOT / "head_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
