#!/usr/bin/env python3
"""Bounded single-model backbone screen against the frozen CPSAM control.

The image views, labels, splits, objective, optimizer, update count, seed, and
terminal-checkpoint policy match the September 28 CPSAM metric35 control. The
two new arms change the pretrained encoder and its fixed input normalization.
No ensemble, learned tile aggregation, model selection, or official release
change occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path("experiments/phase2_backbone_screen_20261001")
SOURCE = ROOT / "source/phase2_backbone_screen_20261001.py"
PREVIOUS = Path("experiments/phase2_metric35_ssl_20260928")
DATA = Path("datasets/clear_ec_official")
HF_CACHE = ROOT / "hf_cache/hub"
ARCHITECTURES = {
    "dino_vitb14": {
        "timm_name": "vit_base_patch14_dinov2.lvd142m",
        "repo": "timm/vit_base_patch14_dinov2.lvd142m",
        "revision": "4685c99dabffe5affac90bd99dbffd25801ae58d",
        "weight_sha256": "55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b",
        "dim": 768,
        "license": "apache-2.0",
    },
    "convnext_base": {
        "timm_name": "convnext_base.fb_in22k_ft_in1k",
        "repo": "timm/convnext_base.fb_in22k_ft_in1k",
        "revision": "67758fcbcf9f007ec7630bf8636b74a8f24e52a0",
        "weight_sha256": "4853efc8d5dd3ed9b0f888a9c601d7556373599f944f9ac79af8b8ec81aa7381",
        "dim": 1024,
        "license": "apache-2.0",
    },
}
SEED = 42
FOLDS = (2, 3)
EXPECTED_BASELINE = 9.133842659525422
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def old_api():
    sys.path.insert(0, str(PREVIOUS / "source"))
    import phase2_metric35_ssl as old
    old.frozen_api()
    return old


def weight_path(architecture: str) -> Path:
    item = ARCHITECTURES[architecture]
    return (HF_CACHE / ("models--" + item["repo"].replace("/", "--")) /
            "snapshots" / item["revision"] / "model.safetensors")


def metrics(predicted: np.ndarray, target: np.ndarray) -> list[float]:
    values = []
    for metric in range(3):
        valid = (target[:, metric] > 0) & np.isfinite(target[:, metric])
        if not valid.any() or not np.isfinite(predicted[:, metric]).all():
            raise ValueError(f"Invalid metric {metric}")
        values.append(float((np.abs(predicted[valid, metric] - target[valid, metric]) /
                             target[valid, metric]).mean() * 100))
    return values


def prepare() -> None:
    import timm
    import torch

    if (ROOT / "manifest.json").exists():
        raise FileExistsError(ROOT / "manifest.json")
    old = old_api()
    old_manifest = old.verify()
    if old_manifest["batch"] != 48 or old_manifest["k_crops"] != 1 or old_manifest["seed"] != SEED:
        raise ValueError("CPSAM control recipe changed")
    weights = {}
    for architecture, item in ARCHITECTURES.items():
        path = weight_path(architecture)
        actual = sha(path)
        if actual != item["weight_sha256"]:
            raise ValueError(f"Pretrained weight changed: {architecture}")
        weights[architecture] = dict(**item, path=str(path), sha256=actual)
    groups = []
    with np.load(DATA / "cache_native_meta_512.npz", allow_pickle=True) as meta:
        ids, labels = meta["ids"].astype(str), meta["Y"].astype(np.float32)
    with np.load(old_manifest["folds"]["2"]["reference"], allow_pickle=True) as ref:
        for fold in FOLDS:
            prior = old_manifest["folds"][str(fold)]
            with np.load(prior["split_path"]) as split:
                train, validation = split["train"], split["validation"]
            if len(train) != prior["train_rows"] or len(validation) != prior["validation_rows"]:
                raise ValueError(f"Split size changed: {fold}")
            mask = ref["folds"] == fold
            np.testing.assert_array_equal(ref["indices"][mask], validation)
            np.testing.assert_array_equal(ref["ids"][mask].astype(str), ids[validation])
            np.testing.assert_array_equal(ref["target"][mask], labels[validation])
            with np.load(PREVIOUS / "runs" / f"fold{fold}/control/oof.npz", allow_pickle=True) as control:
                np.testing.assert_array_equal(control["indices"], validation)
                np.testing.assert_array_equal(control["ids"].astype(str), ids[validation])
                np.testing.assert_array_equal(control["target"], labels[validation])
                group = dict(target=labels[validation][ref["primary"][mask].astype(bool)],
                             predicted=control["pred"][ref["primary"][mask].astype(bool)])
            groups.append(group)
    baseline = float(np.mean(metrics(np.concatenate([g["predicted"] for g in groups]),
                                     np.concatenate([g["target"] for g in groups]))))
    if abs(baseline - EXPECTED_BASELINE) > 1e-5:
        raise ValueError(f"Historical single-model control drift: {baseline}")
    manifest = dict(created_at=time.strftime("%FT%T%z"),
                    source_sha256=sha(SOURCE), old_manifest_sha256=sha(PREVIOUS / "manifest.json"),
                    weights=weights, folds=old_manifest["folds"],
                    baseline_single_mape=baseline,
                    setup=dict(seed=SEED, batch=48, k_crops=1, updates=5215,
                               objective="mixed_mape", encoder_lr=1e-5,
                               head_lr=1e-3, weight_decay=1e-4,
                               input="central 512; native 256 crops; C5 x D8 evaluation",
                               dino_pad="256 to 266 with replicate border; no cell-size resize",
                               head="mean pooled features, LayerNorm, three 128-width heads",
                               checkpoint="fixed terminal; no validation selection",
                               labels="train slide-smoothed, validate original",
                               primary="single-model mean CD/CV/HEX MAPE on 3572 duplicate-clean rows",
                               maximum_research_gpus=4, official_release_untouched=True),
                    runtime=dict(torch=torch.__version__, timm=timm.__version__))
    save(ROOT / "manifest.json", manifest)
    print(json.dumps({"prepared": True, "baseline_single_mape": baseline,
                      "weights": {key: value["sha256"] for key, value in weights.items()}}), flush=True)


def verify() -> dict:
    manifest = read(ROOT / "manifest.json")
    if sha(SOURCE) != manifest["source_sha256"] or \
            sha(PREVIOUS / "manifest.json") != manifest["old_manifest_sha256"]:
        raise ValueError("Research source or control manifest changed")
    for architecture in ARCHITECTURES:
        if sha(weight_path(architecture)) != manifest["weights"][architecture]["sha256"]:
            raise ValueError(f"Pretrained weight changed: {architecture}")
    return manifest


def build_model(architecture: str, device: str = "cuda"):
    import timm
    import torch
    from torch import nn
    from torch.nn import functional as F

    item = ARCHITECTURES[architecture]
    kwargs = dict(pretrained=True, num_classes=0)
    if architecture == "dino_vitb14":
        kwargs["img_size"] = 266
    encoder = timm.create_model(item["timm_name"], **kwargs)
    if hasattr(encoder, "set_grad_checkpointing"):
        encoder.set_grad_checkpointing(True)

    class SingleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.norm = nn.LayerNorm(item["dim"])
            def head():
                layers = nn.Sequential(nn.Linear(item["dim"], 128), nn.GELU(),
                                       nn.Dropout(0.2), nn.Linear(128, 1))
                nn.init.zeros_(layers[-1].weight)
                nn.init.zeros_(layers[-1].bias)
                return layers
            self.cd, self.cv, self.hex = head(), head(), head()
            self.register_buffer("input_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
            self.register_buffer("input_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        def forward(self, x):
            x = (x - self.input_mean) / self.input_std
            if architecture == "dino_vitb14":
                x = F.pad(x, (5, 5, 5, 5), mode="replicate")
                tokens = self.encoder.forward_features(x)
                vector = tokens[:, self.encoder.num_prefix_tokens:].mean(dim=1)
            else:
                feature_map = self.encoder.forward_features(x)
                vector = feature_map.mean(dim=(-2, -1))
            vector = self.norm(vector)
            return torch.cat((self.cd(vector), self.cv(vector), self.hex(vector)), dim=1)

    model = SingleModel().to(device)
    return model


def run_one(architecture: str, fold: int, smoke: bool, manifest: dict) -> dict:
    import cv2
    import torch
    from torch.nn import functional as F

    old = old_api()
    shared = old.frozen_api()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one isolated GPU is required")
    prior = manifest["folds"][str(fold)]
    with np.load(prior["split_path"]) as split:
        train, validation = split["train"].copy(), split["validation"].copy()
    with np.load(DATA / "cache_native_meta_512.npz", allow_pickle=True) as meta:
        ids, raw = meta["ids"].astype(str), meta["Y"].astype(np.float32)
    smoothed = old.slide_targets(ids, raw, train, validation)
    z = torch.tensor(((np.log(smoothed.clip(1e-6)) - shared.LM) / shared.LS).astype(np.float32), device="cuda")
    images = np.memmap(DATA / "img_native_512.dat", mode="r", dtype=np.uint8,
                       shape=(len(ids), 512, 512))
    views = old.CentralViews(images)
    destination = ROOT / ("smoke" if smoke else "runs") / f"fold{fold}" / architecture
    destination.mkdir(parents=True, exist_ok=False)
    shared.seed(SEED)
    network = build_model(architecture)
    config = dict(architecture=architecture, fold=fold, smoke=smoke,
                  seed=SEED, updates=3 if smoke else prior["updates"],
                  split_sha256=prior["split_sha256"],
                  pretrained_sha256=manifest["weights"][architecture]["sha256"],
                  validation_selection=False, single_model=True)
    save(destination / "config.json", config)
    timing = shared.train(network, views, train, z, None, config["updates"],
                          destination, 1e-5, SEED)
    with open(PREVIOUS / "runs" / f"fold{fold}/control/result.json") as stream:
        old_hash = json.load(stream)["training"]["first_three_batches_sha256"]
    if timing["first_three_batches_sha256"] != old_hash:
        raise ValueError("Training images/crops differ from CPSAM control")
    checkpoint = destination / "selected.pt"
    torch.save(dict(model=network.state_dict(), config=config), checkpoint)
    record = torch.load(checkpoint, map_location="cpu", weights_only=True)
    network.load_state_dict(record["model"], strict=True)
    evaluation = shared.evaluate(network, views, validation[:4] if smoke else validation,
                                 raw, ids, destination)
    result = dict(training=timing, evaluation=evaluation,
                  checkpoint_sha256=sha(checkpoint), oof_sha256=sha(destination / "oof.npz"),
                  control_first_three_batches_sha256=old_hash,
                  strict_reload=True, completed_at=time.strftime("%FT%T%z"))
    save(destination / "result.json", result)
    print(json.dumps({"architecture": architecture, "fold": fold,
                      "smoke": smoke, "complete": True,
                      "seconds_per_step": timing["elapsed_seconds"] / config["updates"],
                      "peak_gib": timing["peak_gib"],
                      "metrics": evaluation["metrics"]}), flush=True)
    del network, z, views, images
    torch.cuda.empty_cache()
    return result


def worker(architecture: str, fold: int) -> None:
    import torch
    manifest = verify()
    if (ROOT / "runs" / f"fold{fold}" / architecture / "result.json").exists():
        raise FileExistsError("Full run already exists; do not silently rerun")
    run_one(architecture, fold, smoke=True, manifest=manifest)
    torch.cuda.empty_cache()
    run_one(architecture, fold, smoke=False, manifest=manifest)
    save(ROOT / "runs" / f"fold{fold}" / architecture / "complete.json",
         dict(architecture=architecture, fold=fold, finished_at=time.strftime("%FT%T%z"),
              source_sha256=manifest["source_sha256"], verified_baseline=EXPECTED_BASELINE))


def audit() -> None:
    manifest = verify()
    with np.load(DATA / "cache_native_meta_512.npz", allow_pickle=True) as meta:
        ids, labels = meta["ids"].astype(str), meta["Y"].astype(np.float32)
    groups = []
    for fold in FOLDS:
        prior = manifest["folds"][str(fold)]
        with np.load(prior["split_path"]) as split:
            validation = split["validation"].copy()
        with np.load(prior["reference"], allow_pickle=True) as ref:
            mask = ref["folds"] == fold
            primary = ref["primary"][mask].astype(bool)
        group = dict(fold=fold, target=labels[validation][primary].astype(float), predictions={})
        files = {"cpsam_control": PREVIOUS / "runs" / f"fold{fold}/control/oof.npz"}
        for architecture in ARCHITECTURES:
            base = ROOT / "runs" / f"fold{fold}" / architecture
            result = read(base / "result.json")
            if not (base / "complete.json").exists() or \
                    sha(base / "selected.pt") != result["checkpoint_sha256"] or \
                    sha(base / "oof.npz") != result["oof_sha256"]:
                raise ValueError(f"Incomplete or changed run: {fold}/{architecture}")
            files[architecture] = base / "oof.npz"
        for name, file in files.items():
            with np.load(file, allow_pickle=True) as output:
                np.testing.assert_array_equal(output["indices"], validation)
                np.testing.assert_array_equal(output["ids"].astype(str), ids[validation])
                np.testing.assert_array_equal(output["target"], labels[validation])
                predicted = output["pred"].astype(float)
            if predicted.shape != (len(validation), 3) or not np.isfinite(predicted).all():
                raise ValueError(f"Invalid prediction {fold}/{name}")
            group["predictions"][name] = predicted[primary]
        groups.append(group)
    targets = np.concatenate([g["target"] for g in groups])
    report = dict(n=len(targets), folds=list(FOLDS), scores={},
                  input="central512 native256; all single-model scores, no ensemble",
                  caveat="Reused development folds; architecture and pretrained weights change together. Not an independent Phase II result.")
    for name in ("cpsam_control", *ARCHITECTURES):
        prediction = np.concatenate([g["predictions"][name] for g in groups])
        values = metrics(prediction, targets)
        report["scores"][name] = dict(overall=float(np.mean(values)), per_metric=values,
                                      per_fold={str(g["fold"]): float(np.mean(metrics(g["predictions"][name], g["target"])))
                                                for g in groups})
    if abs(report["scores"]["cpsam_control"]["overall"] - EXPECTED_BASELINE) > 1e-5:
        raise ValueError("CPSAM single-model baseline reconstruction changed")
    save(ROOT / "audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "worker", "audit"))
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES))
    parser.add_argument("--fold", type=int, choices=FOLDS)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
    if args.command == "prepare":
        prepare()
    elif args.command == "worker":
        if args.architecture is None or args.fold is None:
            parser.error("worker requires architecture and fold")
        worker(args.architecture, args.fold)
    else:
        audit()


if __name__ == "__main__":
    main()
