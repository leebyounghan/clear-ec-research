#!/usr/bin/env python3
"""Fixed-weight OOF ensembles; exploratory cross-fitted scalar calibration.

No leaderboard labels, ROI annotations, or all-data model predictions are used.
Cross-fitting calibration on ordinary OOF predictions is NOT fully nested
base-model validation. Reports explicitly retain that limitation.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

NEW_ROOT = Path("experiments/submission_20260904")


def metrics(pred, target):
    valid = (target > 0) & np.isfinite(target)
    if pred.shape != target.shape or not np.isfinite(pred).all():
        raise ValueError("invalid predictions")
    if not valid.sum(0).all():
        raise ValueError("metric without valid labels")
    return (np.where(valid, np.abs(pred - target) / np.where(valid, target, 1), 0)
            .sum(0) / valid.sum(0) * 100)


def crossfit_scale(pred, target, folds):
    output = np.empty_like(pred, dtype=float)
    choices = {}
    for fold in range(5):
        test = folds == fold
        if not test.any():
            raise ValueError("missing fold")
        choices[str(fold)] = []
        for m in range(3):
            train = (~test) & (target[:, m] > 0) & np.isfinite(target[:, m])
            p, y = pred[train, m], target[train, m]
            if not len(p) or (p <= 0).any():
                raise ValueError("calibration needs positive predictions and labels")
            ratios, weights = y / p, p / y
            order = np.argsort(ratios)
            j = np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)
            scale = float(ratios[order][j])
            output[test, m] = pred[test, m] * scale
            choices[str(fold)].append(scale)
    return output, choices


def load_group(paths):
    if len(paths) != 5:
        raise ValueError("exactly five folds required")
    entries = []
    for fold, run in enumerate(paths):
        with np.load(run / "best_oof.npz", allow_pickle=True) as d:
            indices, ids = d["indices"].copy(), d["ids"].astype(str)
            pred, target = d["pred"].astype(float), d["target"].astype(float)
        metrics(pred, target)
        split_path = run / "split_indices.npz"
        if split_path.exists():
            with np.load(split_path) as split:
                assert np.array_equal(indices, split["validation"])
                assert not np.intersect1d(indices, split["train"]).size
        entries.append((indices, ids, pred, target, np.full(len(indices), fold)))
    data = [np.concatenate([e[k] for e in entries]) for k in range(5)]
    if len(np.unique(data[0])) != len(data[0]):
        raise ValueError("duplicate OOF indices")
    order = np.argsort(data[0])
    return tuple(x[order] for x in data)


def sources(new_root=NEW_ROOT):
    def paths(root, tag, seed=42, pool="mean", ysrc="slidesmooth"):
        return [Path(root) / f"{tag}_f{f}_seed{seed}_c256k1_{pool}_multitask_{ysrc}" for f in range(5)]
    return {
        "original_base": paths("experiments/oof_h200", "baseline"),
        "original_jepa": paths("experiments/oof_h200", "jepa_small_mean"),
        "fixed_base": paths("experiments/followup_20260902/runs", "fixed_base"),
        "fixed_jepa": paths("experiments/followup_20260902/runs", "fixed_jepa"),
        "base_seed77": paths(new_root / "runs", "base_seed77", seed=77),
        "base_seed2026": paths(new_root / "runs", "base_seed2026", seed=2026),
        "attention": paths(new_root / "runs", "attention", pool="attn"),
        "raw_targets": paths(new_root / "runs", "raw_targets", ysrc="raw"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=NEW_ROOT)
    args = parser.parse_args()
    loaded, incomplete = {}, []
    for name, paths in sources(args.root).items():
        # best_oof is intermediate until final D8 evaluation emits result.json.
        if not all((p / "result.json").exists() and (p / "best_oof.npz").exists() for p in paths):
            incomplete.append(name)
            continue
        loaded[name] = load_group(paths)
    if not loaded:
        raise RuntimeError("no complete five-fold groups")
    reference = next(iter(loaded.values()))
    for name, data in loaded.items():
        for k in (0, 1, 3, 4):
            aligned = (np.array_equal(data[k], reference[k], equal_nan=True) if k == 3
                       else np.array_equal(data[k], reference[k]))
            if not aligned:
                raise ValueError(f"OOF alignment mismatch: {name}, field {k}")
    indices, ids, _, target, folds = reference
    predictions = {name: data[2] for name, data in loaded.items()}
    # Fixed equal weights; never optimize weights against the reported rows.
    blends = {
        "historical_pair": ["original_base", "original_jepa"],
        "corrected_pair": ["fixed_base", "fixed_jepa"],
        "base_seed77_pair": ["fixed_base", "base_seed77"],
        "base_three_seeds": ["fixed_base", "base_seed77", "base_seed2026"],
        "base_attention_pair": ["fixed_base", "attention"],
        "base_raw_pair": ["fixed_base", "raw_targets"],
    }
    for name, members in blends.items():
        if all(m in predictions for m in members):
            predictions[name] = np.mean([predictions[m] for m in members], axis=0)
    report = dict(
        validation_scope="Ordinary five-fold OOF; fixed equal-weight ensembles. Scalar calibration is exploratory meta cross-fitting, NOT fully nested base-model validation. Historical data reused. No leaderboard score prediction.",
        n=len(target), invalid_targets_by_metric=((target <= 0) | ~np.isfinite(target)).sum(0).tolist(),
        complete_groups=list(loaded), incomplete_groups=incomplete, methods={}, residual_correlations={},
    )
    saved = {}
    for name, pred in predictions.items():
        scaled, scales = crossfit_scale(pred, target, folds)
        for variant, values in ((name, pred), (name + "__crossfit_scale", scaled)):
            mm = metrics(values, target)
            report["methods"][variant] = dict(metrics=mm.tolist(), overall=float(mm.mean()),
                per_fold=[float(metrics(values[folds == f], target[folds == f]).mean()) for f in range(5)])
            saved[variant] = values
        report["methods"][name + "__crossfit_scale"]["scales"] = scales
    for a, b in itertools.combinations(loaded, 2):
        correlations = []
        for m in range(3):
            valid = (target[:, m] > 0) & np.isfinite(target[:, m])
            ea = (loaded[a][2][valid, m] - target[valid, m]) / target[valid, m]
            eb = (loaded[b][2][valid, m] - target[valid, m]) / target[valid, m]
            corr = float(np.corrcoef(ea, eb)[0, 1])
            correlations.append(corr if np.isfinite(corr) else None)
        report["residual_correlations"][a + "__" + b] = correlations
    args.root.mkdir(parents=True, exist_ok=True)
    # Each complete-group count gets its own snapshot; old results are preserved.
    destination = args.root / f"ensemble_audit_{len(loaded):02d}_groups.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(destination)
    np.savez_compressed(args.root / f"ensemble_predictions_{len(loaded):02d}_groups.npz",
                        indices=indices, ids=ids, target=target, folds=folds, **saved)
    print(json.dumps({"report": str(destination), "groups": list(loaded),
                      "scores": {k: v["overall"] for k, v in report["methods"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
