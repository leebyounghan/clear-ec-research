#!/usr/bin/env python3
"""Paired, fixed-weight ablation of the historical six-member CLEAR-EC recipe.

Research only: reused OOF folds/checkpoints, no untouched test-set estimate.
The sixth SSL member is evaluated on the duplicate-clean primary subset only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FIRST_FIVE = ("seed42", "seed77", "seed2026", "metric35", "predbag2_35")
SCALE = np.array([0.9970446110788349, 0.987972130802936, 0.9892234872488541])


def path_for(root: Path, name: str, fold: int) -> Path:
    if name == "seed42":
        return root / f"followup_20260902/runs/fixed_base_f{fold}_seed42_c256k1_mean_multitask_slidesmooth/best_oof.npz"
    if name in ("seed77", "seed2026"):
        seed = name[4:]
        return root / f"submission_20260904/runs/base_seed{seed}_f{fold}_seed{seed}_c256k1_mean_multitask_slidesmooth/best_oof.npz"
    crops = 2 if name == "predbag2_35" else 1
    return root / f"methods_20260904/runs/{name}_f{fold}_seed42_c256k{crops}_mean_multitask_slidesmooth/best_oof.npz"


def read_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key].copy() for key in data.files}


def mape(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    valid = (target > 0) & np.isfinite(target)
    if not valid.any(axis=0).all() or pred.shape != target.shape or not np.isfinite(pred).all():
        raise ValueError("invalid prediction/target shape or values")
    return np.array([(np.abs(pred[:, j][valid[:, j]] - target[:, j][valid[:, j]]) /
                      target[:, j][valid[:, j]]).mean() * 100 for j in range(3)])


def summarize(pred: np.ndarray, target: np.ndarray, folds: np.ndarray) -> dict:
    metric = mape(pred, target)
    return {"overall": float(metric.mean()), "CD_CV_HEX": metric.tolist(),
            "per_fold": {str(f): float(mape(pred[folds == f], target[folds == f]).mean())
                         for f in np.unique(folds)}}


def paired_ci(left: np.ndarray, right: np.ndarray, target: np.ndarray,
              folds: np.ndarray, seed: int = 923, repeats: int = 2000) -> dict:
    """Exploratory image bootstrap, stratified by fold; not slide-clustered."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(folds == f) for f in np.unique(folds)]
    observed = float(mape(left, target).mean() - mape(right, target).mean())
    draws = np.empty(repeats)
    for i in range(repeats):
        sample = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        draws[i] = mape(left[sample], target[sample]).mean() - mape(right[sample], target[sample]).mean()
    return {"delta_lower_is_better": observed,
            "image_bootstrap_95pct": np.quantile(draws, [0.025, 0.975]).tolist(),
            "caveat": "Image-level bootstrap is descriptive only: within-slide dependence and historical model selection are not captured."}


def target_tails(pred: np.ndarray, target: np.ndarray) -> dict:
    result = {}
    for j, name in enumerate(("CD", "CV", "HEX")):
        valid = (target[:, j] > 0) & np.isfinite(target[:, j])
        y = target[valid, j]
        p = pred[valid, j]
        cuts = np.quantile(y, [0.1, 0.9])
        result[name] = {}
        for label, mask in (("bottom_10pct", y <= cuts[0]),
                            ("middle_80pct", (y > cuts[0]) & (y < cuts[1])),
                            ("top_10pct", y >= cuts[1])):
            result[name][label] = {"n": int(mask.sum()),
                                   "mape": float((np.abs(p[mask] - y[mask]) / y[mask]).mean() * 100),
                                   "signed_relative_bias_pct": float(((p[mask] - y[mask]) / y[mask]).mean() * 100)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, default=Path("experiments"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiments
    chunks = []
    for fold in range(5):
        entries = {name: read_npz(path_for(root, name, fold)) for name in FIRST_FIVE}
        ref = entries["seed42"]
        for name, item in entries.items():
            for key in ("indices", "ids", "target"):
                np.testing.assert_array_equal(ref[key], item[key], err_msg=f"{name} fold{fold} {key}")
            if len(np.unique(item["indices"])) != len(item["indices"]):
                raise ValueError(f"duplicate OOF indices: {name}, fold {fold}")
        chunks.append({"fold": np.full(len(ref["indices"]), fold),
                       "indices": ref["indices"], "ids": ref["ids"].astype(str),
                       "target": ref["target"].astype(float),
                       **{name: item["pred"].astype(float) for name, item in entries.items()}})
    joined = {key: np.concatenate([item[key] for item in chunks]) for key in chunks[0]}
    if len(np.unique(joined["indices"])) != len(joined["indices"]):
        raise ValueError("duplicate cross-fold OOF indices")
    y, f = joined["target"], joined["fold"]
    first = np.stack([joined[name] for name in FIRST_FIVE])
    report = {"scope": "Historical reused five-fold OOF; validation-selected base checkpoints. Fixed equal weights, no tuning or official-score prediction.",
              "n_first_five": len(y), "first_five": {}, "sixth": {}, "correlations": {}}
    for name, pred in zip(FIRST_FIVE, first):
        report["first_five"][name] = summarize(pred, y, f)
    five = first.mean(axis=0)
    report["first_five"]["equal_five"] = summarize(five, y, f)
    report["first_five"]["equal_five_calibrated"] = summarize(five * SCALE, y, f)
    report["first_five"]["equal_five_target_tails"] = target_tails(five, y)
    for idx, name in enumerate(FIRST_FIVE):
        pred = np.delete(first, idx, axis=0).mean(axis=0)
        report["first_five"][f"without_{name}"] = summarize(pred, y, f)
        report["first_five"][f"without_{name}_paired_ci"] = paired_ci(pred, five, y, f)
    for a, name_a in enumerate(FIRST_FIVE):
        for b in range(a + 1, 5):
            name_b = FIRST_FIVE[b]
            valid = (y > 0) & np.isfinite(y)
            ea = np.where(valid, (first[a] - y) / np.where(valid, y, 1), np.nan)
            eb = np.where(valid, (first[b] - y) / np.where(valid, y, 1), np.nan)
            report["correlations"][f"{name_a}__{name_b}"] = [float(np.corrcoef(ea[valid[:, j], j], eb[valid[:, j], j])[0, 1]) for j in range(3)]

    # The historical sixth raw/SSL comparisons have a primary exclusion mask.
    # Recompute the five-member reference from its five individual OOF arrays;
    # ensure agreement with the frozen prior reference before evaluating the sixth.
    subsets = []
    for fold in (2, 3, 4):
        parent = root / ("overnight_target_20260912" if fold < 4 else "target_fold4_20260912")
        ref = read_npz(parent / "preflight/reference.npz")
        mask = (ref["folds"] == fold) & ref["primary"].astype(bool)
        row = joined["fold"] == fold
        np.testing.assert_array_equal(joined["indices"][row], ref["indices"][ref["folds"] == fold])
        np.testing.assert_array_equal(joined["ids"][row], ref["ids"][ref["folds"] == fold].astype(str))
        np.testing.assert_array_equal(joined["target"][row], ref["target"][ref["folds"] == fold])
        np.testing.assert_allclose(five[row], ref["current5"][ref["folds"] == fold], atol=1e-5, rtol=1e-5)
        raw = read_npz(parent / f"runs/raw_fold{fold}/oof.npz")
        ssl = read_npz(root / f"raw_ssl_next_20260913/ssl_runs/fold{fold}/standard/oof.npz")
        for item in (raw, ssl):
            np.testing.assert_array_equal(ref["indices"][ref["folds"] == fold], item["indices"])
            np.testing.assert_array_equal(ref["ids"][ref["folds"] == fold].astype(str), item["ids"].astype(str))
        local_mask = ref["primary"][ref["folds"] == fold].astype(bool)
        subsets.append({"fold": np.full(local_mask.sum(), fold),
                        "target": ref["target"][ref["folds"] == fold][local_mask].astype(float),
                        "five": five[row][local_mask],
                        "raw": raw["pred"][local_mask].astype(float),
                        "ssl": ssl["pred"][local_mask].astype(float)})
    sub = {key: np.concatenate([item[key] for item in subsets]) for key in subsets[0]}
    report["sixth"]["n_primary"] = len(sub["target"])
    ys, fs = sub["target"], sub["fold"]
    report["sixth"]["five_only"] = summarize(sub["five"], ys, fs)
    report["sixth"]["five_only_target_tails"] = target_tails(sub["five"], ys)
    for name in ("raw", "ssl"):
        report["sixth"][f"{name}_standalone"] = summarize(sub[name], ys, fs)
        six = (5 * sub["five"] + sub[name]) / 6
        report["sixth"][f"six_with_{name}"] = summarize(six, ys, fs)
        report["sixth"][f"six_with_{name}_calibrated"] = summarize(six * SCALE, ys, fs)
        report["sixth"][f"six_with_{name}_paired_vs_five"] = paired_ci(six, sub["five"], ys, fs)
        report["sixth"][f"six_with_{name}_target_tails"] = target_tails(six, ys)
        valid = (ys > 0) & np.isfinite(ys)
        e5 = np.where(valid, (sub["five"] - ys) / np.where(valid, ys, 1), np.nan)
        e6 = np.where(valid, (sub[name] - ys) / np.where(valid, ys, 1), np.nan)
        report["sixth"][f"residual_corr_five_{name}"] = [float(np.corrcoef(e5[valid[:, j], j], e6[valid[:, j], j])[0, 1]) for j in range(3)]
    report["sixth"]["ssl_vs_raw_paired_ci"] = paired_ci(
        (5 * sub["five"] + sub["ssl"]) / 6,
        (5 * sub["five"] + sub["raw"]) / 6, ys, fs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), "n_first_five": len(y), "n_primary": len(ys),
                      "five": report["first_five"]["equal_five"]["overall"],
                      "raw_six": report["sixth"]["six_with_raw"]["overall"],
                      "ssl_six": report["sixth"]["six_with_ssl"]["overall"]}, indent=2))


if __name__ == "__main__":
    main()
