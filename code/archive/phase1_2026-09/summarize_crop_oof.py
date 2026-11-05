#!/usr/bin/env python3
"""Cross-fitted audit of crop aggregation for C5 OOF predictions."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


BASE = Path(os.environ.get("OOF_ROOT", "experiments/oof_h200"))
TAG = os.environ.get("TAG", "baseline")
SEED = int(os.environ.get("SEED", "42"))
LM = np.array([7.919919967651367, -0.9824771285057068, -0.6211431622505188], np.float64)
LS = np.array([0.18971863389015198, 0.18813571333885193, 0.2748945653438568], np.float64)


def run_dir(fold: int) -> Path:
    return BASE / f"{TAG}_f{fold}_seed{SEED}_c256k1_mean_multitask_slidesmooth"


parts = []
for fold in range(5):
    path = run_dir(fold) / "crop_tiles.npz"
    data = np.load(path, allow_pickle=True)
    parts.append({key: data[key] for key in data.files} | {"fold": np.full(len(data["target"]), fold)})

z = np.concatenate([p["z_tile"] for p in parts]).astype(np.float64)
q = np.concatenate([p["quality_tile"] for p in parts]).astype(np.float64)
y = np.concatenate([p["target"] for p in parts]).astype(np.float64)
folds = np.concatenate([p["fold"] for p in parts])
valid_y = (y > 0) & np.isfinite(y)
true_z = (np.log(np.where(valid_y, y, 1.0)) - LM) / LS


def decode(values: np.ndarray) -> np.ndarray:
    return np.exp(values * LS + LM)


def metrics(pred: np.ndarray) -> list[float]:
    assert np.isfinite(pred).all(), "non-finite predictions"
    return [float(np.mean(np.abs(pred[valid_y[:, m], m] - y[valid_y[:, m], m])
                          / y[valid_y[:, m], m]) * 100) for m in range(3)]


def metric_one(pred: np.ndarray, index: np.ndarray, metric: int) -> float:
    keep = valid_y[index, metric]
    return float(np.mean(np.abs(pred[keep] - y[index[keep], metric]) / y[index[keep], metric]))


outputs: dict[str, np.ndarray] = {
    "current_geomean": decode(z.mean(axis=1)),
    "median_z": decode(np.median(z, axis=1)),
    "trimmed_z": decode(np.sort(z, axis=1)[:, 1:4].mean(axis=1)),
    "arithmetic_pred_mean": np.exp(z * LS[None, None, :] + LM[None, None, :]).mean(axis=1),
}

# Honest outer-fold selection of a per-metric tile-prediction quantile.
quantile_z = np.empty_like(true_z)
quantile_choices: dict[str, dict[str, float]] = {}
for outer in range(5):
    train = folds != outer
    test = folds == outer
    quantile_choices[str(outer)] = {}
    for metric in range(3):
        best = None
        for value in np.linspace(0, 1, 41):
            candidate_z = np.quantile(z[train, :, metric], value, axis=1)
            candidate_y = np.exp(candidate_z * LS[metric] + LM[metric])
            score = metric_one(candidate_y, np.where(train)[0], metric)
            if best is None or score < best[0]:
                best = (score, value)
        assert best is not None
        quantile_choices[str(outer)][str(metric)] = float(best[1])
        quantile_z[test, metric] = np.quantile(z[test, :, metric], best[1], axis=1)
outputs["crossfit_quantile"] = decode(quantile_z)

# A scalar-only control distinguishes crop selection from simple MAPE calibration.
scaled_mean = np.empty_like(y)
for outer in range(5):
    test = folds == outer
    for metric in range(3):
        train = (folds != outer) & valid_y[:, metric]
        pred = outputs["current_geomean"][train, metric]
        target = y[train, metric]
        ratios, weights = target / pred, pred / target
        order = np.argsort(ratios)
        scale = ratios[order][np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)]
        scaled_mean[test, metric] = outputs["current_geomean"][test, metric] * scale
outputs["crossfit_scaled_mean"] = scaled_mean

# Cross-fitted focus weighting. Quality dimensions: contrast, Laplacian variance,
# gradient magnitude, saturation. Alpha=0 recovers equal weighting.
focus_z = np.empty_like(true_z)
focus_choices: dict[str, dict[str, list[float]]] = {}
alphas = [-8, -4, -2, -1, -.5, 0, .5, 1, 2, 4, 8]
q_work = q.copy()
q_work[:, :, 1:3] = np.log1p(q_work[:, :, 1:3] * 1e4)
for outer in range(5):
    train = folds != outer
    test = folds == outer
    focus_choices[str(outer)] = {}
    for metric in range(3):
        best = None
        for feature in range(q.shape[2]):
            centre = np.median(q_work[train, :, feature])
            scale = np.std(q_work[train, :, feature]) + 1e-8
            score_all = (q_work[:, :, feature] - centre) / scale
            for alpha in alphas:
                logits = alpha * score_all
                logits -= logits.max(axis=1, keepdims=True)
                weights = np.exp(logits)
                weights /= weights.sum(axis=1, keepdims=True)
                candidate_z = np.sum(weights[train] * z[train, :, metric], axis=1)
                candidate_y = np.exp(candidate_z * LS[metric] + LM[metric])
                score = metric_one(candidate_y, np.where(train)[0], metric)
                if best is None or score < best[0]:
                    best = (score, feature, alpha, weights)
        assert best is not None
        _, feature, alpha, weights = best
        focus_choices[str(outer)][str(metric)] = [int(feature), float(alpha)]
        focus_z[test, metric] = np.sum(weights[test] * z[test, :, metric], axis=1)
outputs["crossfit_focus"] = decode(focus_z)


def features(metric: int, with_quality: bool) -> np.ndarray:
    tile = z[:, :, metric]
    base = np.concatenate(
        [tile, np.sort(tile, axis=1), tile.mean(1, keepdims=True), tile.std(1, keepdims=True)], axis=1
    )
    return np.concatenate([base, q_work.reshape(len(q_work), -1)], axis=1) if with_quality else base


def ridge_fit(x_train: np.ndarray, t_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-8
    a = (x_train - mean) / std
    b = (x_test - mean) / std
    a = np.concatenate((np.ones((len(a), 1)), a), axis=1)
    b = np.concatenate((np.ones((len(b), 1)), b), axis=1)
    penalty = np.eye(a.shape[1]) * alpha
    penalty[0, 0] = 0
    coef = np.linalg.solve(a.T @ a + penalty, a.T @ t_train)
    return b @ coef


# Nested cross-fitted ridge stacker. Inner folds choose feature set and regularization;
# the outer fold is untouched until that choice has been made.
ridge_z = np.empty_like(true_z)
ridge_choices: dict[str, dict[str, list[object]]] = {}
alpha_grid = [.01, .1, 1, 10, 100, 1000]
for outer in range(5):
    test_outer = folds == outer
    ridge_choices[str(outer)] = {}
    for metric in range(3):
        train_outer = (folds != outer) & valid_y[:, metric]
        best = None
        for with_quality in (False, True):
            x = features(metric, with_quality)
            for alpha in alpha_grid:
                inner_pred = np.empty(np.sum(train_outer), np.float64)
                inner_rows = np.where(train_outer)[0]
                for inner in range(5):
                    if inner == outer:
                        continue
                    fit = train_outer & (folds != inner)
                    valid = (folds == inner) & valid_y[:, metric]
                    positions = np.searchsorted(inner_rows, np.where(valid)[0])
                    inner_pred[positions] = ridge_fit(x[fit], true_z[fit, metric], x[valid], alpha)
                decoded = np.exp(inner_pred * LS[metric] + LM[metric])
                score = metric_one(decoded, inner_rows, metric)
                if best is None or score < best[0]:
                    best = (score, with_quality, alpha, x)
        assert best is not None
        _, with_quality, alpha, x = best
        ridge_choices[str(outer)][str(metric)] = [bool(with_quality), float(alpha)]
        ridge_z[test_outer, metric] = ridge_fit(
            x[train_outer], true_z[train_outer, metric], x[test_outer], alpha
        )
outputs["nested_ridge"] = decode(ridge_z)

# Non-deployable ceiling: choose the tile closest to GT independently per image/metric.
tile_y = np.exp(z * LS[None, None, :] + LM[None, None, :])
oracle = np.empty_like(y)
for metric in range(3):
    error = np.abs(tile_y[:, :, metric] - y[:, None, metric]) / np.maximum(y[:, None, metric], 1e-6)
    choice = np.argmin(error, axis=1)
    oracle[:, metric] = tile_y[np.arange(len(y)), choice, metric]
outputs["oracle_best_tile_ceiling"] = oracle

report = {
    "tag": TAG,
    "n": len(y),
    "invalid_targets_by_metric": (~valid_y).sum(axis=0).tolist(),
    "validation_scope": "Exploratory meta cross-fitting on existing OOF models, not fully nested base-model validation.",
    "methods": {name: {"metrics": metrics(pred), "overall": float(np.mean(metrics(pred)))} for name, pred in outputs.items()},
    "quantile_choices": quantile_choices,
    "focus_choices": focus_choices,
    "ridge_choices": ridge_choices,
}
out = BASE / f"crop_audit_{TAG}.json"
out.write_text(json.dumps(report, indent=2, allow_nan=False))
np.savez_compressed(BASE / f"crop_audit_{TAG}_predictions.npz", ids=np.concatenate([p["ids"] for p in parts]),
                    folds=folds, target=y, **outputs)
print(json.dumps(report, indent=2))
print(f"saved {out}")
