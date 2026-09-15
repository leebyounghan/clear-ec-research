#!/usr/bin/env python3
"""Validate crop postprocessing with distinct train/calibration/test slide groups.

Five rotating splits use 60% base training, 20% calibration and 20% test.
All models use a predeclared last epoch. Test labels do not fit normalization,
base weights, calibration or ridge. These are not historically unseen data.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path("experiments/followup_20260902")


def weighted_scale(pred, target):
    ratios = target / pred
    weights = pred / target
    order = np.argsort(ratios)
    return float(ratios[order][np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)])


def ridge(x, target, test, alpha=100.0):
    mean, scale = x.mean(0), x.std(0) + 1e-8
    x = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    test = np.column_stack([np.ones(len(test)), (test - mean) / scale])
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    return test @ np.linalg.solve(x.T @ x + penalty, x.T @ target)


def features(log_tiles):
    return np.column_stack([log_tiles, np.sort(log_tiles, axis=1), log_tiles.mean(1), log_tiles.std(1)])


parts = []
for outer in range(5):
    calibration_fold = (outer + 1) % 5
    run = ROOT / "runs" / f"nested_o{outer}_f{calibration_fold}_seed42_c256k1_mean_multitask_slidesmooth"
    result = json.loads((run / "result.json").read_text())
    assert result["select_mode"] == "last" and result["exclude_fold"] == outer
    split = np.load(run / "split_indices.npz")
    cal = np.load(run / "crop_tiles.npz", allow_pickle=True)
    test = np.load(run / "crop_holdout.npz", allow_pickle=True)
    assert np.array_equal(cal["indices"], split["validation"])
    assert np.array_equal(test["indices"], split["holdout"])
    assert not np.intersect1d(split["train"], test["indices"]).size
    assert not np.intersect1d(cal["indices"], test["indices"]).size
    lc = cal["z_tile"].astype(float) * cal["ls"] + cal["lm"]
    lt = test["z_tile"].astype(float) * test["ls"] + test["lm"]
    yc, yt = cal["target"].astype(float), test["target"].astype(float)
    predictions = {key: np.empty_like(yt) for key in ("mean", "scaled_mean", "quantile", "ridge100")}
    choices = []
    for m in range(3):
        valid = (yc[:, m] > 0) & np.isfinite(yc[:, m])
        target = yc[valid, m]
        base_cal = np.exp(lc[valid, :, m].mean(1))
        base_test = np.exp(lt[:, :, m].mean(1))
        predictions["mean"][:, m] = base_test
        scale = weighted_scale(base_cal, target)
        predictions["scaled_mean"][:, m] = base_test * scale
        grid = np.linspace(0, 1, 41)
        scores = [np.mean(np.abs(np.exp(np.quantile(lc[valid, :, m], q, axis=1)) - target) / target) for q in grid]
        q = float(grid[np.argmin(scores)])
        predictions["quantile"][:, m] = np.exp(np.quantile(lt[:, :, m], q, axis=1))
        predictions["ridge100"][:, m] = np.exp(ridge(features(lc[valid, :, m]), np.log(target), features(lt[:, :, m])))
        choices.append(dict(metric=m, quantile=q, scale=scale))
    assert all(np.isfinite(p).all() for p in predictions.values())
    parts.append(dict(indices=test["indices"], ids=test["ids"], target=yt, predictions=predictions,
                      fold=np.full(len(yt), outer), choices=choices))

indices = np.concatenate([p["indices"] for p in parts])
assert len(np.unique(indices)) == len(indices)
y = np.concatenate([p["target"] for p in parts])
ids = np.concatenate([p["ids"] for p in parts])
folds = np.concatenate([p["fold"] for p in parts])
valid = (y > 0) & np.isfinite(y)
preds = {name: np.concatenate([p["predictions"][name] for p in parts]) for name in parts[0]["predictions"]}
errors = {name: np.where(valid, np.abs(pred - y) / np.where(valid, y, 1), 0) * 100 for name, pred in preds.items()}

def clean(value):
    return str(value).rsplit('.', 1)[0] if '.' in str(value) else str(value)

with open("datasets/clear_ec_official/final_train_ids.csv", newline="") as stream:
    slides = {clean(row["ID"]): row["slide_id"] for row in csv.DictReader(stream)}
_, group = np.unique([slides.get(clean(i), f"missing:{i}") for i in ids], return_inverse=True)
count = int(group.max() + 1)
denom = np.zeros((count, 3)); np.add.at(denom, group, valid)
deltas = {}
for name, err in errors.items():
    grouped = np.zeros((count, 3)); np.add.at(grouped, group, err - errors["mean"])
    deltas[name] = grouped
rng = np.random.default_rng(92026)
samples = {name: [] for name in preds}
for _ in range(2000):
    draw = rng.integers(0, count, count)
    divisor = denom[draw].sum(0)
    for name in preds:
        samples[name].append(float(np.mean(deltas[name][draw].sum(0) / divisor)))

methods = {}
for name, err in errors.items():
    metric = err.sum(0) / valid.sum(0)
    methods[name] = dict(metrics=metric.tolist(), overall=float(metric.mean()),
                         paired_slide_bootstrap_delta_ci95=np.quantile(samples[name], [.025, .975]).tolist(),
                         per_fold=[float((err[folds == f].sum(0) / valid[folds == f].sum(0)).mean()) for f in range(5)])
report = dict(protocol="Five fixed-epoch train60/calibration20/test20 slide-disjoint rotations. Historically reused dataset; not a pristine final test.",
              n=len(y), invalid_by_metric=(~valid).sum(0).tolist(), methods=methods,
              choices=[p["choices"] for p in parts])
path = ROOT / "nested_holdout_report.json"
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(report, indent=2, allow_nan=False)); temporary.replace(path)
np.savez_compressed(ROOT / "nested_holdout_predictions.npz", indices=indices, ids=ids, target=y, folds=folds, **preds)
print(json.dumps(report, indent=2, allow_nan=False))
