"""Search robust per-metric OOF blends and multiplicative calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--root", default="experiments/oof_h200")
parser.add_argument("--tags", nargs="+", default=["baseline", "attn"])
parser.add_argument("--output", default="experiments/oof_h200/ensemble_analysis.json")
args = parser.parse_args()


def load_tag(tag: str):
    entries = []
    for path in sorted(Path(args.root).glob(f"{tag}_f*/best_oof.npz")):
        data = np.load(path, allow_pickle=True)
        entries.append((data["indices"], data["pred"], data["target"]))
    if not entries:
        raise FileNotFoundError(f"no OOF predictions for tag={tag}")
    indices = np.concatenate([entry[0] for entry in entries])
    pred = np.concatenate([entry[1] for entry in entries])
    target = np.concatenate([entry[2] for entry in entries])
    order = np.argsort(indices)
    return indices[order], pred[order], target[order], len(entries)


loaded = {tag: load_tag(tag) for tag in args.tags}
reference_indices, _, reference_target, _ = loaded[args.tags[0]]
for tag, (indices, _, target, _) in loaded.items():
    if not np.array_equal(indices, reference_indices):
        raise ValueError(f"OOF indices for {tag} do not align")
    if not np.allclose(target, reference_target):
        raise ValueError(f"OOF targets for {tag} do not align")


def mape(pred: np.ndarray, target: np.ndarray, metric: int) -> float:
    valid = target[:, metric] > 0
    return float(np.mean(np.abs(pred[valid, metric] - target[valid, metric]) / target[valid, metric]) * 100)


predictions = {tag: loaded[tag][1] for tag in args.tags}
report = {
    "tags": args.tags,
    "folds": {tag: loaded[tag][3] for tag in args.tags},
    "members": {},
    "blend": {},
}
for tag, pred in predictions.items():
    metrics = [mape(pred, reference_target, metric) for metric in range(3)]
    report["members"][tag] = {"metrics": metrics, "overall": float(np.mean(metrics))}

if len(args.tags) == 2:
    first, second = args.tags
    for metric, name in enumerate(("CD", "CV", "HEX")):
        best = None
        valid = reference_target[:, metric] > 0
        target_metric = reference_target[valid, metric]
        # weight is assigned to the first member; calibration is multiplicative.
        for weight in np.linspace(0.0, 1.0, 101):
            blended = (
                weight * predictions[first][valid, metric]
                + (1.0 - weight) * predictions[second][valid, metric]
            )
            for calibration in np.linspace(0.97, 1.03, 121):
                score = float(np.mean(np.abs(blended * calibration - target_metric) / target_metric) * 100)
                if best is None or score < best["mape"]:
                    best = {
                        "weight_first": float(weight),
                        "weight_second": float(1.0 - weight),
                        "calibration": float(calibration),
                        "mape": score,
                    }
        report["blend"][name] = best
    report["blend_overall"] = float(np.mean([value["mape"] for value in report["blend"].values()]))

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
