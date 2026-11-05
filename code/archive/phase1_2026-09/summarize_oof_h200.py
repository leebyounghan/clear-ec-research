"""Summarize complete OOF runs and fold-0 screening candidates."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--root", default="experiments/oof_h200")
parser.add_argument("--output", default="experiments/oof_h200/summary.json")
args = parser.parse_args()

root = Path(args.root)
groups: dict[str, list[tuple[Path, np.lib.npyio.NpzFile]]] = defaultdict(list)
for path in sorted(root.glob("*/best_oof.npz")):
    # run names always start TAG_f{fold}_...
    name = path.parent.name
    marker = name.find("_f")
    tag = name[:marker] if marker >= 0 else name
    groups[tag].append((path, np.load(path, allow_pickle=True)))

summary = {}
for tag, entries in sorted(groups.items()):
    indices = np.concatenate([data["indices"] for _, data in entries])
    pred = np.concatenate([data["pred"] for _, data in entries])
    target = np.concatenate([data["target"] for _, data in entries])
    order = np.argsort(indices)
    indices, pred, target = indices[order], pred[order], target[order]
    metric_values = []
    for metric in range(3):
        valid = target[:, metric] > 0
        metric_values.append(
            (np.abs(pred[valid, metric] - target[valid, metric]) / target[valid, metric]).mean() * 100
        )
    metrics = np.asarray(metric_values, dtype=np.float32)
    summary[tag] = {
        "folds": len(entries),
        "images": int(len(indices)),
        "unique_images": int(len(np.unique(indices))),
        "metrics": metrics.tolist(),
        "overall": float(metrics.mean()),
        "runs": [str(path.parent) for path, _ in entries],
    }

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
