"""Diagnostic-only ROI geometry audit on the 25 green reference images."""
import json
from pathlib import Path
import numpy as np

ROOT = Path("datasets/clear_ec_official")
annotations = json.loads((ROOT / "green_eval/annotations.json").read_text())
lookup = {}
for fold in range(5):
    path = Path(f"experiments/oof_h200/baseline_f{fold}_seed42_c256k1_mean_multitask_slidesmooth/crop_tiles.npz")
    with np.load(path, allow_pickle=True) as data:
        ids, targets, z = data["ids"], data["target"], data["z_tile"]
        lm = data["lm"] if "lm" in data else np.array([7.919919967651367, -.9824771285057068, -.6211431622505188])
        ls = data["ls"] if "ls" in data else np.array([.18971863389015198, .18813571333885193, .2748945653438568])
        for i, name in enumerate(ids):
            lookup[str(name).rsplit('.', 1)[0]] = (targets[i], z[i] * ls + lm)

central = (392, 230, 904, 742)
tiles = [(392 + col, 230 + row, 648 + col, 486 + row)
         for row, col in [(0, 0), (0, 256), (256, 0), (256, 256), (128, 128)]]

def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

def intersection(a, b):
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))

def inside(points, box):
    return ((points[:, 0] >= box[0]) & (points[:, 0] < box[2]) &
            (points[:, 1] >= box[1]) & (points[:, 1] < box[3]))

rows, ys, means, roi_weighted, marker_weighted = [], [], [], [], []
for name, annotation in annotations.items():
    target, log_tile = lookup[name]
    recorded_gt = np.array([annotation["gt"][key] for key in ["CD", "CV", "HEX"]])
    np.testing.assert_allclose(target, recorded_gt, rtol=1e-5)
    box = annotation["roi_box_mha"]
    points = np.asarray(annotation["markers_mha"])
    overlaps = np.asarray([intersection(box, tile) for tile in tiles], dtype=float)
    counts = np.asarray([inside(points, tile).sum() for tile in tiles], dtype=float)
    overlap_weights = overlaps / overlaps.sum() if overlaps.sum() else np.full(5, .2)
    marker_weights = counts / counts.sum() if counts.sum() else np.full(5, .2)
    ys.append(target); means.append(np.exp(log_tile.mean(0)))
    roi_weighted.append(np.exp(overlap_weights @ log_tile))
    marker_weighted.append(np.exp(marker_weights @ log_tile))
    rows.append(dict(id=name, roi_box_fraction_in_central512=intersection(box, central) / area(box),
                     marker_fraction_in_central512=float(inside(points, central).mean()),
                     roi_center_in_central512=bool(inside(np.array([[(box[0]+box[2])/2, (box[1]+box[3])/2]]), central)[0]),
                     markers=len(points), markers_per_tile=counts.tolist()))
y = np.asarray(ys)
report = dict(n=len(rows), role="Diagnostic only. ROI coordinates/markers are unavailable at deployment and not used for model training.",
              original_scalar_labels_match=True,
              mean_roi_box_coverage=float(np.mean([r["roi_box_fraction_in_central512"] for r in rows])),
              mean_marker_coverage=float(np.mean([r["marker_fraction_in_central512"] for r in rows])),
              roi_centers_outside=sum(not r["roi_center_in_central512"] for r in rows),
              metrics={name: (np.mean(np.abs(np.asarray(pred) - y) / y, axis=0) * 100).tolist()
                       for name, pred in [("equal_crops", means), ("known_roi_overlap_weights_not_deployable", roi_weighted),
                                          ("known_marker_weights_not_deployable", marker_weighted)]}, rows=rows)
out = Path("experiments/followup_20260902/green25_geometry.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, allow_nan=False))
print(json.dumps({k:v for k,v in report.items() if k != "rows"}, indent=2))
