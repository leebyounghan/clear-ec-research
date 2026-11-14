#!/bin/bash
# CV lane: per-detector TTA t0.25 graph members on folds 1-3, merge to five folds
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2
PY=/workspace/venv/bin/python
for s in s46 s47 s48; do date; $PY source/graph_train_cv_20261108.py tta_${s}_t025_f123 --graphs /workspace/clear-ec-research/experiments/lane_cv_20261108/graphs/tta_${s}_t025.npz --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 3 --no-ctx --folds 123; done
$PY -c "
import numpy as np
for s in (\"s46\", \"s47\", \"s48\"):
    a = np.load(f\"graph_runs/tta_{s}_t025.npz\"); b = np.load(f\"graph_runs/tta_{s}_t025_f123.npz\")
    np.savez_compressed(f\"graph_runs/tta_{s}_t025_5f.npz\", **{f\"f{f}\": (a if f in (0, 4) else b)[f\"f{f}\"] for f in range(5)}); print(\"merged\", s)"
date; echo QDONE
