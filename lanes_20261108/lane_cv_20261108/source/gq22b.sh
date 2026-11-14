#!/bin/bash
# CV lane: 40-epoch TTA t0.25 graph members on folds 1-3 (merge to five folds), then the shipped single-view trio at 40 epochs on five folds
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2
PY=/workspace/venv/bin/python
G=/workspace/clear-ec-research/experiments/lane_cv_20261108/graphs
SW=/workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep
until grep -q QDONE logs/gq20.log; do sleep 60; done
for s in s46 s47 s48; do date; $PY source/graph_train_cv_20261108.py e40_${s}_f123 --graphs $G/tta_${s}_t025.npz --d 64 --layers 5 --epochs 40 --lr 1e-3 --seeds 3 --no-ctx --folds 123; done
$PY -c "
import numpy as np
for s in (\"s46\", \"s47\", \"s48\"):
    a = np.load(f\"graph_runs/e40_{s}.npz\"); b = np.load(f\"graph_runs/e40_{s}_f123.npz\")
    np.savez_compressed(f\"graph_runs/e40_{s}_5f.npz\", **{f\"f{f}\": (a if f in (0, 4) else b)[f\"f{f}\"] for f in range(5)}); print(\"merged\", s)"
echo E40MERGED
for g in cell_graphs cell_graphs_pre47 cell_graphs_pre48; do date; $PY source/graph_train_cv_20261108.py e40raw5f_$g --graphs $SW/$g.npz --d 64 --layers 5 --epochs 40 --lr 1e-3 --seeds 3 --no-ctx --folds 01234; done
date; echo QDONE
