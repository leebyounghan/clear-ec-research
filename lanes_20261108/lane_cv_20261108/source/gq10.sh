#!/bin/bash
# CV lane: seeds3_tta4 graph member on folds 1-3, merge to five folds
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2
PY=/workspace/venv/bin/python
until grep -q QDONE logs/gq9.log; do sleep 60; done
date; $PY source/graph_train_cv_20261108.py tta_seeds3_tta4_f123 --graphs /workspace/clear-ec-research/experiments/lane_cv_20261108/graphs/tta_seeds3_tta4.npz --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 3 --no-ctx --folds 123
$PY -c "
import numpy as np
a = np.load(\"graph_runs/tta_seeds3_tta4.npz\"); b = np.load(\"graph_runs/tta_seeds3_tta4_f123.npz\")
np.savez_compressed(\"graph_runs/tta_seeds3_tta4_5f.npz\", **{f\"f{f}\": (a if f in (0, 4) else b)[f\"f{f}\"] for f in range(5)}); print(\"merged\")"
date; echo QDONE
