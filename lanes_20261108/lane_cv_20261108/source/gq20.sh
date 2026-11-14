#!/bin/bash
# CV lane: 50-epoch TTA t0.25 graph members on folds 1-3, merge to five folds
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2
PY=/workspace/venv/bin/python
G=/workspace/clear-ec-research/experiments/lane_cv_20261108/graphs
for s in s46 s47 s48; do date; $PY source/graph_train_cv_20261108.py e50_${s}_f123 --graphs $G/tta_${s}_t025.npz --d 64 --layers 5 --epochs 50 --lr 1e-3 --seeds 3 --no-ctx --folds 123; done
$PY -c "
import numpy as np
for s, a0 in ((\"s46\", \"e50_s46\"), (\"s47\", \"cap_d64_l5_e50_s47\"), (\"s48\", \"e50_s48\")):
    a = np.load(f\"graph_runs/{a0}.npz\"); b = np.load(f\"graph_runs/e50_{s}_f123.npz\")
    np.savez_compressed(f\"graph_runs/e50_{s}_5f.npz\", **{f\"f{f}\": (a if f in (0, 4) else b)[f\"f{f}\"] for f in range(5)}); print(\"merged\", s)"
date; echo QDONE
