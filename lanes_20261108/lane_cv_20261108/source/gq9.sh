#!/bin/bash
# CV lane: TTA-peak graph members on folds 1-3 (to make them five-fold), then merge with folds 0,4
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2
PY=/workspace/venv/bin/python
until grep -q QDONE logs/gq8.log; do sleep 60; done
for s in s46_tta4 s47_tta4 s48_tta4; do date; $PY source/graph_train_cv_20261108.py tta_${s}_f123 --graphs /workspace/clear-ec-research/experiments/lane_cv_20261108/graphs/tta_$s.npz --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 3 --no-ctx --folds 123; done
$PY - <<PYEOF
import numpy as np
for s in ("s46_tta4", "s47_tta4", "s48_tta4"):
    a = np.load(f"graph_runs/tta_{s}.npz"); b = np.load(f"graph_runs/tta_{s}_f123.npz")
    np.savez_compressed(f"graph_runs/tta_{s}_5f.npz", **{f"f{f}": (a if f in (0, 4) else b)[f"f{f}"] for f in range(5)}); print("merged", s)
PYEOF
date; echo QDONE
