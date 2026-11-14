#!/bin/bash
# CV lane: full-data graph networks of the TTA quartet (same settings as the OOF members), weights saved to graph_models/
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2
PY=/workspace/venv/bin/python
for g in tta_s46_tta4 tta_s47_tta4 tta_s48_tta4 tta_seeds3_tta4; do
  T0=$(date +%s); $PY source/graph_train_cv_20261108.py full_$g --graphs /workspace/clear-ec-research/experiments/lane_cv_20261108/graphs/$g.npz --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 3 --no-ctx --folds full; echo "$(date -u) full_$g rc $? seconds $(( $(date +%s) - T0 ))"
done
echo QGDONE
