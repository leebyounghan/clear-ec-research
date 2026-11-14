#!/bin/bash
# CV lane: node selection variants on the 3-seed x 4-flip averaged peaks (lower threshold, spacing prior), folds 0 and 4, 3 seeds
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2 PREP_WORKERS=2
PY=/workspace/venv/bin/python
F=out/tta_peaks/seeds3_tta4
date; $PY source/graph_prep_sp_20261108.py tta3_t025.npz $F 3 380 300 0.25 0
date; $PY source/graph_prep_sp_20261108.py tta3_t045.npz $F 3 380 300 0.45 0
date; $PY source/graph_prep_sp_20261108.py tta3_sp07_t02.npz $F 3 380 300 0.2 0.7
for g in tta3_t025 tta3_t045 tta3_sp07_t02; do date; $PY source/graph_train_cv_20261108.py $g --graphs /workspace/clear-ec-research/experiments/lane_cv_20261108/graphs/$g.npz --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 3 --no-ctx --folds 04; done
date; echo QDONE
