#!/bin/bash
# CV lane: lower node thresholds on TTA peaks (t0.20 on 3-seed average; t0.25 on each detector), folds 0 and 4
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2 PREP_WORKERS=2
PY=/workspace/venv/bin/python
date; $PY source/graph_prep_sp_20261108.py tta3_t020.npz out/tta_peaks/seeds3_tta4 3 380 300 0.20 0
for s in s46 s47 s48; do date; $PY source/graph_prep_sp_20261108.py tta_${s}_t025.npz out/tta_peaks/${s}_tta4 3 380 300 0.25 0; done
for g in tta3_t020 tta_s46_t025 tta_s47_t025 tta_s48_t025; do date; $PY source/graph_train_cv_20261108.py $g --graphs /workspace/clear-ec-research/experiments/lane_cv_20261108/graphs/$g.npz --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 3 --no-ctx --folds 04; done
date; echo QDONE
