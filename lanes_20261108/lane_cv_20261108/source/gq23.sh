#!/bin/bash
# CV lane: whole-frame graph (K=0, all cells) on the 3-seed x 4-flip averaged peaks, t0.25, folds 0 and 4 (one try: more cells read)
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2 PREP_WORKERS=2
PY=/workspace/venv/bin/python
date; $PY source/graph_prep_sp_20261108.py tta3_frame_t025.npz out/tta_peaks/seeds3_tta4 0 1296 972 0.25 0
date; $PY source/graph_train_cv_20261108.py tta3_frame_t025 --graphs /workspace/clear-ec-research/experiments/lane_cv_20261108/graphs/tta3_frame_t025.npz --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 3 --no-ctx --folds 04
date; echo QDONE
