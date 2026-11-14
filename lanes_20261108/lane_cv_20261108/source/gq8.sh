#!/bin/bash
# CV lane: graphs from flip-TTA / 3-seed-averaged pre-only heatmap peaks (standard nodes t0.35, 3 windows), graph network folds 0 and 4, 3 seeds
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2 PREP_WORKERS=2
PY=/workspace/venv/bin/python
until grep -q TTADONE logs/tta_infer.log; do sleep 60; done
for s in s46_tta4 s47_tta4 s48_tta4 seeds3_tta4 seeds3_single; do date; $PY source/graph_prep_sp_20261108.py tta_$s.npz out/tta_peaks/$s 3 380 300 0.35 0; done
for s in s46_tta4 s47_tta4 s48_tta4 seeds3_tta4 seeds3_single; do date; $PY source/graph_train_cv_20261108.py tta_$s --graphs /workspace/clear-ec-research/experiments/lane_cv_20261108/graphs/tta_$s.npz --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 3 --no-ctx --folds 04; done
date; echo QDONE
