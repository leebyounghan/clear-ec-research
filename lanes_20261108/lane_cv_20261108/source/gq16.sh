#!/bin/bash
# CV lane: graph capacity on the cleaner TTA t0.25 nodes (s47 detector), folds 0 and 4; reference = tta_s47_t025 (d64, 5 layers, 25 epochs)
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2
PY=/workspace/venv/bin/python
G=/workspace/clear-ec-research/experiments/lane_cv_20261108/graphs/tta_s47_t025.npz
until grep -q QDONE logs/gq15.log; do sleep 60; done
date; $PY source/graph_train_cv_20261108.py cap_d96_l6_e40_s47 --graphs $G --d 96 --layers 6 --epochs 40 --lr 1e-3 --seeds 3 --no-ctx --folds 04
date; $PY source/graph_train_cv_20261108.py cap_d64_l5_e50_s47 --graphs $G --d 64 --layers 5 --epochs 50 --lr 1e-3 --seeds 3 --no-ctx --folds 04
date; echo QDONE
