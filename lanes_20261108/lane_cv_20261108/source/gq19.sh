#!/bin/bash
# CV lane: control for the longer-training gain: shipped single-view pre48 graphs at 50 epochs, folds 0 and 4
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2
PY=/workspace/venv/bin/python
until grep -q QDONE logs/gq18.log; do sleep 60; done
date; $PY source/graph_train_cv_20261108.py e50_raw_p48 --graphs /workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep/cell_graphs_pre48.npz --d 64 --layers 5 --epochs 50 --lr 1e-3 --seeds 3 --no-ctx --folds 04
date; echo QDONE
