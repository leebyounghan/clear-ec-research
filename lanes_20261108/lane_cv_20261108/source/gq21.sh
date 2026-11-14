#!/bin/bash
# CV lane: 40-epoch TTA t0.25 graph trio (between 25 and 50), folds 0 and 4
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2
PY=/workspace/venv/bin/python
G=/workspace/clear-ec-research/experiments/lane_cv_20261108/graphs
until grep -q QDONE logs/gq19.log; do sleep 60; done
for s in s46 s47 s48; do date; $PY source/graph_train_cv_20261108.py e40_${s} --graphs $G/tta_${s}_t025.npz --d 64 --layers 5 --epochs 40 --lr 1e-3 --seeds 3 --no-ctx --folds 04; done
date; echo QDONE
