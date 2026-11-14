#!/bin/bash
# CV lane: longer graph training on TTA t0.25 nodes: 50 epochs for s46/s48, 80 epochs for s47, folds 0 and 4
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2
PY=/workspace/venv/bin/python
G=/workspace/clear-ec-research/experiments/lane_cv_20261108/graphs
for s in s46 s48; do date; $PY source/graph_train_cv_20261108.py e50_${s} --graphs $G/tta_${s}_t025.npz --d 64 --layers 5 --epochs 50 --lr 1e-3 --seeds 3 --no-ctx --folds 04; done
date; $PY source/graph_train_cv_20261108.py e80_s47 --graphs $G/tta_s47_t025.npz --d 64 --layers 5 --epochs 80 --lr 1e-3 --seeds 3 --no-ctx --folds 04
date; echo QDONE
