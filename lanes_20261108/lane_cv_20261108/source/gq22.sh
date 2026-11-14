#!/bin/bash
# CV lane: shipped single-view graph trio (noctx/pre47/pre48) at 50 epochs, five folds
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2
PY=/workspace/venv/bin/python
SW=/workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep
until grep -q QDONE logs/gq20.log; do sleep 60; done
for g in cell_graphs cell_graphs_pre47 cell_graphs_pre48; do date; $PY source/graph_train_cv_20261108.py e50raw5f_$g --graphs $SW/$g.npz --d 64 --layers 5 --epochs 50 --lr 1e-3 --seeds 3 --no-ctx --folds 01234; done
date; echo QDONE
