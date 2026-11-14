#!/bin/bash
# GPU 2 queue 4 (CV lane): window attention over the legible windows (never run before), folds 0 and 4, 3 seeds; control = cvg48_ctrl (same graphs, no attention)
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2 FOLDS=04
PY=/workspace/venv/bin/python
SW=/workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep
date; $PY source/graph_attn_cv_20261108.py attn_k3_p48 --graphs $SW/cell_graphs_pre48_xy.npz --attention --layers 5 --seeds 3
date; $PY source/graph_attn_cv_20261108.py attn_k5_s46 --graphs /workspace/clear-ec-research/experiments/lane_cv_20261108/graphs/k5_t035.npz --attention --layers 5 --seeds 3
date; echo QDONE
