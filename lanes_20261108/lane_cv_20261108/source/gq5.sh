#!/bin/bash
# GPU 2 queue 5 (CV lane): window attention v3 = simulation-pretrained frozen graph encoder, only attention and head learn (pre48 graphs, folds 0 and 4)
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2 FOLDS=04
PY=/workspace/venv/bin/python
SW=/workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep
date; $PY source/graph_attn_cv_20261108.py attn_sim_frz_p48 --graphs $SW/cell_graphs_pre48_xy.npz --attention --layers 8 --init sim2_l8 --freeze --seeds 3
date; $PY source/graph_attn_cv_20261108.py noattn_sim_frz_p48 --graphs $SW/cell_graphs_pre48_xy.npz --layers 8 --init sim2_l8 --freeze --seeds 3
date; echo QDONE
