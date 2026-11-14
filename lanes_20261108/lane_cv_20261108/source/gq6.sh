#!/bin/bash
# GPU 2 queue 6 (CV lane): sim-pretrained graph encoder members on the other two pre-only detectors (frozen, and fine-tuned from the sim init), folds 0 and 4
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2 FOLDS=04
PY=/workspace/venv/bin/python
SW=/workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep
date; $PY source/graph_attn_cv_20261108.py noattn_sim_frz_s46 --graphs $SW/cell_graphs_xy.npz --layers 8 --init sim2_l8 --freeze --seeds 3
date; $PY source/graph_attn_cv_20261108.py noattn_sim_frz_p47 --graphs $SW/cell_graphs_pre47_xy.npz --layers 8 --init sim2_l8 --freeze --seeds 3
date; $PY source/graph_attn_cv_20261108.py noattn_sim_ft_p48 --graphs $SW/cell_graphs_pre48_xy.npz --layers 8 --init sim2_l8 --seeds 3
date; echo QDONE
