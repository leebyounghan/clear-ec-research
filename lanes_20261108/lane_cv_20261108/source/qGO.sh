#!/bin/bash
# CV lane, GPU 2 (beside the SD member): order-loss versions of the best pipeline's GRAPH ensemble sets (cell_graphs noctx x10 seeds, pre47 x3, pre48 x3),
# five folds, same recipe as graph_sweep/d64_l5_* plus 1.0 x the noise-weighted pairwise order loss on CV (as the order quartet). Three processes, one per set.
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2
PY=/workspace/venv/bin/python; A="--order 1.0 --d 64 --layers 5 --epochs 25 --lr 1e-3 --wd 0.01 --drop 0.1 --no-ctx --folds 01234"
echo "$(date -u) start"
$PY source/graph_order_cv_20261110.py gordG_noctx_5f --graphs cell_graphs.npz --seeds 10 $A > logs/gordG_noctx_5f.log 2>&1 &
$PY source/graph_order_cv_20261110.py gordG_pre47_5f --graphs cell_graphs_pre47.npz --seeds 3 $A > logs/gordG_pre47_5f.log 2>&1 &
$PY source/graph_order_cv_20261110.py gordG_pre48_5f --graphs cell_graphs_pre48.npz --seeds 3 $A > logs/gordG_pre48_5f.log 2>&1 &
wait; echo "$(date -u) QGODONE"
