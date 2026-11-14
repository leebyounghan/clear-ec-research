#!/bin/bash
# CV lane, CPU pod: all-rows fits of the best pipeline's GRAPH ensemble (deploy), same recipe as meta_ceiling sweep/graph_sweep/{d64_l5_noctx_s10, d64_l5_pre47_s3, d64_l5_pre48_s3}:
# d64, 5 layers, 25 epochs, lr 1e-3, wd 0.01, drop 0.1, no ctx, 9 node features; seeds 0-9 / 0-2 / 0-2. graph_order_cv_20261110.py with --order 0 is the same network and loss.
# Weights: graph_models/gfull_<set>_s<k>_s<k>_e{21..25}.pt; in-sample all-row predictions: graph_models/gfull_<set>_s<k>_fullfit_pred.npz.
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
PY=/workspace/venv/bin/python
ARGS="--order 0 --d 64 --layers 5 --epochs 25 --lr 1e-3 --wd 0.01 --drop 0.1 --seeds 1 --no-ctx --folds full"
mkdir -p locks/qC4; JOBS=()
for k in 0 1 2 3 4 5 6 7 8 9; do JOBS+=("noctx cell_graphs.npz $k"); done
for k in 0 1 2; do JOBS+=("pre47 cell_graphs_pre47.npz $k" "pre48 cell_graphs_pre48.npz $k"); done
worker() { for j in "${JOBS[@]}"; do set -- $j; s=$1; g=$2; k=$3; n=gfull_${s}_s$k
    mkdir locks/qC4/$n 2>/dev/null || continue
    GRAPH_DEV=cpu GRAPH_THREADS=1 OMP_NUM_THREADS=1 $PY source/graph_order_cv_20261110.py $n --graphs $g $ARGS --seed0 $k > logs/$n.log 2>&1; rc=$?
    echo "$(date -u) $n rc $rc"; done; }
echo "$(date -u) start"; for i in $(seq ${1:-16}); do worker & done; wait; echo "$(date -u) QC4DONE"
