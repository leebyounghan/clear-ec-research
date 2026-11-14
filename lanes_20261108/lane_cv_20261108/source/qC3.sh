#!/bin/bash
# CV lane, CPU pod: rerun the order-loss graph full-data fits for seeds 1 and 2 (all four sets). The first runs ignored --seed0 for --folds full
# and trained seed 0 three times (fixed in graph_order_cv_20261110.py; wrong outputs in graph_models/_archive_wrong_seed_20261110). 8 workers x 1 thread.
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
PY=/workspace/venv/bin/python; G=/workspace/clear-ec-research/experiments/lane_cv_20261108/graphs
ARGS="--order 1.0 --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 1 --no-ctx"
mkdir -p locks/qC3; JOBS=()
for s in seeds3 s46 s47 s48; do for k in 1 2; do JOBS+=("$s $k"); done; done
worker() { for j in "${JOBS[@]}"; do set -- $j; s=$1; k=$2; n=gord_${s}_tta4_full_s$k
    mkdir locks/qC3/$n 2>/dev/null || continue
    GRAPH_DEV=cpu GRAPH_THREADS=1 OMP_NUM_THREADS=1 $PY source/graph_order_cv_20261110.py $n --graphs $G/tta_${s}_tta4.npz $ARGS --seed0 $k --folds full > logs/$n.log 2>&1; rc=$?
    echo "$(date -u) $n rc $rc"; done; }
echo "$(date -u) start"; for i in $(seq ${1:-8}); do worker & done; wait; echo "$(date -u) QC3DONE"
