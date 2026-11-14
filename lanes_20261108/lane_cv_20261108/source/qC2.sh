#!/bin/bash
# CV lane, CPU pod: order-loss graph members for the s46/s47/s48 sets, one seed per job, 1 thread per job.
# Job order: folds 0,4 (screen) -> folds 1,2,3 -> full data. Pool of workers sharing one job list via mkdir locks,
# so more workers can join later: bash source/qC2.sh N   (N workers, default 12).
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
PY=/workspace/venv/bin/python; G=/workspace/clear-ec-research/experiments/lane_cv_20261108/graphs
ARGS="--order 1.0 --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 1 --no-ctx"
mkdir -p locks/qC2
JOBS=()
for f in 0 4 1 2 3 full; do for s in s46 s47 s48; do for k in 0 1 2; do JOBS+=("$s $f $k"); done; done; done
worker() {
  for j in "${JOBS[@]}"; do set -- $j; s=$1; f=$2; k=$3
    if [ "$f" = full ]; then n=gord_${s}_tta4_full_s$k; else n=gord_${s}_tta4_f${f}_s$k; fi
    mkdir locks/qC2/$n 2>/dev/null || continue
    GRAPH_DEV=cpu GRAPH_THREADS=1 OMP_NUM_THREADS=1 $PY source/graph_order_cv_20261110.py $n --graphs $G/tta_${s}_tta4.npz $ARGS --seed0 $k --folds $f > logs/$n.log 2>&1
    echo "$(date -u) $n rc $?"
  done
}
N=${1:-12}; echo "$(date -u) start $N workers"
for i in $(seq $N); do worker & done; wait
echo "$(date -u) QC2WORKERS_DONE ($N)"
