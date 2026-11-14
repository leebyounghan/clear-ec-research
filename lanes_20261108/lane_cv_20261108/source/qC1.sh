#!/bin/bash
# CV lane, CPU pod: order-loss graph seeds3_tta4 member, folds 1,2,3 (3 seeds each) then full data (3 seeds); 4 workers x 1 thread.
# Folds 0 and 4 are in graph_runs/gord_seeds3_tta4_04.npz (fold 0 GPU, fold 4 CPU). Merges to graph_runs/gord_seeds3_tta4_5f.npz.
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export PY=/workspace/venv/bin/python
export ARGS="--order 1.0 --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 1 --no-ctx --graphs /workspace/clear-ec-research/experiments/lane_cv_20261108/graphs/tta_seeds3_tta4.npz"
job() { f=$1; k=$2; if [ "$f" = full ]; then n=gord_seeds3_tta4_full_s$k; else n=gord_seeds3_tta4_f${f}_s$k; fi
  if [ -f graph_runs/$n.npz ] || [ -f graph_models/${n}_fullfit_pred.npz ]; then return; fi
  GRAPH_DEV=cpu GRAPH_THREADS=1 OMP_NUM_THREADS=1 $PY source/graph_order_cv_20261110.py $n $ARGS --seed0 $k --folds $f > logs/$n.log 2>&1; echo "$(date -u) $n rc $?"; }
export -f job
echo "$(date -u) start"
printf '%s\n' "1 0" "1 1" "1 2" "2 0" "2 1" "2 2" "3 0" "3 1" "3 2" "full 0" "full 1" "full 2" | xargs -P 4 -L 1 bash -c 'job $0 $1'
$PY -c "
import numpy as np
a = np.load('graph_runs/gord_seeds3_tta4_04.npz'); out = {'f0': a['f0'], 'f4': a['f4']}
for f in (1, 2, 3): out[f'f{f}'] = np.mean([np.load(f'graph_runs/gord_seeds3_tta4_f{f}_s{k}.npz')[f'f{f}'] for k in range(3)], 0).astype(np.float32)
np.savez_compressed('graph_runs/gord_seeds3_tta4_5f.npz', **out); print('merged five folds')"
echo "$(date -u) QC1DONE"
