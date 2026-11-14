#!/bin/bash
# G5: co-twin training, fold 0 then fold 4, one GPU. Each followed by the runner's audit.
cd /workspace/clear-ec-research
L=experiments/lane_gpu2_20261106; PY=/workspace/venv/bin/python; W=$L/source/recipeA_cotwin_1gpu_20261106.py; DL=2026-11-12T00:00:00Z
export OMP_NUM_THREADS=2
for f in 0 4; do
  if [ $f = 0 ]; then SPLIT=experiments/ssl_screen_20260911/preflight/fold0.npz; else SPLIT=experiments/target_fold4_20260912/preflight/fold4.npz; fi
  R=$L/runs/cotwin_mb4_f${f}_s46
  A="--root $R --split $SPLIT --fold $f --pooling std --seed 46 --ssl-root experiments/phase2_crossfold_20261015/fold$f/ssl5 --head-lr 1e-3 --clip 1 --epochs 20 --schedule-epochs 10 --deadline-utc $DL"
  echo "$(date -u) train f$f start"
  CUDA_VISIBLE_DEVICES=2 $PY $W train $A > $L/logs/cotwin_f$f.log 2>&1; rc=$?
  echo "$(date -u) train f$f rc $rc"
  [ $rc = 0 ] || exit $rc
  CUDA_VISIBLE_DEVICES= $PY $W audit $A > $L/logs/cotwin_f${f}_audit.log 2>&1; echo "$(date -u) audit f$f rc $?"
done
echo "$(date -u) cotwin chain done"
