#!/bin/bash
# prepare, CPU tests and smoke for the co-twin runs. Training starts separately.
cd /workspace/clear-ec-research
L=experiments/lane_gpu2_20261106; PY=/workspace/venv/bin/python; W=$L/source/recipeA_cotwin_1gpu_20261106.py
DL=2026-11-12T00:00:00Z
export OMP_NUM_THREADS=2
for f in 0 4; do
  if [ $f = 0 ]; then SPLIT=experiments/ssl_screen_20260911/preflight/fold0.npz; else SPLIT=experiments/target_fold4_20260912/preflight/fold4.npz; fi
  A="--root $L/runs/cotwin_mb4_f${f}_s46 --split $SPLIT --fold $f --pooling std --seed 46 --ssl-root experiments/phase2_crossfold_20261015/fold$f/ssl5 --head-lr 1e-3 --clip 1 --epochs 20 --schedule-epochs 10 --deadline-utc $DL"
  [ -f $L/runs/cotwin_mb4_f${f}_s46/manifest.json ] || CUDA_VISIBLE_DEVICES= $PY $W prepare $A || { echo "prepare f$f failed"; exit 1; }
  [ -f $L/runs/cotwin_mb4_f${f}_s46/cpu_tests.json ] || CUDA_VISIBLE_DEVICES= $PY $W cpu_tests $A || { echo "cpu_tests f$f failed"; exit 1; }
  [ -f $L/runs/cotwin_mb4_f${f}_s46/smoke.json ] || CUDA_VISIBLE_DEVICES=2 $PY $W smoke $A || { echo "smoke f$f failed"; exit 1; }
  echo "preflight f$f ok"
done
echo "preflight all ok"
