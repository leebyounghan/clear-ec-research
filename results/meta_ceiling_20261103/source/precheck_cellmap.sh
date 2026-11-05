#!/bin/bash
# Prepare and CPU-test the cell-map root ahead of the queue, so a broken wrapper shows up before GPUs are waiting on it.
cd /workspace/clear-ec-research
PY=/workspace/venv/bin/python; SRC=experiments/meta_ceiling_20261103/source; F=0
SPLIT=experiments/ssl_screen_20260911/preflight/fold0.npz; SSL=experiments/phase2_crossfold_20261015/fold0/ssl5
for pair in cellmap:recipeA_cellmap_20261105.py; do
  name=${pair%%:*}; W=$SRC/${pair##*:}; ROOT=experiments/meta_ceiling_20261103/$name/f0_s46; mkdir -p $ROOT; LOG=$ROOT/driver.log
  ARGS="--root $ROOT --split $SPLIT --fold $F --pooling std --seed 46 --ssl-root $SSL --head-lr 0.001 --clip 1 --epochs 20 --schedule-epochs 10 --deadline-utc 2026-11-11T12:00:00Z"
  echo "### prepare $name $(date -u)" >> $LOG; env CUDA_VISIBLE_DEVICES="" $PY $W prepare $ARGS >> $LOG 2>&1 || { echo "FAILED prepare $name"; continue; }
  echo "### cpu_tests $name $(date -u)" >> $LOG; env CUDA_VISIBLE_DEVICES="" $PY $W cpu_tests $ARGS >> $LOG 2>&1 || { echo "FAILED cpu_tests $name"; continue; }
  echo "passed $name"
done
echo precheck finished
