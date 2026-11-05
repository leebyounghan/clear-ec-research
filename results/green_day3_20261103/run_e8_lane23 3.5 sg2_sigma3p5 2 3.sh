#!/bin/bash
# E8 (lane23 3.5 sg2_sigma3p5 2 3): stride-2 specialist with target sigma  px (tag ), folds 0-4 on GPUs  and : train then infer per fold.
cd /workspace/clear-ec-research
PY=/workspace/venv/bin/python; S=experiments/green_day3_20261103/source/green_day3_e8_sigma_20261103.py; LOG=experiments/green_day3_20261103/e8_sigma; mkdir -p $LOG
run_folds() { gpu=$1; shift; for f in "$@"; do
  if [ ! -f experiments/green_track_20261028//fold$f/specialist_sg2.pt ]; then CUDA_VISIBLE_DEVICES=$gpu $PY -u $S  train --fold $f --tag  >> $LOG/_gpu$gpu.log 2>&1 || { echo "FAILED train $f" >> $LOG/_gpu$gpu.log; continue; }; fi
  if [ ! -f experiments/green_track_20261028//fold$f/fold${f}_sg2_peaks.npz ]; then CUDA_VISIBLE_DEVICES=$gpu $PY -u $S  infer --fold $f --tag  >> $LOG/_gpu$gpu.log 2>&1 || echo "FAILED infer $f" >> $LOG/_gpu$gpu.log; fi
done; }
run_folds  0 2 4 &
run_folds  1 3 &
wait
ls experiments/green_track_20261028//fold*/fold*_sg2_peaks.npz | wc -l
echo "e8  done $(date -u)"
