#!/bin/bash
# G2: frame-augmented srep pre-training (seed 46, 2000 updates) + TTA peaks, folds 0-4, on GPUs 2 and 3 (lane23 job).
cd /workspace/clear-ec-research
PY=/workspace/venv/bin/python; S=experiments/green_day2_20261101/source/green_day2_fullframe_20261101.py
LOG=experiments/green_day2_20261101/sg2pre_aug; mkdir -p $LOG
run_fold() { gpu=$1; shift; for f in "$@"; do
  if [ ! -f $LOG/fold$f/sg2_pre.pt ]; then CUDA_VISIBLE_DEVICES=$gpu $PY -u $S pretrain-aug --fold $f --seed 46 --updates 2000 >> $LOG/pretrain_gpu$gpu.log 2>&1 || { echo "FAILED pretrain fold $f" >> $LOG/pretrain_gpu$gpu.log; continue; }; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $S infer-pre-tta --family aug_s46 --fold $f >> $LOG/infer_gpu$gpu.log 2>&1 || echo "FAILED infer fold $f" >> $LOG/infer_gpu$gpu.log
done; }
run_fold 2 0 2 4 &
run_fold 3 1 3 &
wait
ls experiments/green_day2_20261101/preonly_tta/aug_s46/fold*/fold*_sg2_peaks.npz | wc -l
echo "g2 done $(date -u)"
