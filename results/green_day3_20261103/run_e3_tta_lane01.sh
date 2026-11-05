#!/bin/bash
# E3 TTA4 predictions for folds 0 and 1 (GPU 0 / GPU 1 in parallel), then the comparison (CPU). lane01 job.
cd /workspace/clear-ec-research
PY=/workspace/venv/bin/python; S=experiments/green_day3_20261103/source/green_day3_e3_tta_20261103.py; LOG=experiments/green_day3_20261103/e3_boxpool; mkdir -p $LOG
CUDA_VISIBLE_DEVICES=0 $PY -u $S --fold 0 > $LOG/tta_fold0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY -u $S --fold 1 > $LOG/tta_fold1.log 2>&1 &
wait
for f in 0 1; do CUDA_VISIBLE_DEVICES="" $PY -u experiments/green_day3_20261103/source/green_day3_e3_compare_20261103.py --fold $f > $LOG/compare_tta_fold$f.log 2>&1; done
echo "e3 tta done $(date -u)"
