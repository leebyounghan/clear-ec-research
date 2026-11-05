#!/bin/bash
# E1: CPSAM (cpsam_v2 foundation) segmentation of all 8954 frames, GPUs 2 and 3 (lane23 job). Smoke first.
cd /workspace/clear-ec-research
PY=/workspace/venv/bin/python; S=experiments/green_day3_20261103/source/green_day3_e1_cpsam_20261103.py; LOG=experiments/green_day3_20261103/e1_cpsam; mkdir -p $LOG
CUDA_VISIBLE_DEVICES=2 $PY -u $S segment --model v2 --rows-from 0 --rows-to 8954 --limit 3 > $LOG/smoke.log 2>&1 || { echo "smoke failed"; tail -5 $LOG/smoke.log; exit 1; }
CUDA_VISIBLE_DEVICES=2 $PY -u $S segment --model v2 --rows-from 0 --rows-to 4477 >> $LOG/segment_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY -u $S segment --model v2 --rows-from 4477 --rows-to 8954 >> $LOG/segment_gpu3.log 2>&1 &
wait
ls $LOG/v2/rows_*.npz | wc -l
echo "e1 done $(date -u)"
