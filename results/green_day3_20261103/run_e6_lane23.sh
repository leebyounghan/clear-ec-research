#!/bin/bash
# E6: E2-CNN area predictions at CPSAM / consensus points (GPUs 2 and 3 split by rows), then CPU evaluation. lane23 job.
cd /workspace/clear-ec-research
PY=/workspace/venv/bin/python; S=experiments/green_day3_20261103/source/green_day3_e6_pred_area_cv_20261103.py; LOG=experiments/green_day3_20261103/e6_pred_area_cv; mkdir -p $LOG
while [ ! -f $LOG/points_consensus_win.npz ]; do sleep 30; done
CUDA_VISIBLE_DEVICES=2 $PY -u $S infer --rows-from 0 --rows-to 4477 > $LOG/infer_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY -u $S infer --rows-from 4477 --rows-to 8954 > $LOG/infer_gpu3.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES="" $PY -u $S evaluate > $LOG/evaluate.log 2>&1 || { echo "FAILED evaluate"; tail -5 $LOG/evaluate.log; exit 1; }
echo "e6 done $(date -u)"
