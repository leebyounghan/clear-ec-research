#!/bin/bash
# E2: learned per-cell geometry head: labels (CPU) -> train (GPU 2) -> infer (GPU 2: sg2_s46, GPU 3: sg2pre_s46) -> evaluate (CPU). lane23 job.
cd /workspace/clear-ec-research
PY=/workspace/venv/bin/python; S=experiments/green_day3_20261103/source/green_day3_e2_cellhead_20261103.py; LOG=experiments/green_day3_20261103/e2_cellhead; mkdir -p $LOG
[ -f $LOG/labels.npz ] || CUDA_VISIBLE_DEVICES="" $PY -u $S labels > $LOG/labels.log 2>&1 || { echo "FAILED labels"; tail -5 $LOG/labels.log; exit 1; }
[ -f $LOG/model.pt ] || CUDA_VISIBLE_DEVICES=2 $PY -u $S train --epochs 30 > $LOG/train.log 2>&1 || { echo "FAILED train"; tail -5 $LOG/train.log; exit 1; }
CUDA_VISIBLE_DEVICES=2 $PY -u $S infer --families sg2_s46 > $LOG/infer_sg2_s46.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY -u $S infer --families sg2pre_s46 > $LOG/infer_sg2pre_s46.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 $PY -u $S evaluate > $LOG/evaluate.log 2>&1 || { echo "FAILED evaluate"; tail -5 $LOG/evaluate.log; exit 1; }
echo "e2 done $(date -u)"
