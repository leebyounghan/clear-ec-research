#!/bin/bash
# E11b SREP + synthetic v5 mixed training (lane23, GPU 2), then score25.
cd /workspace/clear-ec-research
PY=/workspace/venv/bin/python; S=experiments/green_day3_20261103/source/green_day3_e11_nolabel_20261103.py; LOG=experiments/green_day3_20261103/e11_nolabel; mkdir -p $LOG
export E10_DATA=experiments/meta_ceiling_20261103/synth_data/v5
[ -f $LOG/detector_mixed.pt ] || CUDA_VISIBLE_DEVICES=2 $PY -u $S train-mixed --updates 6000 > $LOG/train_mixed.log 2>&1 || { echo "FAILED train-mixed" >> $LOG/driver_b.log; exit 1; }
CUDA_VISIBLE_DEVICES=2 $PY -u $S score25 --ckpt $LOG/detector_mixed.pt --tag mixed > $LOG/score25_mixed.log 2>&1 || echo "FAILED score25 mixed" >> $LOG/driver_b.log
echo "e11b done $(date -u)" >> $LOG/driver_b.log
