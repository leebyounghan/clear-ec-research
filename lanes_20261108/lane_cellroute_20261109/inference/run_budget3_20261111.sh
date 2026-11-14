#!/bin/bash
# cell-route lane 2026-11-11: budget re-measure on the current stacks (CD two-seed D, CV 6-component, HEX seedaudit10, real cps5 f99):
# fp32 timing + fp16-nodet storage parity, before writing the half-precision copies (coordinator: approved if median change < 0.05 % per column)
cd /workspace/clear-ec-research
L=experiments/lane_cellroute_20261109
G=experiments/lane_backbone_20261108/source/gpu3_launch_20261110.sh
for v in fp32 fp16-nodet; do
  bash $G 25000 "CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout -k 10 7200 /workspace/venv/bin/python $L/inference/test_budget.py 20 $v budget_${v}_v2 > $L/logs/budget_${v}_v2.log 2>&1 < /dev/null"
  grep -q '^SUMMARY' $L/logs/budget_${v}_v2.log || { echo "stop: $v had no SUMMARY $(date -u)"; exit 1; }
  echo "done $v $(date -u)"
done
