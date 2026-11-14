#!/bin/bash
# CV lane: full-data frozen heads: token statistics with the full-data recipe A encoder on all rows, then heads on all good rows
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2
PY=/workspace/venv/bin/python
T0=$(date +%s); $PY source/token_stats_cv_20261108.py own 99 > logs/own_extract_f99.log 2>&1; echo "$(date -u) extract f99 rc $? seconds $(( $(date +%s) - T0 ))"
T0=$(date +%s); $PY source/frozen_full_heads_20261109.py > logs/frozen_full_heads.log 2>&1; echo "$(date -u) heads rc $? seconds $(( $(date +%s) - T0 ))"
echo QHDONE
