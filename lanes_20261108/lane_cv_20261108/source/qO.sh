#!/bin/bash
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2
PY=/workspace/venv/bin/python
ORDER_LAMBDA=1 $PY source/frozen_order_head_20261109.py ord1 std,spectrum,cov_sqrt,autocorr --folds 0,4 > logs/fz_ord1_04.log 2>&1; echo "ord1 rc $?"
echo QODONE
