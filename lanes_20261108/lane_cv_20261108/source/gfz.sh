#!/bin/bash
# CV lane: frozen token-statistic heads on folds 1-3 (own encoders), to make the CV stack five-fold
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2
PY=/workspace/venv/bin/python
until grep -q ALLDONE logs/prep_batch3.log; do sleep 20; done
for f in 1 2 3; do date; $PY source/token_stats_cv_20261108.py own $f > logs/own_extract_f$f.log 2>&1; echo "own extract f$f rc $?"; done
date; $PY source/frozen_read_own_cv_20261108.py std,spectrum,cov_sqrt,autocorr,winhead,F2:quantile,all --folds 1,2,3 > logs/own_read_123.log 2>&1; echo "own read rc $?"
date; echo GFZDONE
