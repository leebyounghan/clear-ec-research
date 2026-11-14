#!/bin/bash
# Fallback read: extract with each fold's own encoder, then F1 own and F2 own on folds 0 and 4, beside the co-twin training.
cd /workspace/clear-ec-research/experiments/lane_gpu2_20261106
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2
PY=/workspace/venv/bin/python
for f in 0 4; do $PY source/token_stats_20261106.py own $f > logs/own_extract_f$f.log 2>&1; echo "$(date -u) own extract f$f rc $?"; done
$PY source/frozen_read_own_20261106.py std,mean_std > logs/own_0.log 2>&1; echo "$(date -u) own_0 rc $?"
$PY source/frozen_read_own_20261106.py dist,quant,cov,cov_sqrt,all,F2:hlg > logs/own_a.log 2>&1 &
A=$!
$PY source/frozen_read_own_20261106.py autocorr,spectrum,multiscale,localhet,winhead,cls,F2:quantile > logs/own_b.log 2>&1 &
B=$!
wait $A; echo "$(date -u) own_a rc $?"
wait $B; echo "$(date -u) own_b rc $?"
echo "$(date -u) chain done"
