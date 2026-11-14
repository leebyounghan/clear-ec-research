#!/bin/bash
# F1 v2 and F2 on folds 0 and 4, then F1b on five folds, beside the co-twin training on the same GPU.
cd /workspace/clear-ec-research/experiments/lane_gpu2_20261106
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2
PY=/workspace/venv/bin/python
echo "$(date -u) F1 v2 start"
$PY source/frozen_read_20261106.py F1 mean_std > logs/F1v2_0.log 2>&1; echo "$(date -u) F1v2_0 rc $?"
$PY source/frozen_read_20261106.py F1 dist,quant,cov,cov_sqrt,all > logs/F1v2_a.log 2>&1 &
A=$!
$PY source/frozen_read_20261106.py F1 autocorr,spectrum,multiscale,localhet,winhead,cls > logs/F1v2_b.log 2>&1 &
B=$!
wait $A; echo "$(date -u) F1v2_a rc $?"
wait $B; echo "$(date -u) F1v2_b rc $?"
$PY source/frozen_read_20261106.py F2 hlg,quantile > logs/F2v2.log 2>&1; echo "$(date -u) F2 rc $?"
$PY source/corr_read_20261106.py > logs/corr.log 2>&1; echo "$(date -u) corr rc $?"
echo "$(date -u) chain done"
