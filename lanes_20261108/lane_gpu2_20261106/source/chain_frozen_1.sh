#!/bin/bash
# Lane GPU 2: after extraction, frozen reads F1 and F2 on folds 0 and 4, then keepalive until the next job.
cd /workspace/clear-ec-research/experiments/lane_gpu2_20261106
while ! grep -q 'extract done' logs/extract.log; do
  pgrep -f 'token_stats_20261106.py extract' > /dev/null || { echo "$(date -u) extract died"; exit 1; }
  sleep 15
done
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2
PY=/workspace/venv/bin/python
echo "$(date -u) F1 start"
$PY source/frozen_read_20261106.py F1 mean_std > logs/F1_0.log 2>&1 || echo "$(date -u) F1_0 rc $?"
$PY source/frozen_read_20261106.py F1 dist,quant,cov,cov_sqrt,all > logs/F1_a.log 2>&1 &
A=$!
$PY source/frozen_read_20261106.py F1 autocorr,spectrum,multiscale,localhet,winhead,cls > logs/F1_b.log 2>&1 &
B=$!
wait $A; echo "$(date -u) F1_a rc $?"
wait $B; echo "$(date -u) F1_b rc $?"
$PY source/frozen_read_20261106.py F2 hlg,quantile > logs/F2.log 2>&1; echo "$(date -u) F2 rc $?"
echo "$(date -u) chain done"
