#!/bin/bash
# deploy lane 2026-11-12 (D-GPU-001): tier p (the largest tier inside 300 s on the real T4, run6) on the same 200 rows, started when the t1
# runs of run_t4_200_20261112.sh have finished (one job per GPU, lane CPU cap kept). Same script, TIERX=p.
cd /workspace/clear-ec-research
L=experiments/lane_cellroute_20261109
for i in $(seq 1 360); do grep -q 'done t4 200 t1' $L/logs/run_t4_200_t1.log 2>/dev/null && break; sleep 60; done
grep -q 'done t4 200 t1' $L/logs/run_t4_200_t1.log || { echo "t1 runs not finished after 6 h"; exit 1; }
TIERX=p bash $L/inference/run_t4_200_20261112.sh
