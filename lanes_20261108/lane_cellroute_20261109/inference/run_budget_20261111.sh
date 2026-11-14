#!/bin/bash
# cell-route lane budget runs (2026-11-11): storage variants fp32, fp16, bf16mix on 20 deploy frames, one after another, each launched through the
# backbone lane's GPU 3 launch guard (flock + free-memory check). Stops if a variant ends without a SUMMARY line.
cd /workspace/clear-ec-research
L=experiments/lane_cellroute_20261109
G=experiments/lane_backbone_20261108/source/gpu3_launch_20261110.sh
for v in fp32 fp16 bf16mix; do
  bash $G 25000 "CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout -k 10 7200 /workspace/venv/bin/python $L/inference/test_budget.py 20 $v budget_$v > $L/logs/budget_$v.log 2>&1 < /dev/null"
  grep -q '^SUMMARY' $L/logs/budget_$v.log || { echo "stop: $v had no SUMMARY $(date -u)"; exit 1; }
  echo "done $v $(date -u)"
done
