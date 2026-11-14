#!/bin/bash
# cell-route lane 2026-11-11: attention-pool member (39-column HEX ridge): smoke, 20-frame no-stub deploy run, budget re-measure (fp32 + fp16-nodet);
# one job at a time on GPU 3 (<= 4 CPU threads), every launch through the backbone lane's guard
cd /workspace/clear-ec-research
L=experiments/lane_cellroute_20261109
G=experiments/lane_backbone_20261108/source/gpu3_launch_20261110.sh
bash $G 8000 "CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout -k 10 1800 /workspace/venv/bin/python $L/inference/test_attnp_f99.py > $L/logs/attnp_f99_smoke.log 2>&1 < /dev/null"
grep -q '^SUMMARY' $L/logs/attnp_f99_smoke.log || { echo "stop: smoke had no SUMMARY $(date -u)"; exit 1; }
echo "done smoke $(date -u)"
bash $G 25000 "CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout -k 10 7200 /workspace/venv/bin/python $L/inference/test_gpu_deploy.py 20 gpu_deploy_20f > $L/logs/gpu_deploy_20f.log 2>&1 < /dev/null"
grep -q '^SUMMARY' $L/logs/gpu_deploy_20f.log || { echo "stop: deploy 20f had no SUMMARY $(date -u)"; exit 1; }
echo "done deploy 20f $(date -u)"
for v in fp32 fp16-nodet; do
  bash $G 25000 "CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout -k 10 7200 /workspace/venv/bin/python $L/inference/test_budget.py 20 $v budget_${v}_v3 > $L/logs/budget_${v}_v3.log 2>&1 < /dev/null"
  grep -q '^SUMMARY' $L/logs/budget_${v}_v3.log || { echo "stop: $v had no SUMMARY $(date -u)"; exit 1; }
  echo "done $v $(date -u)"
done
