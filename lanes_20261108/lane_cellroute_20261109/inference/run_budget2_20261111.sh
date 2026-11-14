#!/bin/bash
# cell-route lane 2026-11-11: selective half-precision storage variants, then the no-stub deploy rerun once CPSAM f99 is complete.
# One job of mine at a time on GPU 3 (<= 4 CPU threads), every launch through the backbone lane's guard.
cd /workspace/clear-ec-research
L=experiments/lane_cellroute_20261109
G=experiments/lane_backbone_20261108/source/gpu3_launch_20261110.sh
for v in fp16-nodet fp16-cdonly; do
  bash $G 25000 "CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout -k 10 7200 /workspace/venv/bin/python $L/inference/test_budget.py 20 $v budget_$v > $L/logs/budget_$v.log 2>&1 < /dev/null"
  grep -q '^SUMMARY' $L/logs/budget_$v.log || { echo "stop: $v had no SUMMARY $(date -u)"; exit 1; }
  echo "done $v $(date -u)"
done
C=experiments/lane_hex_20261108/deploy/mtcps_c512_f99/runs/fold99/cpsam_c512/complete.json
for i in $(seq 1 24); do [ -f $C ] && break; sleep 300; done
[ -f $C ] || { echo "stop: CPSAM f99 not complete after 2 h $(date -u)"; exit 1; }
echo "CPSAM f99 complete, deploy rerun $(date -u)"
bash $G 25000 "CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout -k 10 7200 /workspace/venv/bin/python $L/inference/test_gpu_deploy.py 20 gpu_deploy_20b > $L/logs/gpu_deploy_20b.log 2>&1 < /dev/null"
echo "done deploy rerun rc $? $(date -u)"
