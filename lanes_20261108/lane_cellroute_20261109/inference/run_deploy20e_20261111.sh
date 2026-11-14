#!/bin/bash
# cell-route lane 2026-11-11: no-stub 20-frame deploy GPU run after the HEX joint nested ridge (hex_any, deploy_params_joint), GPU 3 through the backbone lane's guard
cd /workspace/clear-ec-research
L=experiments/lane_cellroute_20261109
bash experiments/lane_backbone_20261108/source/gpu3_launch_20261110.sh 25000 "CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout -k 10 7200 /workspace/venv/bin/python $L/inference/test_gpu_deploy.py 20 gpu_deploy_20e > $L/logs/gpu_deploy_20e.log 2>&1 < /dev/null"
echo "done deploy 20e rc $? $(date -u)"
