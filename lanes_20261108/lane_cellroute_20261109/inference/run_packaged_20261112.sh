#!/bin/bash
# deploy lane 2026-11-12 (D-PKG-001): packaged-mode parity + census on 20 deploy frames, GPU 3 through the backbone lane's guard
cd /workspace/clear-ec-research
L=experiments/lane_cellroute_20261109
bash experiments/lane_backbone_20261108/source/gpu3_launch_20261110.sh 25000 "CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout -k 10 7200 /workspace/venv/bin/python $L/inference/test_packaged.py 20 packaged_parity_20 > $L/logs/packaged_parity_20.log 2>&1 < /dev/null"
echo "done packaged rc $? $(date -u)"
