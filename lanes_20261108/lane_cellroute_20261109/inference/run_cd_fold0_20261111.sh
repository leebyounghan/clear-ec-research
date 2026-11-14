#!/bin/bash
# cell-route lane 2026-11-11: fold-0 CD parity after the CD lane's seed audit (two-seed D), GPU 3 through the backbone lane's guard
cd /workspace/clear-ec-research
L=experiments/lane_cellroute_20261109
bash experiments/lane_backbone_20261108/source/gpu3_launch_20261110.sh 8000 "CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout -k 10 5400 /workspace/venv/bin/python $L/inference/test_cd_fold0.py 120 cd_fold0_120_d2 > $L/logs/cd_fold0_120_d2.log 2>&1 < /dev/null"
echo "done cd fold0 rc $? $(date -u)"
