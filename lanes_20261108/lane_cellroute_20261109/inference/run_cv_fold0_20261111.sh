#!/bin/bash
# cell-route lane 2026-11-11: fold-0 CV parity after the CV lane's 6-component freeze, GPU 3 through the backbone lane's guard
cd /workspace/clear-ec-research
L=experiments/lane_cellroute_20261109
bash experiments/lane_backbone_20261108/source/gpu3_launch_20261110.sh 12000 "CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout -k 10 7200 /workspace/venv/bin/python $L/inference/test_cv_fold0.py 120 cv_fold0_120_cv6 > $L/logs/cv_fold0_120_cv6.log 2>&1 < /dev/null"
echo "done cv fold0 rc $? $(date -u)"
