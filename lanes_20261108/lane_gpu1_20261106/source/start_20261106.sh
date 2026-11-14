#!/bin/bash
L=/workspace/clear-ec-research/experiments/lane_gpu1_20261106
cd /workspace/clear-ec-research
A="bash $L/source/add_arm.sh"
$A crop_mean_f4 4 --crop 544 --micro 16
$A crop_std_f4 4 --crop 544 --micro 16 --pool std
$A cropopt_mean_f4 4 --crop 544 --micro 16 --optics measured
$A cropopt_std_f4 4 --crop 544 --micro 16 --optics measured --pool std
$A cropopt_both_f4 4 --crop 544 --micro 16 --optics measured --pool both
(setsid nohup /workspace/venv/bin/python -u $L/source/queue_gpu1_20261106.py > $L/logs/queue.log 2>&1 < /dev/null &)
sleep 2
cut -c1-120 $L/queue.tsv
ps -eo pid,cmd | grep -v grep | grep queue_gpu1 | cut -c1-120
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=3 /workspace/venv/bin/python $L/source/frozen_read_20261106.py ssl5_f4 4
