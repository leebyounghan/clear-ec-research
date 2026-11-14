#!/bin/bash
cd /workspace/clear-ec-research; L=experiments/lane_gpu0_20261106
sed -i '/run_arm.sh measured_v1 0 46/d' $L/queue_done.txt
sed -i "1a ARM_CONFIG='{\"p\":0.0}' bash experiments/lane_gpu0_20261106/source/run_arm.sh control_single_gpu2 0 46 1 10 recipeA_measured_aug2_20261106.py" $L/queue.txt
setsid nohup bash $L/source/queue_gpu0.sh > $L/logs/queue_runner.log 2>&1 < /dev/null &
sleep 75
cut -c1-120 $L/queue.txt | head -4; echo ---; cut -c1-120 $L/queue_done.txt; echo ---; cut -c1-200 $L/logs/control_single_gpu2_f0_s46.log | tail -3
