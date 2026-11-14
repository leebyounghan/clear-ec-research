#!/bin/bash
L=/workspace/clear-ec-research/experiments/lane_gpu1_20261106
kill 59076; sleep 1
mv $L/done.tsv $L/done_failed_env_20261106.tsv
cd /workspace/clear-ec-research
(setsid nohup /workspace/venv/bin/python -u $L/source/queue_gpu1_20261106.py > $L/logs/queue.log 2>&1 < /dev/null &)
sleep 1
pgrep -af queue_gpu1_20261106 | cut -c1-110
cat $L/done_failed_env_20261106.tsv
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=3
R="/workspace/venv/bin/python $L/source/frozen_read_20261106.py"
$R ext 4 --ref ssl5_f4
$R ssl10_f4 4 --ref ssl5_f4
$R ssl20_f4 4 --ref ssl5_f4
$R tuned_f4 4 --ref ssl5_f4
$R ssl5_f0 0
$R ext 0 --ref ssl5_f0
