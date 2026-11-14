#!/bin/bash
L=/workspace/clear-ec-research/experiments/lane_gpu1_20261106
date -u
cat $L/done.tsv 2>/dev/null
ls $L/features
tail -2 $L/logs/baseline_features.log
for n in $(cut -f1 $L/queue.tsv); do [ -f $L/logs/$n.log ] && { echo "--- $n"; tail -3 $L/logs/$n.log | cut -c1-300; }; done
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv -i 1
nvidia-smi -i 1 --query-compute-apps=pid,used_memory --format=csv,noheader
pgrep -f queue_gpu1_20261106 | wc -l
cat /proc/loadavg
cat /sys/fs/cgroup/memory.current
