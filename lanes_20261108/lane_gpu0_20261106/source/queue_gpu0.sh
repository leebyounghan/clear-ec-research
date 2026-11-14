#!/bin/bash
# Runs the lines of queue.txt in order, one at a time, GPU 0 only. queue.txt is data and may be edited while this runs; a line runs once, tracked in queue_done.txt.
# With nothing to run it starts the keepalive; run_arm.sh stops it before the next smoke step.
cd /workspace/clear-ec-research
L=experiments/lane_gpu0_20261106
touch $L/queue.txt $L/queue_done.txt
while true; do
  next=$(grep -v '^#' $L/queue.txt | grep -v '^$' | grep -vxFf $L/queue_done.txt | head -1)
  if [ -z "$next" ]; then
    kp=$(cat $L/keepalive.pid 2>/dev/null)
    if [ -z "$kp" ] || ! grep -qa gpu_keepalive /proc/$kp/cmdline 2>/dev/null; then
      CUDA_VISIBLE_DEVICES=0 setsid nohup /workspace/venv/bin/python experiments/phase2_pooling_replication_20261026/gpu_keepalive.py --ignore-jobs --duty 0.35 --minutes 1440 > $L/keepalive.log 2>&1 < /dev/null &
      echo $! > $L/keepalive.pid
      echo "$(date -u +%FT%TZ) queue empty, keepalive started pid $(cat $L/keepalive.pid)" >> $L/logs/queue.log
    fi
    sleep 30
    continue
  fi
  echo "$(date -u +%FT%TZ) start: $next" >> $L/logs/queue.log
  echo "$next" >> $L/queue_done.txt
  bash -c "$next"
  rc=$?
  echo "$(date -u +%FT%TZ) end rc=$rc: $next" >> $L/logs/queue.log
  if [ $rc -ne 0 ]; then
    # A failed job does not take the rest of the queue with it: move the runner to keepalive and wait for a fix.
    touch $L/queue_hold
  fi
  while [ -e $L/queue_hold ]; do
    kp=$(cat $L/keepalive.pid 2>/dev/null)
    if [ -z "$kp" ] || ! grep -qa gpu_keepalive /proc/$kp/cmdline 2>/dev/null; then
      CUDA_VISIBLE_DEVICES=0 setsid nohup /workspace/venv/bin/python experiments/phase2_pooling_replication_20261026/gpu_keepalive.py --ignore-jobs --duty 0.35 --minutes 1440 > $L/keepalive.log 2>&1 < /dev/null &
      echo $! > $L/keepalive.pid
    fi
    sleep 30
  done
done
