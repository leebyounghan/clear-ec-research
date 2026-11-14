#!/bin/bash
# queue_gpu0_v2.sh [pid of a job already running]: unattended queue for GPU 0. Runs the lines of queue.txt in order, one at a time; a line runs once, tracked in queue_done.txt.
# A failed job is logged and the next line starts, so nobody needs to be watching. With nothing left to run it starts the keepalive; run_arm.sh stops it before the next smoke step.
cd /workspace/clear-ec-research
L=experiments/lane_gpu0_20261106
touch $L/queue.txt $L/queue_done.txt
if [ -n "${1:-}" ]; then
  echo "$(date -u +%FT%TZ) v2 waiting for running job pid $1" >> $L/logs/queue.log
  while kill -0 "$1" 2>/dev/null; do sleep 20; done
fi
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
done
