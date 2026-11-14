#!/bin/bash
# Backbone lane queue v2, one slot (2026-11-08). Usage: queue_bb2_20261108.sh <slot>
# Reads jobs_<slot>.tsv before every job: lines "name<TAB>shell command" (# comments allowed), run with bash -c from the repo root,
# CUDA_VISIBLE_DEVICES=3, log logs/<name>.log. runs/<name>/queue_started marks a job as taken. When the list is empty and GPU 3 has
# no other compute process, a keepalive starts and is stopped by its pid before the next job. Never edit this file while it runs.
set -u
cd /workspace/clear-ec-research
LANE=experiments/lane_backbone_20261108
SLOT=$1
JOBS=$LANE/jobs_$SLOT.tsv
LOG=$LANE/logs/queue_$SLOT.log
KA=""
echo $$ > $LANE/queue_$SLOT.pid
say() { echo "$(date -u +%FT%TZ) [$SLOT] $*" >> $LOG; }
stop_ka() { if [ -n "$KA" ] && kill -0 "$KA" 2>/dev/null; then kill "$KA"; say "keepalive $KA stopped"; fi; KA=""; }
trap 'stop_ka; say "queue exit"; exit 0' TERM INT
say "queue v2 start pid $$"
while true; do
  [ -f $LANE/STOP_$SLOT ] && { stop_ka; say "STOP file, exit"; exit 0; }
  next=""; cmd=""
  if [ -f "$JOBS" ]; then
    while IFS=$'\t' read -r name rest; do
      [ -z "$name" ] && continue
      case "$name" in \#*) continue;; esac
      [ -e "$LANE/runs/$name/queue_started" ] && continue
      next=$name; cmd=$rest; break
    done < "$JOBS"
  fi
  if [ -z "$next" ]; then
    others=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep -c "$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 3)")
    if [ "$others" = "0" ] && [ -z "$KA" ]; then
      CUDA_VISIBLE_DEVICES=3 nohup /workspace/venv/bin/python experiments/phase2_pooling_replication_20261026/gpu_keepalive.py --ignore-jobs > $LANE/logs/keepalive_$SLOT.log 2>&1 &
      KA=$!; say "list empty, GPU 3 idle: keepalive $KA"
    fi
    sleep 60; continue
  fi
  stop_ka
  mkdir -p $LANE/runs/$next; date -u +%FT%TZ > $LANE/runs/$next/queue_started
  say "start $next: $cmd"
  CUDA_VISIBLE_DEVICES=3 bash -c "$cmd" > $LANE/logs/$next.log 2>&1
  rc=$?
  say "end $next rc=$rc"
  echo "{\"rc\": $rc, \"ended_utc\": \"$(date -u +%FT%TZ)\"}" > $LANE/runs/$next/queue_exit.json
  sleep 5
done
