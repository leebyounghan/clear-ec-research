#!/bin/bash
# Backbone lane queue for any GPU, one slot (2026-11-11; like queue_bb2_20261108.sh but with the GPU as an argument and no keepalive).
# Usage: queue_gpu_20261111.sh <slot> <gpu>. Reads jobs_<slot>.tsv before every job: lines "name<TAB>shell command" (# comments
# allowed), run with bash -c from the repo root with CUDA_VISIBLE_DEVICES=<gpu>, log logs/<name>.log; runs/<name>/queue_started
# marks a job as taken. STOP_<slot> in the lane folder ends the queue. Never edit this file while it runs.
set -u
cd /workspace/clear-ec-research
LANE=experiments/lane_backbone_20261108
SLOT=$1; GPU=$2
JOBS=$LANE/jobs_$SLOT.tsv
LOG=$LANE/logs/queue_$SLOT.log
echo $$ > $LANE/queue_$SLOT.pid
say() { echo "$(date -u +%FT%TZ) [$SLOT] $*" >> $LOG; }
trap 'say "queue exit"; exit 0' TERM INT
say "queue (gpu $GPU) start pid $$"
while true; do
  [ -f $LANE/STOP_$SLOT ] && { say "STOP file, exit"; exit 0; }
  next=""; cmd=""
  if [ -f "$JOBS" ]; then
    while IFS=$'\t' read -r name rest; do
      [ -z "$name" ] && continue
      case "$name" in \#*) continue;; esac
      [ -e "$LANE/runs/$name/queue_started" ] && continue
      next=$name; cmd=$rest; break
    done < "$JOBS"
  fi
  if [ -z "$next" ]; then sleep 60; continue; fi
  mkdir -p $LANE/runs/$next; date -u +%FT%TZ > $LANE/runs/$next/queue_started
  say "start $next: $cmd"
  CUDA_VISIBLE_DEVICES=$GPU bash -c "$cmd" > $LANE/logs/$next.log 2>&1
  rc=$?
  say "end $next rc=$rc"
done
