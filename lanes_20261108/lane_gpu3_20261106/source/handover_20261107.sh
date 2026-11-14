#!/bin/bash
# Waits for lane GPU 1's cropopt_both_f0 encoder, then stops dinov3_b16_s8 fold 0 on GPU 3 by verified pid, so the queue starts the end-to-end job at once. The partial regression folder moves aside; its SSL stays for the requeued run.
# If the encoder job ends without its marker, the stride 8 run keeps the GPU and the queue's guard skips the end-to-end job. "handover end" lines go to queue.log so the session's waiter wakes.
# Usage: handover_20261107.sh [check]. "check" only verifies the pid.
cd /workspace/clear-ec-research/experiments
M=lane_gpu1_20261106/ssl/cropopt_both_f0/run/complete.json
D=lane_gpu1_20261106/done.tsv
LANE=lane_gpu3_20261106; Q=$LANE/queue.log
PID=69291; QP=$(cat $LANE/queue.pid)
say() { echo "$(date -u +%FT%TZ) $*"; }
note() { say "$*" | tee -a "$Q"; }
verify() {
  [ -r /proc/$PID/cmdline ] || { echo "pid $PID gone"; return 1; }
  local cmd ppid gpu
  cmd=$(tr '\0' ' ' < /proc/$PID/cmdline); ppid=$(ps -o ppid= -p $PID | tr -d ' ')
  gpu=$(tr '\0' '\n' < /proc/$PID/environ | grep '^CUDA_VISIBLE_DEVICES=')
  echo "pid $PID ppid $ppid queue $QP $gpu cmd: $cmd"
  [[ "$cmd" == *"family_recipe_20261106.py all --family dinov3_b16_s8 --fold 0 --seed 46"* ]] && [ "$ppid" = "$QP" ] && [ "$gpu" = "CUDA_VISIBLE_DEVICES=3" ]
}
if [ "${1:-}" = check ]; then verify && echo "check passed" || echo "check FAILED"; exit; fi
say "waiting for $M"
for i in $(seq 1 1440); do
  [ -f "$M" ] && break
  if grep -qsE '^cropopt_both_f0[[:space:]]' "$D"; then note "handover end: lane GPU 1 finished cropopt_both_f0 without the encoder marker, stride 8 keeps the GPU"; exit 1; fi
  [ -e /proc/$PID ] || { note "handover end: stride 8 pid $PID ended before the encoder finished"; exit 1; }
  sleep 30
done
[ -f "$M" ] || { note "handover end: no encoder marker after 12 h, stride 8 keeps the GPU"; exit 1; }
say "encoder complete: $(cat "$M")"
verify || { note "handover end: pid check failed, nothing stopped"; exit 1; }
kill -TERM $PID
for i in $(seq 1 60); do [ -e /proc/$PID ] || break; sleep 2; done
if [ -e /proc/$PID ]; then note "handover end: pid $PID still alive 120 s after TERM, not escalated"; exit 1; fi
R=$LANE/runs/dinov3_b16_s8_f0_s46
mv "$R/runs/fold0/dinov3_b16_s8" "$R/stopped_regression_20261107" && say "partial regression moved to $R/stopped_regression_20261107"
say "handover done"
