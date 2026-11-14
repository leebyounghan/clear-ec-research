#!/bin/bash
# GPU launch guard for any GPU (backbone lane, 2026-11-11; generalises gpu3_launch_20261110.sh, which stays as is for GPU 3).
# usage: gpu_launch_20261111.sh GPU NEED_MIB 'command'     (the command runs with CUDA_VISIBLE_DEVICES=GPU)
# One launch at a time per GPU: flock on runs/gpu<GPU>_launch.lock (GPU 3 shares runs/gpu3_launch.lock with the old guard).
# Under the lock, immediately before launch, it waits while HOLD_GPU<GPU> exists in the lane folder, while any pid in
# HOLD_GPU<GPU>_PIDS (one per line) is alive, or while free memory on that GPU is below NEED_MIB. The lock is kept RAMP seconds
# (default 300) after launch, or until the job exits, so the next guarded launch reads memory after this job allocated.
set -u
cd /workspace/clear-ec-research
LANE=experiments/lane_backbone_20261108; G=$1; NEED=$2; shift 2; CMD="$*"; RAMP=${RAMP:-300}
exec 9>$LANE/runs/gpu${G}_launch.lock
flock 9
while true; do
  held=""
  [ -f $LANE/HOLD_GPU$G ] && held="hold file"
  if [ -f $LANE/HOLD_GPU${G}_PIDS ]; then
    while read -r p; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && held="pid $p alive"; done < $LANE/HOLD_GPU${G}_PIDS
  fi
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G)
  [ -z "$held" ] && [ "$free" -ge "$NEED" ] && break
  sleep 60
done
echo "gpu_launch GPU $G $(date -u +%FT%TZ): free $free MiB >= $NEED, no hold; launching"
CUDA_VISIBLE_DEVICES=$G bash -c "$CMD" 9>&- &
pid=$!
for i in $(seq 1 $RAMP); do kill -0 $pid 2>/dev/null || break; sleep 1; done
exec 9>&-
wait $pid
