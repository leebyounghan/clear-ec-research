#!/bin/bash
# GPU 3 launch guard (backbone lane, 2026-11-10). usage: gpu3_launch_20261110.sh NEED_MIB 'command'
# One launch at a time: flock on runs/gpu3_launch.lock. Under the lock, immediately before launch, it waits while
#   - the hold file HOLD_GPU3 exists in the lane folder, or
#   - any pid listed in HOLD_GPU3_PIDS (lane folder, one per line) is alive, or
#   - free memory on GPU 3 is below NEED_MIB.
# After launch it keeps the lock for RAMP seconds (default 300, or until the job exits) so the new job allocates its memory before
# the next guarded launch reads free memory; jobs about to start therefore queue behind it instead of reading the same free memory.
set -u
cd /workspace/clear-ec-research
LANE=experiments/lane_backbone_20261108; NEED=$1; shift; CMD="$*"; RAMP=${RAMP:-300}
exec 9>$LANE/runs/gpu3_launch.lock
flock 9
while true; do
  held=""
  [ -f $LANE/HOLD_GPU3 ] && held="hold file"
  if [ -f $LANE/HOLD_GPU3_PIDS ]; then
    while read -r p; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && held="pid $p alive"; done < $LANE/HOLD_GPU3_PIDS
  fi
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3)
  [ -z "$held" ] && [ "$free" -ge "$NEED" ] && break
  sleep 60
done
echo "gpu3_launch $(date -u +%FT%TZ): free $free MiB >= $NEED, no hold; launching"
bash -c "$CMD" 9>&- &
pid=$!
for i in $(seq 1 $RAMP); do kill -0 $pid 2>/dev/null || break; sleep 1; done
exec 9>&-
wait $pid
