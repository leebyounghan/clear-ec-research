#!/bin/bash
# Stops this lane's keepalive by the pid in keepalive.pid, only if that pid really is gpu_keepalive.py.
L=/workspace/clear-ec-research/experiments/lane_gpu0_20261106
kp=$(cat $L/keepalive.pid 2>/dev/null)
if [ -n "$kp" ] && grep -qa gpu_keepalive /proc/$kp/cmdline 2>/dev/null; then
  echo "stopping keepalive pid $kp: $(tr '\0' ' ' < /proc/$kp/cmdline)"
  kill $kp
  for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 $kp 2>/dev/null || break; sleep 1; done
else
  echo "no keepalive running under pid '$kp'"
fi
