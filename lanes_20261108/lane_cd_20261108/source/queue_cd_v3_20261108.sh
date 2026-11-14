#!/bin/bash
# CD lane queue runner v3, GPU 0 only (2026-11-08). Adds "cmd:" lines (rest run with bash -c from the repo root; logs to logs/NAME.log). Same queue.txt / queue_done.txt as v1. Waits for the v1 runner pid (arg 1) to exit before starting.
# Line: NAME<TAB>ENV (K=V ... or -)<TAB>ARGS. ARGS starting with "tta:" run family_tta_20261108.py with the rest (RUNROOT FAMILY FOLD [SEED]);
# otherwise family_cd_20261108.py all ARGS --root arms/NAME --workers ${WORKERS:-4}. Touch STOP3 in the lane folder to end after the current job.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108
PY=/workspace/venv/bin/python
Q=$L/queue.txt; D=$L/queue_done.txt; LOG=$L/logs/queue.log
OLD=${1:-}
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
if [ -n "$OLD" ]; then while kill -0 "$OLD" 2>/dev/null; do sleep 20; done; fi
say "runner v3 start pid $$"
while true; do
  [ -f $L/STOP3 ] && { say "STOP3 file, exit"; exit 0; }
  job=""
  while IFS= read -r line; do
    [ -z "$line" ] && continue; case "$line" in \#*) continue;; esac
    name=$(printf '%s' "$line" | cut -f1)
    grep -q -P "^${name}\t" $D && continue
    job="$line"; break
  done < $Q
  if [ -z "$job" ]; then say "queue empty"; sleep 120; continue; fi
  n=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | grep -c .)
  if [ "$n" -gt 0 ]; then sleep 60; continue; fi
  name=$(printf '%s' "$job" | cut -f1); envs=$(printf '%s' "$job" | cut -f2); args=$(printf '%s' "$job" | cut -f3)
  [ "$envs" = "-" ] && envs=""
  say "start $name env[$envs] args[$args]"
  case "$args" in
    cmd:*) env $envs CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 bash -c "${args#cmd:}" > $L/logs/$name.log 2>&1;;
    tta:*) env $envs CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 $PY $L/source/family_tta_20261108.py ${args#tta:} > $L/logs/$name.log 2>&1;;
    *) env $envs CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY $L/source/family_cd_20261108.py all $args --root $L/arms/$name --workers ${WORKERS:-4} > $L/logs/$name.log 2>&1;;
  esac
  rc=$?
  printf '%s\t%s\t%s\n' "$name" "$rc" "$(date -u +%FT%TZ)" >> $D
  say "end $name rc=$rc"
  [ $rc -ne 0 ] && sleep 30
done
