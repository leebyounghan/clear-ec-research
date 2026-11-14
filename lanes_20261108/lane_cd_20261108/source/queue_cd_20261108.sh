#!/bin/bash
# CD lane queue runner, GPU 0 only (2026-11-08). Runs queue.txt top to bottom, one job at a time; re-reads the file every loop, so lines can be added while it runs.
# Line: NAME<TAB>ENV (space-separated K=V or -)<TAB>runner args (without --root). Lines starting with # are skipped. Finished names go to queue_done.txt with rc.
# Waits until no compute process is on GPU 0 before starting a job. Touch STOP in the lane folder to end after the current job.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108
PY=/workspace/venv/bin/python
Q=$L/queue.txt; D=$L/queue_done.txt; LOG=$L/logs/queue.log
touch $D
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
say "runner start pid $$"
while true; do
  [ -f $L/STOP ] && { say "STOP file, exit"; exit 0; }
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
  env $envs CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY $L/source/family_cd_20261108.py all $args --root $L/arms/$name --workers 4 > $L/logs/$name.log 2>&1
  rc=$?
  printf '%s\t%s\t%s\n' "$name" "$rc" "$(date -u +%FT%TZ)" >> $D
  say "end $name rc=$rc"
  [ $rc -ne 0 ] && sleep 30
done
