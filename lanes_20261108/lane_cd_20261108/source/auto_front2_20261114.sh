#!/bin/bash
# Front of the sweep queue, part 2 (00:10 EST; cdw3 launched 00:07 by auto_front_20261113.sh): source specialists srcold
# (fold-0 training rows with old-format IDs only) and srcnew (new-format only), each validated on all fold-0 validation rows; same SSL init and
# settings as control P; runner bb_runner_sweep3_20261113.py (= sweep2 + split ids 301/302). PAUSE_SWEEP stays until all three are launched.
# Launch rule per arm: <= 9 training/SSL/TTA processes, nr_throttled up by <= 20 of 600 periods in the last minute (the <= 3 rule held cdw3 back 39 min), card with the fewest jobs and >= 40 GB free.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; LOG=$L/logs; PY=/workspace/venv/bin/python; SSL=experiments/lane_gpu3_20261106/runs/dinov3_b16_f0_s46/ssl/last.pt
RUN=$L/source/bb_runner_sweep3_20261113.py; touch $L/PAUSE_SWEEP
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG/auto_front2.log; }
cs() { grep -E "^(nr_throttled|usage_usec) " /sys/fs/cgroup/cpu.stat | awk '{print $2}' | tr '\n' ' '; }
ARMS=("srcold|301|" "srcnew|302|")
say "start pid $$"; read u0 t0 <<< "$(cs)"; t_start=$(date +%s); i=0
while [ $i -lt ${#ARMS[@]} ] && [ $(( $(date +%s) - t_start )) -lt 14400 ]; do
  sleep 60; read u1 t1 <<< "$(cs)"; dth=$((t1 - t0)); t0=$t1; u0=$u1
  nj=$(ps -eo pid=,ppid=,args= | awk '/python/ && /(bb_runner|ssl_variant|family_tta)/ && !/(gpu_launch|bash -c|timeout|awk)/ {p[$1]=$2} END {n=0; for (k in p) if (!(p[k] in p)) n++; print n}')
  [ "$nj" -le 9 ] && [ "$dth" -le 20 ] || { say "wait: jobs $nj throttled +$dth"; continue; }
  best=""; bestn=99
  while IFS=, read -r gi gu fr; do
    gu=$(echo $gu | tr -d ' '); n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$gu"); fr=$(echo $fr | tr -d ' ')
    [ "$fr" -ge 40000 ] && [ "$n" -lt "$bestn" ] && { best=$gi; bestn=$n; }
  done < <(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits)
  [ -z "$best" ] && { say "wait: no GPU"; continue; }
  IFS='|' read -r name fo sw <<< "${ARMS[$i]}"
  echo "$name|fold $fo $sw|gpu$best|$(date -u +%FT%TZ)" >> $L/sweep_done_20261113.txt
  (CUDA_VISIBLE_DEVICES=$best OMP_NUM_THREADS=2 setsid nohup timeout 14400 $PY $RUN --family dinov3_b16 --fold $fo --seed 46 --ssl $SSL --schedule-epochs 10 --micro 16 --workers 2 --tta-final --epochs 20 --keep-epochs 16,17,18,19,20 $sw --root $L/arms/sweep_${name}_f0_s46 > $LOG/sweep_${name}.log 2>&1 < /dev/null &)
  say "launched $name (fold $fo $sw) on GPU $best (jobs before $nj, throttled +$dth)"; i=$((i + 1)); sleep 240; read u0 t0 <<< "$(cs)"
done
rm -f $L/PAUSE_SWEEP; say "done; PAUSE_SWEEP removed"
