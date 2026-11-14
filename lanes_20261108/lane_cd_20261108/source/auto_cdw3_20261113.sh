#!/bin/bash
# Sweep arm cdw3 (coordinator 23:10 EST): CD-weighted joint loss 1.8/0.6/0.6, DINOv3-B/16 fold 0 s46, same SSL init and settings as control P,
# runner copy bb_runner_sweep2_20261113.py (--target-w). Jumps the sweep queue: PAUSE_SWEEP is set; this waits until the pod has at most 9
# training/SSL/TTA processes and nr_throttled rose by at most 3 in the last minute, launches on the card with the fewest jobs (>= 40 GB free),
# records it in sweep_done, then removes PAUSE_SWEEP so the queue resumes.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; LOG=$L/logs; PY=/workspace/venv/bin/python; SSL=experiments/lane_gpu3_20261106/runs/dinov3_b16_f0_s46/ssl/last.pt
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG/auto_cdw3.log; }
cs() { grep -E "^(nr_throttled|usage_usec) " /sys/fs/cgroup/cpu.stat | awk '{print $2}' | tr '\n' ' '; }
say "start pid $$"; read u0 t0 <<< "$(cs)"; t_start=$(date +%s)
while [ $(( $(date +%s) - t_start )) -lt 10800 ]; do
  sleep 60; read u1 t1 <<< "$(cs)"; dth=$((t1 - t0)); t0=$t1; u0=$u1
  nj=$(ps -eo pid=,ppid=,args= | awk '/python/ && /(bb_runner|ssl_variant|family_tta)/ && !/(gpu_launch|bash -c|timeout|awk)/ {p[$1]=$2} END {n=0; for (k in p) if (!(p[k] in p)) n++; print n}')
  [ "$nj" -le 9 ] && [ "$dth" -le 3 ] || { say "wait: jobs $nj throttled +$dth"; continue; }
  best=""; bestn=99
  while IFS=, read -r gi gu fr; do
    gu=$(echo $gu | tr -d ' '); n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$gu"); fr=$(echo $fr | tr -d ' ')
    [ "$fr" -ge 40000 ] && [ "$n" -lt "$bestn" ] && { best=$gi; bestn=$n; }
  done < <(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits)
  [ -z "$best" ] && { say "wait: no GPU"; continue; }
  echo "cdw3|--target-w 1.8,0.6,0.6|gpu$best|$(date -u +%FT%TZ)" >> $L/sweep_done_20261113.txt
  (CUDA_VISIBLE_DEVICES=$best OMP_NUM_THREADS=2 setsid nohup timeout 14400 $PY $L/source/bb_runner_sweep2_20261113.py --family dinov3_b16 --fold 0 --seed 46 --ssl $SSL --schedule-epochs 10 --micro 16 --workers 2 --tta-final --epochs 20 --keep-epochs 16,17,18,19,20 --target-w 1.8,0.6,0.6 --root $L/arms/sweep_cdw3_f0_s46 > $LOG/sweep_cdw3.log 2>&1 < /dev/null &)
  say "launched cdw3 on GPU $best (jobs before $nj, throttled +$dth)"; sleep 240; rm -f $L/PAUSE_SWEEP; say "PAUSE_SWEEP removed"; exit 0
done
say "gave up"; rm -f $L/PAUSE_SWEEP
