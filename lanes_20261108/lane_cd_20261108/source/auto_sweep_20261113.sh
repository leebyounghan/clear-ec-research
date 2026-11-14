#!/bin/bash
# D-CD-SWEEP queue runner (CD lane, 11-13): one-factor arms on DINOv3-B/16 fold 0 s46, same SSL init and settings as control P
# (bb_runner_sweep_20261113.py = bb_runner_20261108 + --batch/--clip), 2 loader workers each. Launch rule, checked every 60 s: no PAUSE_SWEEP
# file, fewer than MAXJ training/SSL/TTA processes on the pod, nr_throttled up by at most 3 of 600 periods over the last 60 s and < 14 cores used; GPU = the card with
# the fewest compute processes and >= 40 GB free; 300 s ramp after each launch. Queue: lane_cd_20261108/sweep_queue_20261113.txt (name|switches);
# done list sweep_done_20261113.txt.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; Q=$L/sweep_queue_20261113.txt; DONE=$L/sweep_done_20261113.txt; LOG=$L/logs; PY=/workspace/venv/bin/python
RUN=$L/source/bb_runner_sweep_20261113.py; SSL=experiments/lane_gpu3_20261106/runs/dinov3_b16_f0_s46/ssl/last.pt; MAXJ=${MAXJ:-10}
touch $DONE; say() { echo "$(date -u +%FT%TZ) $*" >> $LOG/auto_sweep.log; }
say "start pid $$ MAXJ $MAXJ"
cs() { grep -E "^(nr_throttled|usage_usec) " /sys/fs/cgroup/cpu.stat | awk '{print $2}' | tr '\n' ' '; }
read u0 t0 <<< "$(cs)"
while true; do
  sleep 60; read u1 t1 <<< "$(cs)"; dth=$((t1 - t0)); cores=$(( (u1 - u0) / 60000000 )); t0=$t1; u0=$u1
  next=$(grep -v -x -F -f <(cut -d'|' -f1 $DONE | sed 's/$/|/' ) $Q 2>/dev/null | awk -F'|' -v d="$(cut -d'|' -f1 $DONE | tr '\n' ',')" 'BEGIN{n=split(d,a,",");for(i=1;i<=n;i++)s[a[i]]=1} !($1 in s){print; exit}')
  [ -z "$next" ] && { say "queue empty; exit"; exit 0; }
  [ -f $L/PAUSE_SWEEP ] && continue
  nj=$(ps -eo pid=,ppid=,args= | awk '/python/ && /(bb_runner|ssl_variant|family_tta)/ && !/(gpu_launch|bash -c|timeout|awk)/ {p[$1]=$2} END {n=0; for (k in p) if (!(p[k] in p)) n++; print n}')
  [ "$nj" -ge "$MAXJ" ] || [ "$dth" -gt 3 ] || [ "$cores" -ge 14 ] && { say "wait: jobs $nj throttled +$dth cores $cores"; continue; }
  best=""; bestn=99
  while IFS=, read -r gi gu fr; do
    gu=$(echo $gu | tr -d ' '); n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$gu"); fr=$(echo $fr | tr -d ' ')
    [ "$fr" -ge 40000 ] && [ "$n" -lt "$bestn" ] && { best=$gi; bestn=$n; }
  done < <(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits)
  [ -z "$best" ] && { say "wait: no GPU with 40 GB free"; continue; }
  name=${next%%|*}; sw=${next#*|}; root=$L/arms/sweep_${name}_f0_s46
  echo "$name|$sw|gpu$best|$(date -u +%FT%TZ)" >> $DONE
  (CUDA_VISIBLE_DEVICES=$best OMP_NUM_THREADS=2 setsid nohup timeout 14400 $PY $RUN --family dinov3_b16 --fold 0 --seed 46 --ssl $SSL --schedule-epochs 10 --micro 16 --workers 2 --tta-final --epochs 20 --keep-epochs 16,17,18,19,20 --root $root $sw > $LOG/sweep_${name}.log 2>&1 < /dev/null &)
  say "launched $name ($sw) on GPU $best; jobs before $nj, cores $cores, throttled +$dth"
  sleep 240; read u0 t0 <<< "$(cs)"
done
