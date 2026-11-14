#!/bin/bash
# v2 (00:40 EST 11-14): adds the PRL probe arm (prl1: per metric drop the 1 of 16 rows with the largest loss gradient from epoch 2) after recipe A f0 / f4, before the sweep queue.
# Launch order (coordinator 00:35 EST 11-14): after follow-up f3, f4 and the full-data build have launched (their guards), then srcnew, ViT-L
# fold 0 encoder LR 2e-5, recipe A s46 folds 0 and 4 encoder LR 2e-5 (mixed loss, bb_runner_cdonly_20261111 without TARGET_W = recipe loss,
# per-fold SSL5 init, as the cdonly arms), then the sweep queue resumes (PAUSE_SWEEP removed). Each launch: <= MAXJ jobs, nr_throttled up by
# <= 20 of 600 periods in the last minute, the card with the most free memory and enough of it; 240 s between launches. No running arm is stopped.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; LOG=$L/logs; PY=/workspace/venv/bin/python; MAXJ=${MAXJ:-11}; touch $L/PAUSE_SWEEP
SSL0=experiments/lane_gpu3_20261106/runs/dinov3_b16_f0_s46/ssl/last.pt
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG/auto_master2.log; }
cs() { grep -E "^(nr_throttled|usage_usec) " /sys/fs/cgroup/cpu.stat | awk '{print $2}' | tr '\n' ' '; }
ITEMS=("srcnew|40000|$PY $L/source/bb_runner_sweep3_20261113.py --family dinov3_b16 --fold 302 --seed 46 --ssl $SSL0 --schedule-epochs 10 --micro 16 --workers 2 --tta-final --epochs 20 --keep-epochs 16,17,18,19,20 --root $L/arms/sweep_srcnew_f0_s46"
       "vitl_elr2e5|65000|$PY experiments/lane_backbone_20261108/source/bb_runner_20261108.py --family dinov3_l16t --fold 0 --seed 46 --ssl experiments/lane_backbone_20261108/runs/vitl16_avg_f0_s46/ssl/last.pt --epochs 12 --schedule-epochs 10 --micro 8 --workers 2 --ema 0.999,0.9995 --swa-start 6 --tta-final --keep-epochs 8,9,10,11,12 --enc-lr 2e-5 --root $L/arms/sweep_vitl_elr2e5_f0_s46"
       "ra_elr2e5_f0|40000|$PY $L/source/bb_runner_cdonly_20261111.py --family cpdino_b8 --fold 0 --seed 46 --epochs 20 --schedule-epochs 10 --micro 4 --workers 2 --tta-final --enc-lr 2e-5 --root $L/arms/sweep_ra_elr2e5_f0_s46"
       "ra_elr2e5_f4|40000|$PY $L/source/bb_runner_cdonly_20261111.py --family cpdino_b8 --fold 4 --seed 46 --epochs 20 --schedule-epochs 10 --micro 4 --workers 2 --tta-final --enc-lr 2e-5 --root $L/arms/sweep_ra_elr2e5_f4_s46"
       "prl1|40000|$PY $L/source/bb_runner_prl_20261114.py --family dinov3_b16 --fold 0 --seed 46 --ssl $SSL0 --schedule-epochs 10 --micro 16 --workers 2 --tta-final --epochs 20 --keep-epochs 16,17,18,19,20 --prl 1 --root $L/arms/sweep_prl1_f0_s46")
say "start pid $$ MAXJ $MAXJ"; t_start=$(date +%s)
while [ $(( $(date +%s) - t_start )) -lt 14400 ]; do
  n=0; for f in 3 4 99; do grep -q launching $LOG/fu_elr2e5_f$f.log 2>/dev/null && n=$((n + 1)); done
  [ $n = 3 ] && break; say "wait: follow-up launched $n/3"; sleep 120
done
read u0 t0 <<< "$(cs)"; i=0
while [ $i -lt ${#ITEMS[@]} ] && [ $(( $(date +%s) - t_start )) -lt 21600 ]; do
  sleep 60; read u1 t1 <<< "$(cs)"; dth=$((t1 - t0)); t0=$t1; u0=$u1
  IFS='|' read -r name need cmd <<< "${ITEMS[$i]}"
  nj=$(ps -eo pid=,ppid=,args= | awk '/python/ && /(bb_runner|ssl_variant|family_tta)/ && !/(gpu_launch|bash -c|timeout|awk)/ {p[$1]=$2} END {n=0; for (k in p) if (!(p[k] in p)) n++; print n}')
  read gi fr <<< "$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -1 | tr ',' ' ')"
  [ "$nj" -le "$MAXJ" ] && [ "$dth" -le 20 ] && [ "$fr" -ge "$need" ] || { say "wait $name: jobs $nj throttled +$dth best free $fr on GPU $gi"; continue; }
  echo "$name|master|gpu$gi|$(date -u +%FT%TZ)" >> $L/sweep_done_20261113.txt
  (CUDA_VISIBLE_DEVICES=$gi OMP_NUM_THREADS=2 setsid nohup timeout 36000 $cmd > $LOG/sweep_${name}.log 2>&1 < /dev/null &)
  say "launched $name on GPU $gi (jobs $nj, free $fr, throttled +$dth)"; i=$((i + 1)); sleep 240; read u0 t0 <<< "$(cs)"
done
rm -f $L/PAUSE_SWEEP; say "done; PAUSE_SWEEP removed"
