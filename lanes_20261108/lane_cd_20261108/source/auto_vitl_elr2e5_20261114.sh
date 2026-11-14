#!/bin/bash
# Coordinator 00:25 EST 11-14 item 2: ViT-L/16 fold 0 s46 at encoder LR 2e-5 with the unchanged mixed loss, like for like with the plain
# fold-0 ViT-L s46 (lane_backbone_20261108/runs/vitl16_avg_f0_s46: bb_runner_20261108, 12 epochs, micro 8, ema 0.999/0.9995, swa6, same SSL).
# Starts after the five encoder-LR-2e-5 follow-up jobs have launched, when the pod has <= 10 jobs and a card has >= 65 GB free (guard).
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; LOG=$L/logs; PY=/workspace/venv/bin/python; G=experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG/auto_vitl_elr2e5.log; }
say "start pid $$"; t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt 14400 ]; do
  sleep 120
  n=0; for f in 3 4 99; do grep -q launching $LOG/fu_elr2e5_f$f.log 2>/dev/null && n=$((n + 1)); done
  [ $n = 3 ] || { say "wait: follow-up launched $n/3"; continue; }
  nj=$(ps -eo pid=,ppid=,args= | awk '/python/ && /(bb_runner|ssl_variant|family_tta)/ && !/(gpu_launch|bash -c|timeout|awk)/ {p[$1]=$2} END {n=0; for (k in p) if (!(p[k] in p)) n++; print n}')
  read gi fr <<< "$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -1 | tr ',' ' ')"
  [ "$nj" -le 10 ] && [ "$fr" -ge 65000 ] || { say "wait: jobs $nj, best free $fr MiB on GPU $gi"; continue; }
  (OMP_NUM_THREADS=2 setsid nohup bash $G $gi 65000 "$PY experiments/lane_backbone_20261108/source/bb_runner_20261108.py --family dinov3_l16t --fold 0 --seed 46 --ssl experiments/lane_backbone_20261108/runs/vitl16_avg_f0_s46/ssl/last.pt --epochs 12 --schedule-epochs 10 --micro 8 --workers 2 --ema 0.999,0.9995 --swa-start 6 --tta-final --keep-epochs 8,9,10,11,12 --enc-lr 2e-5 --root $L/arms/sweep_vitl_elr2e5_f0_s46" > $LOG/sweep_vitl_elr2e5.log 2>&1 < /dev/null &)
  say "launched ViT-L 2e-5 on GPU $gi (jobs $nj, free $fr)"; exit 0
done
say "gave up"
