#!/bin/bash
# v3 (06:50 EST 11-14, coordinator 06:49): recipe A 2e-5 stopped; f1 already runs on GPU 2 (v2). f2 on GPU 1 now (next to S2b). f3 and the
# 9,000-row full-data build on GPUs 0 and 3, one ViT-L follow-up job per card, as soon as a card has >= 60 GB free (no longer waiting for the
# arms f0 / f4; replaces 06:27 item 4). Note: when DINOv3 2e-5 f1 / f2 end, cards 0 / 3 have only ~52 GB free while the ViT-L arm and the
# other DINOv3 fold still run there, and a ViT-L run peaks at ~54.6 GB, so the launch on each card comes when its second job ends.
# v1 / v2 were stopped before this started. Same commands as v1 / v2.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; A=$L/arms; LOG=$L/logs; PY=/workspace/venv/bin/python
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG/auto_vitl_fu.log; }
VS="--epochs 12 --schedule-epochs 10 --micro 8 --workers 2 --ema 0.998,0.999 --swa-start 6 --keep-epochs 8,9,10,11,12 --enc-lr 2e-5"
cmd_f() { echo "$PY experiments/lane_backbone_20261108/source/bb_runner_20261108.py --family dinov3_l16t --fold $1 --seed 46 --ssl $A/vitl16_f${1}_s46/ssl/last.pt --tta-final $VS --root $A/vitlfu_elr2e5_f${1}_s46"; }
CMD99="$PY $L/source/bb_runner_full9000_20261113.py --family dinov3_l16t --fold -1 --seed 46 --ssl $A/vitl16_f99_s46/ssl/last.pt $VS --root $A/vitlfu_elr2e5_f99_s46"
go() { echo "$1|vitl_fu3|gpu$2|$(date -u +%FT%TZ)" >> $L/sweep_done_20261113.txt; (CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS=2 setsid nohup timeout 50400 $3 > $LOG/$1.log 2>&1 < /dev/null &); say "launched $1 on GPU $2 (free $4)"; }
say "v3 start pid $$"; t0=$(date +%s)
fr1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')
if [ "$fr1" -ge 60000 ]; then go vitlfu_elr2e5_f2 1 "$(cmd_f 2)" $fr1; else say "GPU 1 has only $fr1 MiB free; f2 not launched"; fi
sleep 120
names=(vitlfu_elr2e5_f3 vitlfu_elr2e5_f99); cmds=("$(cmd_f 3)" "$CMD99"); used=""; i=0
while [ $i -lt 2 ] && [ $(( $(date +%s) - t0 )) -lt 50400 ]; do
  read gi fr <<< "$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v u="$used" '($1==0 || $1==3) && index(u, "g" $1 "g") == 0' | sort -t, -k2 -n -r | head -1 | tr ',' ' ')"
  [ -n "$fr" ] && [ "$fr" -ge 60000 ] || { [ $(( ($(date +%s) - t0) % 600 )) -lt 60 ] && say "wait ${names[$i]}: best free on 0/3 $fr (GPU $gi)"; sleep 60; continue; }
  go ${names[$i]} $gi "${cmds[$i]}" $fr; used="${used}g${gi}g"; i=$((i + 1)); sleep 300
done
say "v3 done"
