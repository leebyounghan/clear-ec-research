#!/bin/bash
# D-CDTRAIN-002 (coordinator 11-11): GPU 3 arm = recipe A s46, CD loss only (TARGET_W=1,0,0), encoder LR 3e-5 (only change), fold 0. Waits for the
# backbone lane's runs/synthft_ra_f4_s46/exit.json (the job on GPU 3), then GPU 3 idle in two checks 2 min apart, then runs once. Gives up after 10 h.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; B=experiments/lane_backbone_20261108/runs; LOG=$L/logs/auto_cdonly_elr_g3.log; G=3
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
say "waiting for $B/synthft_ra_f4_s46/exit.json and GPU $G idle (pid $$)"; t0=$(date +%s); free=0
while [ $(( $(date +%s) - t0 )) -lt 36000 ]; do
  if [ -f $B/synthft_ra_f4_s46/exit.json ]; then
    n=$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | grep -c .)
    if [ "$n" = "0" ]; then free=$((free + 1)); else free=0; fi
    if [ $free -ge 2 ]; then
      say "start cdonly_elr3e5_ra_f0_s46 on GPU $G"
      TARGET_W=1,0,0 CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /workspace/venv/bin/python $L/source/bb_runner_cdonly_20261111.py --family cpdino_b8 --fold 0 --seed 46 --root $L/arms/cdonly_elr3e5_ra_f0_s46 --epochs 20 --micro 4 --tta-final --workers 4 --enc-lr 3e-5 > $L/logs/cdonly_elr3e5_ra_f0_s46.log 2>&1
      say "end rc $?"; exit 0
    fi
  fi
  sleep 120
done
say "gave up after 10 h"
