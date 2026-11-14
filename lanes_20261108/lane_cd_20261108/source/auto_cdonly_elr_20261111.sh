#!/bin/bash
# D-CDTRAIN-001 third arm (coordinator 11-11): recipe A s46, CD loss only (TARGET_W=1,0,0), encoder LR 3e-5 (only change), fold 0. Runs once, on
# whichever of GPUs 0 / 3 frees first after the CD-only folds (launcher logs auto_cdonly_f0.log -> GPU 0, auto_cdonly_f4.log -> GPU 3 contain
# "end rc"), then that card idle in two checks 2 min apart. Gives up after 20 h. Writes only under lane_cd_20261108.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; LOG=$L/logs/auto_cdonly_elr.log
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
say "waiting for a CD-only fold to end (pid $$)"; t0=$(date +%s); G=""; free=0
while [ $(( $(date +%s) - t0 )) -lt 72000 ]; do
  if [ -z "$G" ]; then
    grep -q "end rc" $L/logs/auto_cdonly_f0.log 2>/dev/null && G=0
    [ -z "$G" ] && grep -q "end rc" $L/logs/auto_cdonly_f4.log 2>/dev/null && G=3
    [ -n "$G" ] && say "GPU $G freed by a CD-only fold"
  fi
  if [ -n "$G" ]; then
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
say "gave up after 20 h"
