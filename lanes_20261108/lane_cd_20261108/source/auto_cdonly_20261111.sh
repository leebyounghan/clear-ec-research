#!/bin/bash
# D-CDTRAIN-001 launcher (CD lane, 2026-11-11): recipe A s46 with the CD loss only (TARGET_W=1,0,0), same SSL5, 20 epochs, per-epoch predictions,
# e20 TTA. usage: auto_cdonly_20261111.sh FOLD GPU  -- waits for the backbone lane's runs/synthft_ra_f<FOLD>_s46/exit.json (the job on that card),
# then for GPU <GPU> to show no compute process in two checks 2 min apart, then runs once. Gives up after 10 h. Writes only under lane_cd_20261108.
F=$1; G=$2; cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; B=experiments/lane_backbone_20261108/runs; LOG=$L/logs/auto_cdonly_f$F.log
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
say "waiting for $B/synthft_ra_f${F}_s46/exit.json and GPU $G idle (pid $$)"; t0=$(date +%s); free=0
while [ $(( $(date +%s) - t0 )) -lt 36000 ]; do
  if [ -f $B/synthft_ra_f${F}_s46/exit.json ]; then
    n=$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | grep -c .)
    if [ "$n" = "0" ]; then free=$((free + 1)); else free=0; fi
    if [ $free -ge 2 ]; then
      say "start cdonly_ra_f${F}_s46 on GPU $G"
      TARGET_W=1,0,0 CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /workspace/venv/bin/python $L/source/bb_runner_cdonly_20261111.py --family cpdino_b8 --fold $F --seed 46 --root $L/arms/cdonly_ra_f${F}_s46 --epochs 20 --micro 4 --tta-final --workers 4 > $L/logs/cdonly_ra_f${F}_s46.log 2>&1
      say "end rc $?"; exit 0
    fi
  fi
  sleep 120
done
say "gave up after 10 h"
