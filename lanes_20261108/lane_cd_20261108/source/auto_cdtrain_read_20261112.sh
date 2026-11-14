#!/bin/bash
# D-CDTRAIN-002 reads (CPU pod): runs cd_cdtrain_read_20261112.py (1) once both fold-0 arms have predictions_epoch_10 (interim, copy kept as
# reads/cdtrain_read_interim_e10_20261112.json) and (2) once both have predictions_final_tta.npz (final). Gives up after 18 h.
cd /workspace/clear-ec-research/experiments/lane_cd_20261108/source || exit 1
A=/workspace/clear-ec-research/experiments/lane_cd_20261108/arms; export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
D1=$A/cdonly_ra_f0_s46/raw/runs/fold0/cpdino_b8; D3=$A/cdonly_elr3e5_ra_f0_s46/raw/runs/fold0/cpdino_b8
say() { echo "$(date -u +%FT%TZ) $*" >> ../logs/auto_cdtrain_read.log; }
say "start pid $$"; t0=$(date +%s); interim=0
while [ $(( $(date +%s) - t0 )) -lt 64800 ]; do
  if [ $interim = 0 ] && [ -f $D1/predictions_epoch_10.npz ] && [ -f $D3/predictions_epoch_10.npz ]; then
    sleep 30; timeout 900 /workspace/venv/bin/python cd_cdtrain_read_20261112.py > ../logs/cdtrain_read_interim.log 2>&1; say "interim read rc $?"
    cp ../reads/cdtrain_read_20261112.json ../reads/cdtrain_read_interim_e10_20261112.json; interim=1
  fi
  if [ -f $D1/predictions_final_tta.npz ] && [ -f $D3/predictions_final_tta.npz ]; then
    sleep 60; timeout 900 /workspace/venv/bin/python cd_cdtrain_read_20261112.py > ../logs/cdtrain_read_final.log 2>&1; say "final read rc $?"; exit 0
  fi
  sleep 300
done
say "gave up after 18 h"
