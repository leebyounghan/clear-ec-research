#!/bin/bash
# CPU pod: run the interim slot reads when their epochs exist. ViT-L 2e-5 (SLOT=V, epochs 8-10, folds 0 and 4) and recipe A 2e-5 (SLOT=A,
# epochs 8-12, folds 0 and 4). nice 19, one at a time, 2 threads. Stops after 8 h.
cd /workspace/clear-ec-research/experiments/lane_cd_20261108/source || exit 1
A=../arms; PY=/workspace/venv/bin/python; say() { echo "$(date -u +%FT%TZ) $*" >> ../logs/auto_interim.log; }
say "start pid $$"; t0=$(date +%s); dv=0; da=0
while [ $(( $(date +%s) - t0 )) -lt 28800 ] && { [ $dv = 0 ] || [ $da = 0 ]; }; do
  if [ $dv = 0 ] && [ -f $A/sweep_vitl_elr2e5_f0_s46/raw/runs/fold0/dinov3_l16t/predictions_epoch_10.npz ] && [ -f $A/sweep_vitl_elr2e5_f4_s46/raw/runs/fold4/dinov3_l16t/predictions_epoch_10.npz ]; then
    SLOT=V EPOCHS=8-10 OMP_NUM_THREADS=2 nice -n 19 timeout 7200 $PY cd_interim_slot_20261114.py > ../logs/interim_V_8-10.log 2>&1; say "ViT-L read rc $?"; dv=1
  fi
  if [ $da = 0 ] && [ -f $A/sweep_ra_elr2e5_f0_s46/raw/runs/fold0/cpdino_b8/predictions_epoch_12.npz ] && [ -f $A/sweep_ra_elr2e5_f4_s46/raw/runs/fold4/cpdino_b8/predictions_epoch_12.npz ]; then
    SLOT=A EPOCHS=8-12 OMP_NUM_THREADS=2 nice -n 19 timeout 7200 $PY cd_interim_slot_20261114.py > ../logs/interim_A_8-12.log 2>&1; say "recipe A read rc $?"; da=1
  fi
  sleep 120
done
say "exit"
