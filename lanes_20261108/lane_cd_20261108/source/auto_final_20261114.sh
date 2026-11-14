#!/bin/bash
# CPU pod (04:20 EST 11-14, coordinator 03:58 item 4): true-form (deploy-form) final reads of the heavy-member arms once both folds have
# finished: ViT-L encoder LR 2e-5 (final.pt four-flip TTA, SLOT=V) and recipe A encoder LR 2e-5 (epoch-20 four-flip TTA, SLOT=A), via
# cd_final_slot_20261114.py. nice 19, 2 threads, one at a time. Stops after 14 h.
cd /workspace/clear-ec-research/experiments/lane_cd_20261108/source || exit 1
A=../arms; PY=/workspace/venv/bin/python; say() { echo "$(date -u +%FT%TZ) $*" >> ../logs/auto_final.log; }
say "start pid $$"; t0=$(date +%s); dv=0; da=0
while [ $(( $(date +%s) - t0 )) -lt 50400 ] && { [ $dv = 0 ] || [ $da = 0 ]; }; do
  if [ $dv = 0 ] && [ -f $A/sweep_vitl_elr2e5_f0_s46/raw/runs/fold0/dinov3_l16t/predictions_final_tta.npz ] && [ -f $A/sweep_vitl_elr2e5_f4_s46/raw/runs/fold4/dinov3_l16t/predictions_final_tta.npz ]; then
    sleep 60; SLOT=V OMP_NUM_THREADS=2 nice -n 19 timeout 7200 $PY cd_final_slot_20261114.py > ../logs/final_V.log 2>&1; say "ViT-L final read rc $?"; dv=1
  fi
  if [ $da = 0 ] && [ -f $A/sweep_ra_elr2e5_f0_s46/raw/runs/fold0/cpdino_b8/predictions_final_tta.npz ] && [ -f $A/sweep_ra_elr2e5_f4_s46/raw/runs/fold4/cpdino_b8/predictions_final_tta.npz ]; then
    sleep 60; SLOT=A OMP_NUM_THREADS=2 nice -n 19 timeout 7200 $PY cd_final_slot_20261114.py > ../logs/final_A.log 2>&1; say "recipe A final read rc $?"; da=1
  fi
  sleep 120
done
say "exit"
