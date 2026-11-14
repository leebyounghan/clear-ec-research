#!/bin/bash
# CPU pod (13:33 EST 11-14): hard-image probe reads (cd_hard2_read_20261114.py, self-test passed: arm = reference gives 0 everywhere and the
# reference in V's half equals p3 in the final form). Interim (single view epochs 8-10) when both folds have predictions_epoch_10; final
# (final.pt TTA) when both have predictions_final_tta. nice 19, 2 threads. Stops after 8 h.
cd /workspace/clear-ec-research/experiments/lane_cd_20261108/source || exit 1
PY=/workspace/venv/bin/python; A=../arms; say() { echo "$(date -u +%FT%TZ) $*" >> ../logs/auto_hard2_read.log; }
say "start pid $$"; t0=$(date +%s); di=0; df=0
while [ $(( $(date +%s) - t0 )) -lt 28800 ] && { [ $di = 0 ] || [ $df = 0 ]; }; do
  if [ $di = 0 ] && [ -f $A/sweep_vitl_elr2e5_hard2_f0_s46/raw/runs/fold0/dinov3_l16t/predictions_epoch_10.npz ] && [ -f $A/sweep_vitl_elr2e5_hard2_f4_s46/raw/runs/fold4/dinov3_l16t/predictions_epoch_10.npz ]; then
    FORM=interim EPOCHS=8-10 OMP_NUM_THREADS=2 nice -n 19 timeout 3600 $PY cd_hard2_read_20261114.py > ../logs/hard2_interim.log 2>&1; say "interim rc $?"; di=1
  fi
  if [ $df = 0 ] && [ -f $A/sweep_vitl_elr2e5_hard2_f0_s46/raw/runs/fold0/dinov3_l16t/predictions_final_tta.npz ] && [ -f $A/sweep_vitl_elr2e5_hard2_f4_s46/raw/runs/fold4/dinov3_l16t/predictions_final_tta.npz ]; then
    sleep 30; FORM=final OMP_NUM_THREADS=2 nice -n 19 timeout 3600 $PY cd_hard2_read_20261114.py > ../logs/hard2_final.log 2>&1; say "final rc $?"; df=1
  fi
  sleep 120
done
say "exit"
