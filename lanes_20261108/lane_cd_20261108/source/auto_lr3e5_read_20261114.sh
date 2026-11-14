#!/bin/bash
# CPU pod (14:52 EST 11-14): final read of the ViT-L s46 encoder LR 3e-5 probe (cd_lr3e5_read_20261114.py FORM=final) once both folds have
# predictions_final_tta. Decision-only. Stops after 6 h.
cd /workspace/clear-ec-research/experiments/lane_cd_20261108/source || exit 1
A=../arms; say() { echo "$(date -u +%FT%TZ) $*" >> ../logs/auto_lr3e5_read.log; }
say "start pid $$"; t0=$(date +%s)
until [ -f $A/sweep_vitl_elr3e5_f0_s46/raw/runs/fold0/dinov3_l16t/predictions_final_tta.npz ] && [ -f $A/sweep_vitl_elr3e5_f4_s46/raw/runs/fold4/dinov3_l16t/predictions_final_tta.npz ]; do
  [ $(( $(date +%s) - t0 )) -gt 21600 ] && { say "no final TTA in 6 h"; exit 1; }; sleep 120; done
sleep 30; FORM=final OMP_NUM_THREADS=2 nice -n 19 timeout 3600 /workspace/venv/bin/python cd_lr3e5_read_20261114.py > ../logs/elr3e5_final.log 2>&1; say "final rc $?"
