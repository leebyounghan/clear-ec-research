#!/bin/bash
# CPU pod (15:39 EST 11-14, coordinator 15:12): four-fold read of the ViT-L s46 encoder LR 3e-5 probe (cd_lr3e5_read4_20261114.py
# FORM=final, folds 0 / 1 / 2 / 4; self-test passed) once all four folds have predictions_final_tta. Decision-only. Stops after 8 h.
cd /workspace/clear-ec-research/experiments/lane_cd_20261108/source || exit 1
A=../arms; say() { echo "$(date -u +%FT%TZ) $*" >> ../logs/auto_lr3e5_read4.log; }
have4() { for f in 0 1 2 4; do [ -f $A/sweep_vitl_elr3e5_f${f}_s46/raw/runs/fold$f/dinov3_l16t/predictions_final_tta.npz ] || return 1; done; return 0; }
say "start pid $$"; t0=$(date +%s)
until have4; do [ $(( $(date +%s) - t0 )) -gt 28800 ] && { say "not all four folds in 8 h"; exit 1; }; sleep 120; done
sleep 30; FORM=final NR=20 OMP_NUM_THREADS=2 nice -n 19 timeout 3600 /workspace/venv/bin/python cd_lr3e5_read4_20261114.py > ../logs/elr3e5_4fold_final.log 2>&1; say "final rc $?"
