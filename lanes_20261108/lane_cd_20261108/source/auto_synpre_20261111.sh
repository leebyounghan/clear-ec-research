#!/bin/bash
# D-SYN-002/003 automation (CD lane, 2026-11-11): CPU pod only, 2 threads. Waits for the backbone lane's synthetic runs and then
#  (1) synthetic ViT-L folds 0+4 (exit.json in runs/synthft_vitl_f{0,4}_s46): CD in-place read with SYN_V -> logs/synpre_vitl_read.log
#  (2) synthetic recipe A folds 0+4 (exit.json in runs/synthft_ra_f{0,4}_s46): build reads/recipeA_swapped_syn46_20261111.npz (for the CV and HEX
#      lanes, A-CD-SWAPA-PLAN), its sha256 -> logs/swappedA.sha256, then the CD read with SYN_A, and with SYN_A + SYN_V -> logs/synpre_ra_read.log
# Gives up after 16 h. Reads only; writes only under lane_cd_20261108.
cd /workspace/clear-ec-research/experiments/lane_cd_20261108/source || exit 1
B=/workspace/clear-ec-research/experiments/lane_backbone_20261108/runs; PY=/workspace/venv/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
V="SYN_V=lane_backbone_20261108/runs/synthft_vitl_f{f}_s46 SYN_V_TRACK=raw SYN_V_FAM=dinov3_l16t"
A="SYN_A=lane_backbone_20261108/runs/synthft_ra_f{f}_s46 SYN_A_TRACK=raw SYN_A_FAM=cpdino_b8"
say() { echo "$(date -u +%FT%TZ) $*" >> ../logs/auto_synpre.log; }
say "start pid $$"; doneV=0; doneA=0; t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt 57600 ]; do
  if [ $doneV = 0 ] && [ -f $B/synthft_vitl_f0_s46/exit.json ] && [ -f $B/synthft_vitl_f4_s46/exit.json ]; then
    say "ViT-L runs complete; reading"; env $V timeout 3000 $PY cd_synpre_inplace_20261111.py > ../logs/synpre_vitl_read.log 2>&1; say "ViT-L read rc $?"; doneV=1
  fi
  if [ $doneA = 0 ] && [ -f $B/synthft_ra_f0_s46/exit.json ] && [ -f $B/synthft_ra_f4_s46/exit.json ]; then
    say "recipe A runs complete; building swapped A"
    timeout 900 $PY build_swapped_recipeA_20261111.py > ../logs/swappedA_build.log 2>&1; say "build rc $?"
    sha256sum ../reads/recipeA_swapped_syn46_20261111.npz > ../logs/swappedA.sha256 2>&1
    env $A timeout 3000 $PY cd_synpre_inplace_20261111.py > ../logs/synpre_ra_read.log 2>&1; say "A read rc $?"
    if [ $doneV = 1 ]; then env $A $V timeout 3000 $PY cd_synpre_inplace_20261111.py > ../logs/synpre_both_read.log 2>&1; say "both read rc $?"; fi
    doneA=1
  fi
  [ $doneV = 1 ] && [ $doneA = 1 ] && { say "all done"; exit 0; }
  sleep 120
done
say "gave up after 16 h (doneV=$doneV doneA=$doneA)"
