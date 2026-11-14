#!/bin/bash
# D-CDTRAIN-003 final reads (CPU pod): for each CD-only arm, when its fold 0 and fold 4 runs both have the e20 TTA, run the strict folds 0+4
# cross-fit in place of plain recipe A s46 (snapshot form e8-19 + e20 TTA; incumbent in the same run) -> logs/cdtrain04_<arm>.log. Gives up after 20 h.
cd /workspace/clear-ec-research/experiments/lane_cd_20261108/source || exit 1
A=/workspace/clear-ec-research/experiments/lane_cd_20261108/arms; export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
say() { echo "$(date -u +%FT%TZ) $*" >> ../logs/auto_cdtrain04_read.log; }
say "start pid $$"; t0=$(date +%s); d1=0; d3=0
while [ $(( $(date +%s) - t0 )) -lt 72000 ]; do
  for arm in cdonly_ra cdonly_elr3e5_ra; do
    [ $arm = cdonly_ra ] && [ $d1 = 1 ] && continue; [ $arm = cdonly_elr3e5_ra ] && [ $d3 = 1 ] && continue
    if [ -f $A/${arm}_f0_s46/raw/runs/fold0/cpdino_b8/predictions_final_tta.npz ] && [ -f $A/${arm}_f4_s46/raw/runs/fold4/cpdino_b8/predictions_final_tta.npz ]; then
      sleep 60; say "$arm folds 0+4 complete; reading"
      SYN_A="lane_cd_20261108/arms/${arm}_f{f}_s46" SYN_A_TRACK=raw SYN_A_FAM=cpdino_b8 timeout 3000 /workspace/venv/bin/python cd_synpre_inplace_20261111.py > ../logs/cdtrain04_${arm}.log 2>&1
      say "$arm read rc $?"; [ $arm = cdonly_ra ] && d1=1 || d3=1
    fi
  done
  [ $d1 = 1 ] && [ $d3 = 1 ] && { say "all done"; exit 0; }
  sleep 300
done
say "gave up after 20 h"
