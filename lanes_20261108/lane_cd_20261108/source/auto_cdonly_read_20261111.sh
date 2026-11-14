#!/bin/bash
# D-CDTRAIN-001 read (CPU pod): when both CD-only recipe A runs have their e20 TTA (raw track), run the strict two-fold in-place read
# (cd_synpre_inplace_20261111.py with SYN_A = the CD-only runs) -> logs/cdonly_read.log. Gives up after 20 h.
cd /workspace/clear-ec-research/experiments/lane_cd_20261108/source || exit 1
R=/workspace/clear-ec-research/experiments/lane_cd_20261108/arms; export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
say() { echo "$(date -u +%FT%TZ) $*" >> ../logs/auto_cdonly_read.log; }
say "start pid $$"; t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt 72000 ]; do
  if [ -f $R/cdonly_ra_f0_s46/raw/runs/fold0/cpdino_b8/predictions_final_tta.npz ] && [ -f $R/cdonly_ra_f4_s46/raw/runs/fold4/cpdino_b8/predictions_final_tta.npz ]; then
    sleep 60; say "both CD-only runs have e20 TTA; reading"
    SYN_A="lane_cd_20261108/arms/cdonly_ra_f{f}_s46" SYN_A_TRACK=raw SYN_A_FAM=cpdino_b8 timeout 3000 /workspace/venv/bin/python cd_synpre_inplace_20261111.py > ../logs/cdonly_read.log 2>&1
    say "read rc $?"; exit 0
  fi
  sleep 300
done
say "gave up after 20 h"
