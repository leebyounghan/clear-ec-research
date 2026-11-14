#!/bin/bash
# D-SYN-004 CD read (CPU pod): when runs/synthft_vitl_f{0..4}_s46 all have exit.json, run the strict in-place read on all five folds
# (cd_synpre_inplace_20261111.py with SYN_V, incumbent recomputed in the same run) -> logs/synpre_vitl_5f_read.log. Gives up after 12 h.
cd /workspace/clear-ec-research/experiments/lane_cd_20261108/source || exit 1
B=/workspace/clear-ec-research/experiments/lane_backbone_20261108/runs; export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
say() { echo "$(date -u +%FT%TZ) $*" >> ../logs/auto_synvitl5.log; }
say "start pid $$"; t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt 43200 ]; do
  all=1; for f in 0 1 2 3 4; do [ -f $B/synthft_vitl_f${f}_s46/exit.json ] || all=0; done
  if [ $all = 1 ]; then
    say "all five synthetic ViT-L folds complete; reading"
    SYN_V="lane_backbone_20261108/runs/synthft_vitl_f{f}_s46" SYN_V_TRACK=raw SYN_V_FAM=dinov3_l16t timeout 3000 /workspace/venv/bin/python cd_synpre_inplace_20261111.py > ../logs/synpre_vitl_5f_read.log 2>&1
    say "read rc $?"; exit 0
  fi
  sleep 300
done
say "gave up after 12 h"
