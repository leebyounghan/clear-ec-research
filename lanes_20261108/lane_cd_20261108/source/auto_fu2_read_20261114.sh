#!/bin/bash
# CPU pod (06:00 EST 11-14): final read of the DINOv3-B encoder LR 2e-5 follow-up once the e16-20 TTA of folds 1-4 exist
# (cd_fu2_final_20261114.py strict, then cv10). Each starts only while fewer than 2 lane read processes run on this pod (CPU rule).
cd /workspace/clear-ec-research/experiments/lane_cd_20261108/source || exit 1
PY=/workspace/venv/bin/python; T=../tta; say() { echo "$(date -u +%FT%TZ) $*" >> ../logs/auto_fu2_read.log; }
nreads() { ps -eo args= | grep -cE "^[^ ]*python[^ ]* cd_[a-z0-9_]+\.py"; }
say "start pid $$"; t0=$(date +%s); done_s=0; done_c=0
while [ $(( $(date +%s) - t0 )) -lt 50400 ] && { [ $done_s = 0 ] || [ $done_c = 0 ]; }; do
  sleep 120
  for f in 1 2 3 4; do [ -f $T/fu2_elr2e5_f${f}_tta4_e16_20.npz ] || continue 2; done
  if [ $done_s = 0 ] && [ $(nreads) -lt 2 ]; then
    (OMP_NUM_THREADS=2 nice -n 19 timeout 7200 $PY cd_fu2_final_20261114.py strict > ../logs/fu2_final_strict.log 2>&1; echo "$(date -u +%FT%TZ) strict rc $?" >> ../logs/auto_fu2_read.log) &
    done_s=1; say "strict started"; sleep 30; continue
  fi
  if [ $done_c = 0 ] && [ $(nreads) -lt 2 ]; then
    (NR=20 OMP_NUM_THREADS=2 nice -n 19 timeout 14400 $PY cd_fu2_final_20261114.py cv10 > ../logs/fu2_final_cv10.log 2>&1; echo "$(date -u +%FT%TZ) cv10 rc $?" >> ../logs/auto_fu2_read.log) &
    done_c=1; say "cv10 started"
  fi
done
wait; say "exit"
