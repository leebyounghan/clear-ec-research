#!/bin/bash
# D-CD-TRAIN-SCREEN: wait for the five fold-0 runs, then four-flip TTA over kept epochs 16-20 for P/A/B/C (one GPU each), then the CPU read.
cd /workspace/clear-ec-research || exit 1
A=experiments/lane_cd_20261108/arms; LOG=experiments/lane_cd_20261108/logs; PY=/workspace/venv/bin/python; T=experiments/lane_cd_20261108/source/family_tta_20261108.py
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG/auto_screen_read.log; }
say "start pid $$"; t0=$(date +%s)
done_run() { [ -f $A/screen_$1_f0_s46/$2/runs/fold0/dinov3_b16/predictions_final_tta.npz ]; }
while [ $(( $(date +%s) - t0 )) -lt 10800 ]; do
  all=1; for x in plain:raw llrd075:raw wd005:raw appear05:raw swa6e12:swa6; do done_run ${x%%:*} ${x##*:} || all=0; done
  [ $all = 1 ] && break; sleep 60
done
say "runs complete; TTA e16-20"
g=0; for nm in plain llrd075 wd005 appear05; do
  (CUDA_VISIBLE_DEVICES=$g timeout 1800 $PY $T $A/screen_${nm}_f0_s46/raw dinov3_b16 0 46 screen_${nm}_f0 > $LOG/screen_tta_${nm}.log 2>&1) & g=$((g + 1))
done; wait
say "TTA done; reading"
cd experiments/lane_cd_20261108/source && OMP_NUM_THREADS=4 timeout 1200 $PY cd_screen_read_20261113.py > ../logs/screen_read_20261113.log 2>&1
say "read rc $?"
