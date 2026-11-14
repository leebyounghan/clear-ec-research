#!/bin/bash
# HEX lane (2026-11-08): no-SSL multi-target CPSAM 512 (recipe loss, source/mtcps.py) on folds 4, 0, 1, 2, 3, one after another
# on GPU 1. Waits for pid arg 1 (if given) first. STOP_CPS_QUEUE2 ends it before the next fold. Do not edit while it runs.
L=/workspace/clear-ec-research/experiments/lane_hex_20261108; Q=$L/logs/cps_queue2.log; cd /workspace/clear-ec-research
P=${1:-0}; echo "$(date -u +%FT%TZ) waiting for pid $P" >> $Q; while [ "$P" != 0 ] && [ -d /proc/$P ]; do sleep 30; done
for f in 4 0 1 2 3; do
  if [ -f $L/STOP_CPS_QUEUE2 ]; then echo "$(date -u +%FT%TZ) STOP_CPS_QUEUE2 found, ending before fold $f" >> $Q; exit 0; fi
  echo "$(date -u +%FT%TZ) start multi-target no-SSL fold $f" >> $Q
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 /workspace/venv/bin/python $L/source/mtcps.py all --family cpsam_c512 --fold $f \
    --root experiments/lane_hex_20261108/runs/mtcps_nossl_c512_f$f --ssl-epochs 0 --epochs 12 --schedule-epochs 10 > $L/logs/mtcps_nossl_f$f.log 2>&1
  echo "$(date -u +%FT%TZ) fold $f exit $?" >> $Q
done
echo "$(date -u +%FT%TZ) all folds done" >> $Q
