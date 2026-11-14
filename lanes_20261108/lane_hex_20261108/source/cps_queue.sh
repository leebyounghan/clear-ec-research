#!/bin/bash
# cps_queue.sh WAITPID : HEX-weighted CPSAM 512 without SSL, folds 4, 0, 1, 2, 3 in turn on GPU 1 (coordinator decision (a),
# 2026-11-08). Waits for WAITPID (the HEX-weighted SSL fold-4 run) to exit first. Before each fold, a file STOP_CPS_QUEUE in
# the lane folder ends the queue (used if the fold 0+4 read does not lower HEX). Log: logs/cps_queue.log. (HEX lane)
L=/workspace/clear-ec-research/experiments/lane_hex_20261108; Q=$L/logs/cps_queue.log
echo "$(date -u +%FT%TZ) waiting for pid $1" >> $Q
while kill -0 $1 2>/dev/null; do sleep 60; done
cd /workspace/clear-ec-research
for f in 4 0 1 2 3; do
  if [ -f $L/STOP_CPS_QUEUE ]; then echo "$(date -u +%FT%TZ) STOP_CPS_QUEUE found, ending before fold $f" >> $Q; exit 0; fi
  echo "$(date -u +%FT%TZ) start no-SSL fold $f" >> $Q
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 /workspace/venv/bin/python $L/source/hexcps.py all --family cpsam_c512 --fold $f \
    --root experiments/lane_hex_20261108/runs/hexcps_nossl_c512_f$f --ssl-epochs 0 --epochs 12 --schedule-epochs 10 > $L/logs/hexcps_nossl_f$f.log 2>&1
  echo "$(date -u +%FT%TZ) end no-SSL fold $f exit $?" >> $Q
done
echo "$(date -u +%FT%TZ) all folds done" >> $Q
