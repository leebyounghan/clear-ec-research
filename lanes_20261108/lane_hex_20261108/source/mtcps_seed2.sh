#!/bin/bash
# Seed audit (coordinator 2026-11-11 05:00 EST): mtcnn5_fix member, second seed (47), one fold, same recipe as runs/mtcps_nossl_c512_f$F
# (mtcps.py: recipe multi-target loss, no SSL, 12 epochs, 10-epoch schedule); only --seed changes. Waits for WAIT_PID (0 = none) first.
# usage: mtcps_seed2.sh GPU FOLD [WAIT_PID]   log: logs/mtcps_seed2.log
L=/workspace/clear-ec-research/experiments/lane_hex_20261108; Q=$L/logs/mtcps_seed2.log; cd /workspace/clear-ec-research
G=$1; F=$2; P=${3:-0}
echo "$(date -u +%FT%TZ) fold $F gpu $G: waiting for pid $P" >> $Q; while [ "$P" != 0 ] && [ -d /proc/$P ]; do sleep 30; done
echo "$(date -u +%FT%TZ) start fold $F seed 47 gpu $G" >> $Q; t0=$(date +%s)
CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 /workspace/venv/bin/python $L/source/mtcps.py all --family cpsam_c512 --fold $F --seed 47 \
  --root experiments/lane_hex_20261108/runs/mtcps_nossl_c512_f${F}_s47 --ssl-epochs 0 --epochs 12 --schedule-epochs 10 > $L/logs/mtcps_nossl_f${F}_s47.log 2>&1
echo "$(date -u +%FT%TZ) end fold $F rc $? seconds $(( $(date +%s) - t0 ))" >> $Q
