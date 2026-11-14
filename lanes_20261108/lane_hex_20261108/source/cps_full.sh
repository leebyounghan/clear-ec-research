#!/bin/bash
# HEX lane (2026-11-10): full-data CPSAM 512, recipe multi-target loss, no SSL (the mtcnn5_fix member, kept 11-10), fold 99, epochs
# 1-12 (checkpoints 8-12 kept, the member is the mean log over epochs 8-12). Waits for pid arg 1 (gpu1_deploy_queue_v2.sh) first.
# STOP_CPS_FULL skips it. Log: logs/gpu1_deploy_queue.log. Do not edit while it runs.
L=/workspace/clear-ec-research/experiments/lane_hex_20261108; Q=$L/logs/gpu1_deploy_queue.log; cd /workspace/clear-ec-research
P=$1; echo "$(date -u +%FT%TZ) cps_full: waiting for pid $P" >> $Q; while [ -d /proc/$P ]; do sleep 30; done
[ -f $L/STOP_CPS_FULL ] && { echo "$(date -u +%FT%TZ) STOP_CPS_FULL found, not started" >> $Q; exit 0; }
echo "$(date -u +%FT%TZ) start cps_full_f99" >> $Q; t0=$(date +%s)
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 /workspace/venv/bin/python $L/source/mtcps.py all --family cpsam_c512 --fold 99 \
  --root experiments/lane_hex_20261108/deploy/mtcps_c512_f99 --ssl-epochs 0 --epochs 12 --schedule-epochs 10 > $L/logs/deploy_cps_full_f99.log 2>&1
echo "$(date -u +%FT%TZ) end cps_full_f99 rc $? seconds $(( $(date +%s) - t0 ))" >> $Q
