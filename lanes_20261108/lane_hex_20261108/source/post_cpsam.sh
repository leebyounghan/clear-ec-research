#!/bin/bash
# HEX lane (2026-11-08): when the backbone lane's CPSAM fold 4 (pid arg 1) exits, GPU 1 is ours (coordinator 11-07 23:00 EST):
# lift the throttles (duty 1) and allow 2 queue slots. Only writes files in the lane folder. Do not edit while it runs.
LANE=/workspace/clear-ec-research/experiments/lane_hex_20261108; P=$1
log() { echo "$(date -u +%FT%TZ) $*" >> $LANE/logs/post_cpsam.log; }
log "waiting for CPSAM pid $P"
while [ -d /proc/$P ]; do sleep 60; done
echo 1.0 > $LANE/duty_gpu1.txt; echo 1.0 > $LANE/duty_seg.txt; echo 2 > $LANE/slots_gpu1.txt
log "CPSAM exited; duty 1.0, slots 2"
