#!/bin/bash
# E10 route B (lane01 job after the parent's meta_bridge job): wait for the 3000 synthetic images, then domain check (GPU 1) in parallel with training (GPU 0), then score25 (GPU 0).
cd /workspace/clear-ec-research
PY=/workspace/venv/bin/python; S=experiments/green_day3_20261103/source/green_day3_e10_synth_20261103.py; LOG=experiments/green_day3_20261103/e10_synth; mkdir -p $LOG
export E10_DATA=$(cat $LOG/DATA_PATH 2>/dev/null || echo experiments/meta_ceiling_20261103/synth_data/v1)
for i in $(seq 1 120); do n=$(ls $E10_DATA/images/*.png 2>/dev/null | wc -l); [ "$n" -ge 3000 ] && break; sleep 60; done
echo "data $E10_DATA images available: $(ls $E10_DATA/images/*.png | wc -l) $(date -u)" >> $LOG/driver.log
[ -f $E10_DATA/COMPLETE ] && touch $LOG/GO_TRAIN && echo "GO_TRAIN created from $E10_DATA/COMPLETE $(date -u)" >> $LOG/driver.log
CUDA_VISIBLE_DEVICES=1 $PY -u $S domain-check --limit 24 --tag v5 > $LOG/domain_check_v5.log 2>&1 &
if [ -f $LOG/GO_TRAIN ]; then
  [ -f $LOG/detector_synth.pt ] || CUDA_VISIBLE_DEVICES=0 $PY -u $S train --updates 4000 > $LOG/train.log 2>&1 || { echo "FAILED train" >> $LOG/driver.log; }
else
  echo "GO_TRAIN absent: training held until the renderer is confirmed $(date -u)" >> $LOG/driver.log
fi
wait
if [ -f $LOG/detector_synth.pt ]; then CUDA_VISIBLE_DEVICES=0 $PY -u $S score25 > $LOG/score25.log 2>&1 || echo "FAILED score25" >> $LOG/driver.log; fi
echo "e10 done $(date -u)" >> $LOG/driver.log
