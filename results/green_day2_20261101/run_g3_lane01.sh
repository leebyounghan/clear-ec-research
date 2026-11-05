#!/bin/bash
# G3 folds 2-4, both arms, GPUs 0 and 1 (lane01 job).
cd /workspace/clear-ec-research
PY=/workspace/venv/bin/python; S=experiments/green_day2_20261101/source/green_day2_g3_hex6aux_20261101.py; LOG=experiments/green_day2_20261101/g3_hex6aux; mkdir -p $LOG
while [ ! -f experiments/green_day2_20261101/d2b_active_gate/active_family_features.npz ]; do sleep 60; done
run_items() { gpu=$1; shift; for item in "$@"; do arm=${item%%:*}; f=${item##*:}; if [ -f $LOG/$arm/fold$f/predictions_tta4.npz ]; then continue; fi; CUDA_VISIBLE_DEVICES=$gpu $PY -u $S train --fold $f --arm $arm --epochs 2 >> $LOG/train_gpu$gpu.log 2>&1 || echo "FAILED $item" >> $LOG/train_gpu$gpu.log; done; }
run_items 0 aux:2 control:2 aux:4 &
run_items 1 aux:3 control:3 control:4 &
wait
echo "g3 lane01 done $(date -u)"
