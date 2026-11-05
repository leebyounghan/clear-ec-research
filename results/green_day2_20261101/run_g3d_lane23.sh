#!/bin/bash
# G3d: lambda-3 auxiliary arm, folds 0-4, GPUs 2 and 3 (lane23 job).
cd /workspace/clear-ec-research
PY=/workspace/venv/bin/python; S=experiments/green_day2_20261101/source/green_day2_g3_hex6aux_20261101.py; LOG=experiments/green_day2_20261101/g3_hex6aux; mkdir -p $LOG
run_items() { gpu=$1; shift; for f in "$@"; do if [ -f $LOG/aux3/fold$f/predictions_tta4.npz ]; then continue; fi; CUDA_VISIBLE_DEVICES=$gpu $PY -u $S train --fold $f --arm aux3 --epochs 2 >> $LOG/train_aux3_gpu$gpu.log 2>&1 || echo "FAILED aux3 fold $f" >> $LOG/train_aux3_gpu$gpu.log; done; }
run_items 2 0 2 4 &
run_items 3 1 3 &
wait
echo "g3d done $(date -u)"
