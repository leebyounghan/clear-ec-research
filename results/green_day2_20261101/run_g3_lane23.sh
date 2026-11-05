#!/bin/bash
# G3 folds 0-1, both arms, GPUs 2 and 3 (lane23 job). Smoke first.
cd /workspace/clear-ec-research
PY=/workspace/venv/bin/python; S=experiments/green_day2_20261101/source/green_day2_g3_hex6aux_20261101.py; LOG=experiments/green_day2_20261101/g3_hex6aux; mkdir -p $LOG
while [ ! -f experiments/green_day2_20261101/d2b_active_gate/active_family_features.npz ]; do sleep 60; done
CUDA_VISIBLE_DEVICES=2 $PY -u $S train --fold 0 --arm aux --epochs 1 --smoke > $LOG/smoke.log 2>&1 || { echo "smoke failed"; tail -5 $LOG/smoke.log; exit 1; }
rm -rf $LOG/aux/fold0   # smoke output
run_items() { gpu=$1; shift; for item in "$@"; do arm=${item%%:*}; f=${item##*:}; if [ -f $LOG/$arm/fold$f/predictions_tta4.npz ]; then continue; fi; CUDA_VISIBLE_DEVICES=$gpu $PY -u $S train --fold $f --arm $arm --epochs 2 >> $LOG/train_gpu$gpu.log 2>&1 || echo "FAILED $item" >> $LOG/train_gpu$gpu.log; done; }
run_items 2 aux:0 control:0 &
run_items 3 aux:1 control:1 &
wait
echo "g3 lane23 done $(date -u)"
