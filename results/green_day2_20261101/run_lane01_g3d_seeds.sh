#!/bin/bash
# lane01 follow-up: GPU 0 = lambda-3 arm fold 4 (lane23 skips it when it gets there), then pre-only seed 50 (pretrain + TTA, folds 0-4);
# GPU 1 = pre-only seed 49 (pretrain + TTA, folds 0-4). All evidence-only members for the hex6 member-count curve.
cd /workspace/clear-ec-research
PY=/workspace/venv/bin/python; S=experiments/green_day2_20261101/source/green_day2_fullframe_20261101.py; E=experiments/green_track_20261028/source/green_external_srep_20261030.py; G3=experiments/green_day2_20261101/source/green_day2_g3_hex6aux_20261101.py
LOG=experiments/green_day2_20261101/preonly_seeds; mkdir -p $LOG
seed_items() { gpu=$1; seed=$2; for f in 0 1 2 3 4; do
  if [ ! -f experiments/green_track_20261028/external_srep/s$seed/fold$f/sg2_pre.pt ]; then CUDA_VISIBLE_DEVICES=$gpu $PY -u $E pretrain --fold $f --seed $seed --tag s$seed --updates 2000 >> $LOG/pretrain_gpu$gpu.log 2>&1 || { echo "FAILED pretrain s$seed f$f" >> $LOG/pretrain_gpu$gpu.log; continue; }; fi
  if [ ! -f experiments/green_day2_20261101/preonly_tta/s$seed/fold$f/fold${f}_sg2_peaks.npz ]; then CUDA_VISIBLE_DEVICES=$gpu $PY -u $S infer-pre-tta --family s$seed --fold $f >> $LOG/infer_gpu$gpu.log 2>&1 || echo "FAILED infer s$seed f$f" >> $LOG/infer_gpu$gpu.log; fi
done; }
( if [ ! -f experiments/green_day2_20261101/g3_hex6aux/aux3/fold4/predictions_tta4.npz ]; then CUDA_VISIBLE_DEVICES=0 $PY -u $G3 train --fold 4 --arm aux3 --epochs 2 >> experiments/green_day2_20261101/g3_hex6aux/train_aux3_gpu0.log 2>&1 || echo "FAILED aux3 fold 4" >> experiments/green_day2_20261101/g3_hex6aux/train_aux3_gpu0.log; fi; seed_items 0 50 ) &
seed_items 1 49 &
wait
echo "lane01 g3d+seeds done $(date -u)"
