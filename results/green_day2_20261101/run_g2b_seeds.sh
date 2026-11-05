#!/bin/bash
# G2b: two more pre-only full-frame detectors (srep pre-training seeds 49, 50; original recipe, 2000 updates) + TTA peaks, folds 0-4, GPUs 2 and 3 (lane23 job).
cd /workspace/clear-ec-research
PY=/workspace/venv/bin/python; E=experiments/green_track_20261028/source/green_external_srep_20261030.py; S=experiments/green_day2_20261101/source/green_day2_fullframe_20261101.py
LOG=experiments/green_day2_20261101/preonly_seeds; mkdir -p $LOG
run_items() { gpu=$1; shift; for item in "$@"; do seed=${item%%:*}; f=${item##*:}
  if [ ! -f experiments/green_track_20261028/external_srep/s$seed/fold$f/sg2_pre.pt ]; then CUDA_VISIBLE_DEVICES=$gpu $PY -u $E pretrain --fold $f --seed $seed --tag s$seed --updates 2000 >> $LOG/pretrain_gpu$gpu.log 2>&1 || { echo "FAILED pretrain $item" >> $LOG/pretrain_gpu$gpu.log; continue; }; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $S infer-pre-tta --family s$seed --fold $f >> $LOG/infer_gpu$gpu.log 2>&1 || echo "FAILED infer $item" >> $LOG/infer_gpu$gpu.log
done; }
run_items 2 49:0 49:2 49:4 50:1 50:3 &
run_items 3 49:1 49:3 50:0 50:2 50:4 &
wait
ls experiments/green_day2_20261101/preonly_tta/s49/fold*/fold*_sg2_peaks.npz experiments/green_day2_20261101/preonly_tta/s50/fold*/fold*_sg2_peaks.npz 2>/dev/null | wc -l
echo "g2b done $(date -u)"
