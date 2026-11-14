#!/bin/bash
# Seed audit of the HEX chain (coordinator 2026-11-11 05:00 EST): cvw3 (CV-weighted DINOv3-B) seed 47 on five folds, same recipe as the
# CV lane's s46 arms (SSL init = the fold's s46 SSL, micro 16, schedule 10 epochs) but stopped at epoch 12 (the chain reads epochs 8-12;
# the full-data build used the same --epochs 12 --schedule-epochs 10). usage: seed_audit_queue.sh GPU FOLD [FOLD ...]  (99 = full data)
cd /workspace/clear-ec-research
L=experiments/lane_hex_20261108; PY=/workspace/venv/bin/python; G=$1; shift; Q=$L/logs/seed_audit_gpu$G.log
for f in "$@"; do
  case $f in
    0|4) SSL=experiments/lane_gpu3_20261106/runs/dinov3_b16_f${f}_s46/ssl/last.pt;;
    1|2|3|99) SSL=experiments/lane_cd_20261108/arms/dinov3_b16_f${f}_s46/ssl/last.pt;;
  esac
  if [ "$f" = 99 ]; then R=$L/deploy/cvw3_f99_s47; else R=$L/runs/cvw3_f${f}_s47; fi
  echo "$(date -u +%FT%TZ) start fold $f gpu $G root $R" >> $Q; t0=$(date +%s)
  env CD_LOSS=cvw3 CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 $PY $L/source/family_cvw3_copy.py all --family dinov3_b16 --fold $f --seed 47 --epochs 12 --schedule-epochs 10 --micro 16 --workers 4 --ssl-from $SSL --root $R > $L/logs/cvw3_f${f}_s47.log 2>&1
  echo "$(date -u +%FT%TZ) end fold $f rc $? seconds $(( $(date +%s) - t0 ))" >> $Q
done
echo "$(date -u +%FT%TZ) queue done" >> $Q
