#!/bin/bash
# CV lane, GPU 2 (PI-approved 2026-11-10): DINOv3 B/16 member trained on the cell-area SD (TARGET_SD=1: column 1 = CV x 1e6 / CD) instead of CV,
# recipe loss, seed 46, SSL reused (same recipe as the DINOv3 member otherwise); folds 0,4 (screen) then 1,2,3. Runs after qR6.sh, ahead of s47 full data and s49.
cd /workspace/clear-ec-research
L=experiments/lane_cv_20261108; PY=/workspace/venv/bin/python
until grep -q QR6DONE $L/logs/qR6.log; do sleep 120; done
for F in 0 4 1 2 3; do
  case $F in 0|4) SSL=experiments/lane_gpu3_20261106/runs/dinov3_b16_f${F}_s46/ssl/last.pt;; *) SSL=experiments/lane_cd_20261108/arms/dinov3_b16_f${F}_s46/ssl/last.pt;; esac
  name=dinov3_sd_f${F}_s46
  if [ ! -f $L/arms/$name/runs/fold$F/dinov3_b16/complete.json ]; then
    echo "$(date -u) start $name"; env TARGET_SD=1 CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY $L/source/family_cv_20261108.py all --family dinov3_b16 --fold $F --seed 46 --ssl-from $SSL --root $L/arms/$name --workers 3 > $L/logs/$name.log 2>&1; rc=$?
    echo "$(date -u) end $name rc $rc"
  fi
done
echo QSDDONE
