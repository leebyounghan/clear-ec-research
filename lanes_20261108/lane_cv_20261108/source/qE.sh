#!/bin/bash
# CV lane GPU 2: DINOv3 B/16 on a 544x412 crop at the legible window (CROP_LW=1), recipe loss, seed 46, SSL reused; folds 0,4 then 1,2,3
cd /workspace/clear-ec-research
L=experiments/lane_cv_20261108; PY=/workspace/venv/bin/python
for F in 0 4 1 2 3; do
  case $F in 0|4) SSL=experiments/lane_gpu3_20261106/runs/dinov3_b16_f${F}_s46/ssl/last.pt;; *) SSL=experiments/lane_cd_20261108/arms/dinov3_b16_f${F}_s46/ssl/last.pt;; esac
  name=dinov3_lw_f${F}_s46; [ -f $L/arms/$name/runs/fold$F/dinov3_b16/complete.json ] && continue
  echo "$(date -u) start $name"
  env CROP_LW=1 CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY $L/source/family_cv_20261108.py all --family dinov3_b16 --fold $F --seed 46 --ssl-from $SSL --root $L/arms/$name --workers 3 > $L/logs/$name.log 2>&1
  echo "$(date -u) end $name rc $?"
done
echo QEDONE
