#!/bin/bash
# CV lane, GPU 2, after order-loss DINOv3 s46 five folds (qR5.sh): (1) full-data order-loss DINOv3 s46 (fold 99, CD lane full-data SSL),
# needed if the five-fold read holds; (2) order-loss DINOv3 seed 47, folds 0,4,1,2,3 (same SSL as s46), a second order member for the DINOv3 mean.
cd /workspace/clear-ec-research
L=experiments/lane_cv_20261108; PY=/workspace/venv/bin/python
until grep -q QR5DONE $L/logs/qR5.log; do sleep 120; done
ENV="CD_LOSS=order CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2"
name=dinov3_order_f99_s46
if [ ! -f $L/arms/$name/runs/fold99/dinov3_b16/complete.json ]; then
  echo "$(date -u) start $name"; T0=$(date +%s)
  env $ENV $PY $L/source/family_cv_20261108.py all --family dinov3_b16 --fold 99 --seed 46 --ssl-from experiments/lane_cd_20261108/arms/dinov3_b16_f99_s46/ssl/last.pt --root $L/arms/$name --workers 3 > $L/logs/$name.log 2>&1; rc=$?
  echo "$(date -u) end $name rc $rc seconds $(( $(date +%s) - T0 ))"
fi
for F in 0 4 1 2 3; do
  case $F in 0|4) SSL=experiments/lane_gpu3_20261106/runs/dinov3_b16_f${F}_s46/ssl/last.pt;; *) SSL=experiments/lane_cd_20261108/arms/dinov3_b16_f${F}_s46/ssl/last.pt;; esac
  name=dinov3_order_f${F}_s47
  if [ ! -f $L/arms/$name/runs/fold$F/dinov3_b16/complete.json ]; then
    echo "$(date -u) start $name"; env $ENV $PY $L/source/family_cv_20261108.py all --family dinov3_b16 --fold $F --seed 47 --ssl-from $SSL --root $L/arms/$name --workers 3 > $L/logs/$name.log 2>&1; rc=$?
    echo "$(date -u) end $name rc $rc"
  fi
done
echo QR6DONE
