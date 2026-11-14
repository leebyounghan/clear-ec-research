#!/bin/bash
# CV lane: four-flip TTA (epochs 16-20) for the CV-weighted DINOv3 seed-46 members, then DINOv3 B/16 recipe seed 49 on five folds (SSL reused) with TTA
cd /workspace/clear-ec-research
L=experiments/lane_cv_20261108; PY=/workspace/venv/bin/python
for F in 0 4 1 2 3; do
  R=$L/arms/dinov3_cvw3_f${F}_s46; until [ -f $R/runs/fold$F/dinov3_b16/complete.json ]; do sleep 120; done
  echo "$(date -u) tta cvw3 f$F"; CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 $PY $L/source/family_tta_cv_20261108.py $R dinov3_b16 $F 46 cvw3_s46_f$F > $L/logs/tta_cvw3_f$F.log 2>&1
done
for F in 0 4 1 2 3; do
  case $F in 0|4) SSL=experiments/lane_gpu3_20261106/runs/dinov3_b16_f${F}_s46/ssl/last.pt;; *) SSL=experiments/lane_cd_20261108/arms/dinov3_b16_f${F}_s46/ssl/last.pt;; esac
  name=dinov3_b16_f${F}_s49
  if [ ! -f $L/arms/$name/runs/fold$F/dinov3_b16/complete.json ]; then
    echo "$(date -u) start $name"; env CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY $L/source/family_cv_20261108.py all --family dinov3_b16 --fold $F --seed 49 --ssl-from $SSL --root $L/arms/$name --workers 3 > $L/logs/$name.log 2>&1; echo "$(date -u) end $name rc $?"
  fi
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 $PY $L/source/family_tta_cv_20261108.py $L/arms/$name dinov3_b16 $F 49 dinov3_s49_f$F > $L/logs/tta_s49_f$F.log 2>&1; echo "$(date -u) tta s49 f$F rc $?"
done
echo QTDONE
