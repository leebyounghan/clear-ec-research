#!/bin/bash
# CV lane (coordinator 2026-11-11): DINOv3 B/16 member with label-distribution smoothing on CV (CD_LOSS=lds: per-sample weight = 1 / Gaussian-KDE
# density of the training rows' log CV, Silverman bandwidth, mean 1, cap 5), seed 46, SSL reused; otherwise the standard member recipe.
# usage: GPU=<id> FOLDS="0 4" bash experiments/lane_cv_20261108/source/qLDS.sh   (KG idea-imbalanced-regression-cv-tails)
cd /workspace/clear-ec-research
L=experiments/lane_cv_20261108; PY=/workspace/venv/bin/python; G=${GPU:?set GPU}
for F in ${FOLDS:-0 4 1 2 3}; do
  case $F in 0|4) SSL=experiments/lane_gpu3_20261106/runs/dinov3_b16_f${F}_s46/ssl/last.pt;; *) SSL=experiments/lane_cd_20261108/arms/dinov3_b16_f${F}_s46/ssl/last.pt;; esac
  name=dinov3_lds_f${F}_s46
  if [ ! -f $L/arms/$name/runs/fold$F/dinov3_b16/complete.json ]; then
    echo "$(date -u) start $name"; env CD_LOSS=lds CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY $L/source/family_cv_20261108.py all --family dinov3_b16 --fold $F --seed 46 --ssl-from $SSL --root $L/arms/$name --workers 3 > $L/logs/$name.log 2>&1; rc=$?
    echo "$(date -u) end $name rc $rc"
  fi
done
echo "QLDSDONE ${FOLDS:-all}"
