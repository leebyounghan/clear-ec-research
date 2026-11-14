#!/bin/bash
# CV lane experiment for a borrowed GPU (coordinator 2026-11-10): order-loss DINOv2 B/14 (family runner, CD_LOSS=order), seed 46, five folds, then full data.
# Folds 0 and 4 reuse lane GPU 3's DINOv2 SSL; folds 1-3 and full data run their own 5-epoch SSL (~40 min each). usage: GPU=<id> bash experiments/lane_cv_20261108/source/qD2O.sh
cd /workspace/clear-ec-research
L=experiments/lane_cv_20261108; PY=/workspace/venv/bin/python; G=${GPU:?set GPU}
for F in 0 4 1 2 3 99; do
  case $F in 0|4) SSLARG="--ssl-from experiments/lane_gpu3_20261106/runs/dinov2_b14_f${F}_s46/ssl/last.pt";; *) SSLARG="";; esac
  name=dinov2_order_f${F}_s46
  if [ ! -f $L/arms/$name/runs/fold$F/dinov2_b14/complete.json ]; then
    echo "$(date -u) start $name"; T0=$(date +%s)
    env CD_LOSS=order CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY $L/source/family_cv_20261108.py all --family dinov2_b14 --fold $F --seed 46 $SSLARG --root $L/arms/$name --workers 3 > $L/logs/$name.log 2>&1; rc=$?
    echo "$(date -u) end $name rc $rc seconds $(( $(date +%s) - T0 ))"
  fi
done
echo QD2ODONE
