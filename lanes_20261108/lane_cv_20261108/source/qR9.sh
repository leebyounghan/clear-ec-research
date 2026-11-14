#!/bin/bash
# CV lane, GPU 2, after the SD member (qSD.sh): order-loss DINOv3 B/16 seed 49, folds 0,4,1,2,3 (coordinator 2026-11-10: the third seed for the
# pre-registered rule: stack with the 3-seed mean (46, 47, 49) vs the no-order-DINOv3 candidate -0.0411).
cd /workspace/clear-ec-research
L=experiments/lane_cv_20261108; PY=/workspace/venv/bin/python
until grep -q QSDDONE $L/logs/qSD.log; do sleep 120; done
for F in 0 4 1 2 3; do
  case $F in 0|4) SSL=experiments/lane_gpu3_20261106/runs/dinov3_b16_f${F}_s46/ssl/last.pt;; *) SSL=experiments/lane_cd_20261108/arms/dinov3_b16_f${F}_s46/ssl/last.pt;; esac
  name=dinov3_order_f${F}_s49
  if [ ! -f $L/arms/$name/runs/fold$F/dinov3_b16/complete.json ]; then
    echo "$(date -u) start $name"; env CD_LOSS=order CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY $L/source/family_cv_20261108.py all --family dinov3_b16 --fold $F --seed 49 --ssl-from $SSL --root $L/arms/$name --workers 3 > $L/logs/$name.log 2>&1; rc=$?
    echo "$(date -u) end $name rc $rc"
  fi
done
echo QR9DONE
