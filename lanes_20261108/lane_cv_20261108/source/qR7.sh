#!/bin/bash
# CV lane, GPU 2, after qR6.sh (order-loss DINOv3 s47 five folds): full-data order-loss DINOv3 seed 47 (fold 99, CD lane full-data SSL),
# needed if the two-seed order member (s46 + s47) holds on five folds.
cd /workspace/clear-ec-research
L=experiments/lane_cv_20261108; PY=/workspace/venv/bin/python
until grep -q QR6DONE $L/logs/qR6.log; do sleep 120; done
name=dinov3_order_f99_s47
if [ ! -f $L/arms/$name/runs/fold99/dinov3_b16/complete.json ]; then
  echo "$(date -u) start $name"; T0=$(date +%s)
  env CD_LOSS=order CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY $L/source/family_cv_20261108.py all --family dinov3_b16 --fold 99 --seed 47 --ssl-from experiments/lane_cd_20261108/arms/dinov3_b16_f99_s46/ssl/last.pt --root $L/arms/$name --workers 3 > $L/logs/$name.log 2>&1; rc=$?
  echo "$(date -u) end $name rc $rc seconds $(( $(date +%s) - T0 ))"
fi
echo QR7DONE
