#!/bin/bash
# CV lane, GPU 2, after qR7.sh: third order-loss DINOv3 seed (49), folds 0 and 4 only (folds 1-3 run on GPU 3 by the backbone lane, coordinator 2026-11-10), then its full-data build. Coordinator 2026-11-10: the continuation
# if the s47 five-fold read holds (replace this queue while it is still waiting if it does not). Skips if it would start after 2026-11-11 11:00 UTC.
cd /workspace/clear-ec-research
L=experiments/lane_cv_20261108; PY=/workspace/venv/bin/python
until grep -q QR7DONE $L/logs/qR7.log; do sleep 120; done
if [ $(date -u +%s) -gt $(date -u -d 2026-11-11T11:00:00Z +%s) ]; then echo "$(date -u) too late, skipped"; echo QR8BDONE; exit 0; fi
ENV="CD_LOSS=order CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2"
for F in 0 4; do
  case $F in 0|4) SSL=experiments/lane_gpu3_20261106/runs/dinov3_b16_f${F}_s46/ssl/last.pt;; *) SSL=experiments/lane_cd_20261108/arms/dinov3_b16_f${F}_s46/ssl/last.pt;; esac
  name=dinov3_order_f${F}_s49
  if [ ! -f $L/arms/$name/runs/fold$F/dinov3_b16/complete.json ]; then
    echo "$(date -u) start $name"; env $ENV $PY $L/source/family_cv_20261108.py all --family dinov3_b16 --fold $F --seed 49 --ssl-from $SSL --root $L/arms/$name --workers 3 > $L/logs/$name.log 2>&1; rc=$?
    echo "$(date -u) end $name rc $rc"
  fi
done
name=dinov3_order_f99_s49
if [ ! -f $L/arms/$name/runs/fold99/dinov3_b16/complete.json ]; then
  echo "$(date -u) start $name"; T0=$(date +%s)
  env $ENV $PY $L/source/family_cv_20261108.py all --family dinov3_b16 --fold 99 --seed 49 --ssl-from experiments/lane_cd_20261108/arms/dinov3_b16_f99_s46/ssl/last.pt --root $L/arms/$name --workers 3 > $L/logs/$name.log 2>&1; rc=$?
  echo "$(date -u) end $name rc $rc seconds $(( $(date +%s) - T0 ))"
fi
echo QR8BDONE
