#!/bin/bash
# CV lane job for GPU 3 (backbone lane runs it; coordinator 2026-11-10): order-loss DINOv3 B/16 seed 49, folds 1, 2, 3 (GPU 2 does folds 0, 4 and full data).
# usage: GPU=3 bash experiments/lane_cv_20261108/source/qS49g3.sh    Writes only under lane_cv_20261108. About 1 h per fold alone, ~36 GiB, 3 loader workers.
cd /workspace/clear-ec-research
L=experiments/lane_cv_20261108; PY=/workspace/venv/bin/python; G=${GPU:?set GPU}
for F in 1 2 3; do
  name=dinov3_order_f${F}_s49
  if [ ! -f $L/arms/$name/runs/fold$F/dinov3_b16/complete.json ]; then
    echo "$(date -u) start $name"; env CD_LOSS=order CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY $L/source/family_cv_20261108.py all --family dinov3_b16 --fold $F --seed 49 --ssl-from experiments/lane_cd_20261108/arms/dinov3_b16_f${F}_s46/ssl/last.pt --root $L/arms/$name --workers 3 > $L/logs/$name.log 2>&1; rc=$?
    echo "$(date -u) end $name rc $rc"
  fi
done
echo QS49G3DONE
