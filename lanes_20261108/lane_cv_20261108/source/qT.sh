#!/bin/bash
# CV lane: four-flip TTA (epochs 16-20) for the CV-weighted DINOv3 seed-46 members, then DINOv3 B/16 recipe seed 49 on five folds (SSL reused) with TTA
cd /workspace/clear-ec-research
L=experiments/lane_cv_20261108; PY=/workspace/venv/bin/python
for F in 0 4 1 2 3; do
  R=/arms/dinov3_cvw3_f_s46; until [ -f /runs/fold/dinov3_b16/complete.json ]; do sleep 120; done
  echo Sun Nov 8 23:20:12 UTC 2026 tta cvw3 f; CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2  /source/family_tta_cv_20261108.py  dinov3_b16  46 cvw3_s46_f > /logs/tta_cvw3_f.log 2>&1
done
for F in 0 4 1 2 3; do
  case  in 0|4) SSL=experiments/lane_gpu3_20261106/runs/dinov3_b16_f_s46/ssl/last.pt;; *) SSL=experiments/lane_cd_20261108/arms/dinov3_b16_f_s46/ssl/last.pt;; esac
  name=dinov3_b16_f_s49
  if [ ! -f /arms//runs/fold/dinov3_b16/complete.json ]; then
    echo Sun Nov 8 23:20:12 UTC 2026 start ; env CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2  /source/family_cv_20261108.py all --family dinov3_b16 --fold  --seed 49 --ssl-from  --root /arms/ --workers 3 > /logs/.log 2>&1; echo Sun Nov 8 23:20:12 UTC 2026 end rc 0
  fi
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2  /source/family_tta_cv_20261108.py /arms/ dinov3_b16  49 dinov3_s49_f > /logs/tta_s49_f.log 2>&1; echo Sun Nov 8 23:20:12 UTC 2026 tta s49 f rc 0
done
echo QTDONE
