#!/bin/bash
# CV lane: frozen heads on DINOv3 B/16 token statistics, folds 0 and 4 first (screen), then 1-3 when their encoders exist
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 FEAT=dino
PY=/workspace/venv/bin/python
until [ -f frozen_dino_f4/meta.json ]; do sleep 30; done
date; LOSS_W=1,1,1 $PY source/frozen_cv_head_20261108.py dino std,spectrum,cov_sqrt,autocorr --folds 0,4 > logs/fz_dino_04.log 2>&1; echo "dino 04 rc $?"
date; echo GFZ3DONE
