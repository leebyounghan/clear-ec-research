#!/bin/bash
# CV lane: CV-weighted frozen heads (own encoders), five folds
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2
PY=/workspace/venv/bin/python
date; LOSS_W=0.25,2.5,0.25 $PY source/frozen_cv_head_20261108.py cvw std,spectrum,cov_sqrt,autocorr > logs/fz_cvw.log 2>&1; echo "cvw rc $?"
date; LOSS_W=0,1,0 $PY source/frozen_cv_head_20261108.py cvo std,spectrum,cov_sqrt > logs/fz_cvo.log 2>&1; echo "cvo rc $?"
date; echo GFZ2DONE
