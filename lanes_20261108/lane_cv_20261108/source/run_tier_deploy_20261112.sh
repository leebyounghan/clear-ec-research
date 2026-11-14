#!/bin/bash
# D-PKG-002: deploy parameter set for one CV tier (CPU pod). usage: run_tier_deploy_20261112.sh CONFIG DIR
cd /workspace/clear-ec-research/experiments/lane_cv_20261108 || exit 1
export CV_OOF=cv_tier_$1_20261112 CV_DEPLOY_DIR=$2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2; PY=/workspace/venv/bin/python
test -e $2 && { echo "$2 exists, not overwriting"; exit 1; }
for step in ids lam fit inputs insample; do timeout 1800 $PY source/cv_deploy_fit_20261110.py $step || { echo "step $step failed"; exit 1; }; done
timeout 1800 $PY source/l5_tier_check_20261112.py || exit 1
echo ALLDONE
