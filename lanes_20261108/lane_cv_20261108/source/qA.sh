#!/bin/bash
# CV lane recipe A queue: CV-weighted loss arms, folds 0 then 4
cd /workspace/clear-ec-research
L=experiments/lane_cv_20261108
LOSS_W=0.25,2.5,0.25 bash $L/source/run_arm_cv.sh cvw 0 46 20 10 recipeA_cvloss_20261108.py
LOSS_W=0.25,2.5,0.25 bash $L/source/run_arm_cv.sh cvw 4 46 20 10 recipeA_cvloss_20261108.py
echo QADONE
