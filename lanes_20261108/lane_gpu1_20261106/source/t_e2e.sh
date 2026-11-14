#!/bin/bash
L=/workspace/clear-ec-research/experiments/lane_gpu1_20261106
cp $L/../lane_gpu0_20261106/source/recipeA_single_gpu_20261106.py $L/source/recipeA_single_gpu_20261106.py
grep -n '^MICRO' $L/source/recipeA_single_gpu_20261106.py
# Wrapper test only: the seed 47 root is a test root that stops after smoke, on the existing fold 4 arm.
CUDA_VISIBLE_DEVICES=1 bash $L/source/run_e2e.sh cropopt_both_f4 4 47 smoke 2>&1 | cut -c1-260 | tail -16
