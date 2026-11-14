#!/bin/bash
# CV lane runs the backbone lane's recipe A window member, folds 1 and 2 together, on GPU 2 now (coordinator reorder 2026-11-10 06:48 UTC).
# Exact commands from the backbone lane; roots in its folder. Fold 3 is the CD lane's; never start it here.
cd /workspace/clear-ec-research
L=experiments/lane_cv_20261108; PY=/workspace/venv/bin/python
run() { F=$1; echo "$(date -u) start win2x fold $F"
  CUDA_VISIBLE_DEVICES=2 $PY experiments/lane_backbone_20261108/source/bb_runner_20261108.py --fold $F --seed 46 --root experiments/lane_backbone_20261108/runs/recipeA_win2x_f${F}_s46 --epochs 12 --window experiments/lane_backbone_20261108/data/window_topbox_20261108.npz --window-size 576,448 --window-scale 2 --window-jitter 32 --ema 0.998,0.999 --swa-start 6 --tta-final --workers 2 > $L/logs/win2x_f${F}_s46.log 2>&1
  echo "$(date -u) end win2x fold $F rc $?"; }
run 1 & sleep 60; run 2 & wait
echo QW3DONE
