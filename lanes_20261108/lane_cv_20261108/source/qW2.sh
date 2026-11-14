#!/bin/bash
# CV lane runs the backbone lane's recipe A window member, folds 1 and 2, on GPU 2 after ViT-H+ (qV.sh) ends (coordinator 2026-11-10).
# Exact commands from the backbone lane; roots in its folder; one fold at a time. Fold 3 moved to the CD lane (backbone lane change, 2026-11-10); never start it here. Logs in this lane's logs/.
cd /workspace/clear-ec-research
L=experiments/lane_cv_20261108; PY=/workspace/venv/bin/python
until grep -q QVDONE $L/logs/qV.log; do sleep 120; done
for F in 1 2; do
  echo "$(date -u) start win2x fold $F"
  CUDA_VISIBLE_DEVICES=2 $PY experiments/lane_backbone_20261108/source/bb_runner_20261108.py --fold $F --seed 46 --root experiments/lane_backbone_20261108/runs/recipeA_win2x_f${F}_s46 --epochs 12 --window experiments/lane_backbone_20261108/data/window_topbox_20261108.npz --window-size 576,448 --window-scale 2 --window-jitter 32 --ema 0.998,0.999 --swa-start 6 --tta-final --workers 2 > $L/logs/win2x_f${F}_s46.log 2>&1
  echo "$(date -u) end win2x fold $F rc $?"
done
echo QWDONE
