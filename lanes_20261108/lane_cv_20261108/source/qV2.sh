#!/bin/bash
# CV lane runs ViT-H+/16 fold 2 for the CD lane (exact qV.sh commands) after the window folds (qW3.sh). Fold 4 is done; the first fold-2 SSL try was stopped for the reorder and archived.
cd /workspace/clear-ec-research
C=experiments/lane_cd_20261108; L=experiments/lane_cv_20261108; PY=/workspace/venv/bin/python
until grep -q QW3DONE $L/logs/qW3.log; do sleep 120; done
F=2; R=$C/arms/vith16_f${F}_s46
echo "$(date -u) start vith16 f$F ssl"
env CUDA_VISIBLE_DEVICES=2 $PY $C/source/family_bb_cd_20261108.py ssl --family dinov3_hp16t --fold $F --seed 46 --root $R --workers 4 > $L/logs/vith16_f${F}_ssl.log 2>&1
echo "$(date -u) end vith16 f$F ssl rc $?"; echo "$(date -u) start vith16 f$F regression"
env CUDA_VISIBLE_DEVICES=2 $PY $C/source/bb_runner_cd_20261108.py --family dinov3_hp16t --fold $F --seed 46 --root $R --ssl $R/ssl/last.pt --epochs 12 --micro 4 --ema 0.998,0.999 --swa-start 6 --tta-final --workers 4 --keep-epochs 8,9,10,11,12 > $L/logs/vith16_f${F}_reg.log 2>&1
echo "$(date -u) end vith16 f$F regression rc $?"
echo QV2DONE
