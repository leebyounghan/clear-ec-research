#!/bin/bash
# CV lane experiment for a borrowed GPU (coordinator 2026-11-10): order-loss ViT-L/16 (timm DINOv3 ViT-L, CD lane recipe), seed 46, five folds, SSL reused,
# then the full-data build. usage (from anywhere): GPU=<id> bash experiments/lane_cv_20261108/source/qVLO.sh   (writes only under lane_cv_20261108)
# Per fold ~2.1 h (CD lane measured regression time for this recipe); full data needs lane_cd_20261108/arms/vitl16_f99_s46/ssl/last.pt (CD lane full-data SSL).
cd /workspace/clear-ec-research
L=experiments/lane_cv_20261108; C=experiments/lane_cd_20261108; BB=experiments/lane_backbone_20261108; PY=/workspace/venv/bin/python; G=${GPU:?set GPU}
ARGS="--family dinov3_l16t --seed 46 --epochs 12 --micro 8 --ema 0.998,0.999 --swa-start 6 --tta-final --keep-epochs 8,9,10,11,12 --workers 4"
for F in 0 4 1 2 3; do
  case $F in 0|4) SSL=$BB/runs/vitl16_avg_f${F}_s46/ssl/last.pt;; *) SSL=$C/arms/vitl16_f${F}_s46/ssl/last.pt;; esac
  R=$L/arms/vitl16_order_f${F}_s46
  if [ ! -f $R/raw/runs/fold$F/dinov3_l16t/complete.json ]; then
    echo "$(date -u) start vitl16_order f$F"; CUDA_VISIBLE_DEVICES=$G $PY $L/source/bb_runner_order_cv_20261110.py --fold $F --root $R --ssl $SSL $ARGS > $L/logs/vitl16_order_f${F}_s46.log 2>&1; rc=$?
    echo "$(date -u) end vitl16_order f$F rc $rc"
  fi
done
SSL99=$C/arms/vitl16_f99_s46/ssl/last.pt; R=$L/arms/vitl16_order_f99_s46
until [ -f $SSL99 ]; do echo "$(date -u) waiting for $SSL99"; sleep 600; done
if [ ! -f $R/raw/runs/fold99/dinov3_l16t/complete.json ]; then
  echo "$(date -u) start vitl16_order f99"; T0=$(date +%s); CUDA_VISIBLE_DEVICES=$G $PY $L/source/bb_runner_order_cv_20261110.py --fold 99 --root $R --ssl $SSL99 $ARGS > $L/logs/vitl16_order_f99_s46.log 2>&1; rc=$?
  echo "$(date -u) end vitl16_order f99 rc $rc seconds $(( $(date +%s) - T0 ))"
fi
echo QVLODONE
