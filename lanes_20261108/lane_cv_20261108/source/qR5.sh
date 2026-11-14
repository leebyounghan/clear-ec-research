#!/bin/bash
# CV lane: order-loss version 2 (DINOv3 B/16, CD_LOSS=order, seed 46, SSL reused), folds 0,4,1,2,3, after window folds 1 and 2 (qW3.sh, and fold 2 if restarted separately); ViT-H+ fold 2 dropped (coordinator 2026-11-10). Skips if it would start after 2026-11-11 10:00 UTC (five folds must finish before the 23:00 UTC freeze).
cd /workspace/clear-ec-research
L=experiments/lane_cv_20261108; PY=/workspace/venv/bin/python
until grep -q QW3DONE $L/logs/qW3.log; do sleep 120; done
while ps -eo args | grep -q "[r]ecipeA_win2x_f2_s46"; do sleep 120; done
if [ $(date -u +%s) -gt $(date -u -d 2026-11-11T10:00:00Z +%s) ]; then echo "$(date -u) too late for five folds before the freeze, skipped"; echo QR5DONE; exit 0; fi
for F in 0 4 1 2 3; do
  case $F in 0|4) SSL=experiments/lane_gpu3_20261106/runs/dinov3_b16_f${F}_s46/ssl/last.pt;; *) SSL=experiments/lane_cd_20261108/arms/dinov3_b16_f${F}_s46/ssl/last.pt;; esac
  name=dinov3_order_f${F}_s46
  if [ ! -f $L/arms/$name/runs/fold$F/dinov3_b16/complete.json ]; then
    echo "$(date -u) start $name"; env CD_LOSS=order CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY $L/source/family_cv_20261108.py all --family dinov3_b16 --fold $F --seed 46 --ssl-from $SSL --root $L/arms/$name --workers 3 > $L/logs/$name.log 2>&1; echo "$(date -u) end $name rc $?"
  fi
done
echo QR5DONE
