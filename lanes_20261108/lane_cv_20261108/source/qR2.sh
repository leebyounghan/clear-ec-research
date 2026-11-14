#!/bin/bash
# CV lane: DINOv3 B/16 with the noise-weighted pairwise order loss on CV (CD_LOSS=order), seed 46, SSL reused; folds 0,4 (screen) then 1,2,3; after ViT-H+ folds 4,2 for the CD lane (qV.sh); version 2 restart after sharing slowed ViT-H+ 2x
cd /workspace/clear-ec-research
L=experiments/lane_cv_20261108; PY=/workspace/venv/bin/python
until grep -q QVDONE $L/logs/qV.log; do sleep 120; done
for F in 0 4 1 2 3; do
  case $F in 0|4) SSL=experiments/lane_gpu3_20261106/runs/dinov3_b16_f${F}_s46/ssl/last.pt;; *) SSL=experiments/lane_cd_20261108/arms/dinov3_b16_f${F}_s46/ssl/last.pt;; esac
  name=dinov3_order_f${F}_s46
  if [ ! -f $L/arms/$name/runs/fold$F/dinov3_b16/complete.json ]; then
    echo "$(date -u) start $name"; env CD_LOSS=order CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY $L/source/family_cv_20261108.py all --family dinov3_b16 --fold $F --seed 46 --ssl-from $SSL --root $L/arms/$name --workers 3 > $L/logs/$name.log 2>&1; echo "$(date -u) end $name rc $?"
  fi
done
echo QR2DONE
