#!/bin/bash
# Coordinator 14:47 EST 11-14 (second message): ViT-L s46 encoder LR 3e-5, folds 0 / 4, identical to arms/sweep_vitl_elr2e5_f{0,4}_s46
# except --enc-lr 3e-5 (bb_runner_20261108, same SSL inits, 12 epochs, schedule 10, micro 8, workers 2, EMA 0.999/0.9995 on f0 and
# 0.998/0.999 on f4, SWA from 6, keep 8-12, --tta-final, seed 46). Fold 0 on GPU 0, fold 4 on GPU 2, through the 60 GB guard. Decision-only.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; PY=/workspace/venv/bin/python; RB=experiments/lane_backbone_20261108/runs; R=experiments/lane_backbone_20261108/source/bb_runner_20261108.py; G=experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh
C="--family dinov3_l16t --seed 46 --epochs 12 --schedule-epochs 10 --micro 8 --workers 2 --swa-start 6 --tta-final --keep-epochs 8,9,10,11,12 --enc-lr 3e-5"
(OMP_NUM_THREADS=2 setsid nohup bash $G 0 60000 "$PY $R $C --fold 0 --ssl $RB/vitl16_avg_f0_s46/ssl/last.pt --ema 0.999,0.9995 --root $L/arms/sweep_vitl_elr3e5_f0_s46" > $L/logs/sweep_vitl_elr3e5_f0.log 2>&1 < /dev/null &)
(OMP_NUM_THREADS=2 setsid nohup bash $G 2 60000 "$PY $R $C --fold 4 --ssl $RB/vitl16_avg_f4_s46/ssl/last.pt --ema 0.998,0.999 --root $L/arms/sweep_vitl_elr3e5_f4_s46" > $L/logs/sweep_vitl_elr3e5_f4.log 2>&1 < /dev/null &)
echo "$(date) ViT-L elr3e5 f0 (GPU 0) / f4 (GPU 2) handed to the guard" >> $L/logs/launch_vitl_elr3e5.log
