#!/bin/bash
# Coordinator 13:27 EST 11-14: hard-image probe. ViT-L s46 encoder LR 2e-5 folds 0 / 4, exactly the sweep_vitl_elr2e5_f{0,4}_s46 commands
# (bb_runner_20261108, same SSL, 12 epochs, schedule 10, micro 8, workers 2, EMA 0.999/0.9995 on f0 and 0.998/0.999 on f4, SWA from 6,
# keep 8-12, --tta-final) plus --row-weight data/rowweight_hard2_f<f>_20261114.npz (top decile of member disagreement = 2; a LOSS weight,
# normalised by the mean training weight: 1.818 / 0.909). Fold 0 on GPU 1, fold 4 on GPU 3, alone. Nothing is expanded automatically.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; PY=/workspace/venv/bin/python; RB=experiments/lane_backbone_20261108/runs; R=experiments/lane_backbone_20261108/source/bb_runner_20261108.py
C="--family dinov3_l16t --seed 46 --epochs 12 --schedule-epochs 10 --micro 8 --workers 2 --swa-start 6 --tta-final --keep-epochs 8,9,10,11,12 --enc-lr 2e-5"
(CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 setsid nohup timeout 36000 $PY $R $C --fold 0 --ssl $RB/vitl16_avg_f0_s46/ssl/last.pt --ema 0.999,0.9995 --row-weight $L/data/rowweight_hard2_f0_20261114.npz --root $L/arms/sweep_vitl_elr2e5_hard2_f0_s46 > $L/logs/sweep_vitl_elr2e5_hard2_f0.log 2>&1 < /dev/null &)
(CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 setsid nohup timeout 36000 $PY $R $C --fold 4 --ssl $RB/vitl16_avg_f4_s46/ssl/last.pt --ema 0.998,0.999 --row-weight $L/data/rowweight_hard2_f4_20261114.npz --root $L/arms/sweep_vitl_elr2e5_hard2_f4_s46 > $L/logs/sweep_vitl_elr2e5_hard2_f4.log 2>&1 < /dev/null &)
echo "$(date) hard2 f0 (GPU 1) / f4 (GPU 3) launched" >> $L/logs/launch_hard2.log
