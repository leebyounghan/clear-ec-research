#!/bin/bash
# Rule (a) passed (cdw3 fold-0 member CD in the true e16-20 TTA form 6.4069 vs control P 6.4777, -0.071 <= -0.03): ViT-L folds 0 / 4 with
# --target-w 1.8,0.6,0.6 at encoder LR 1e-5, like for like with the plain ViT-L runs, on GPUs 3 / 0 through the 60 GB guard. The flag
# VITL_CDW3_DECIDED (which releases S2b in auto_master8) is written only after both have launched, so they go ahead of S2b.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; LOG=$L/logs; PY=/workspace/venv/bin/python; G=experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh; RB=experiments/lane_backbone_20261108/runs
(OMP_NUM_THREADS=2 setsid nohup bash $G 3 60000 "$PY $L/source/bb_runner_sweep3_20261113.py --family dinov3_l16t --fold 0 --seed 46 --ssl $RB/vitl16_avg_f0_s46/ssl/last.pt --epochs 12 --schedule-epochs 10 --micro 8 --workers 2 --ema 0.999,0.9995 --swa-start 6 --tta-final --keep-epochs 8,9,10,11,12 --target-w 1.8,0.6,0.6 --root $L/arms/sweep_vitl_cdw3_f0_s46" > $LOG/sweep_vitl_cdw3_f0.log 2>&1 < /dev/null &)
(OMP_NUM_THREADS=2 setsid nohup bash $G 0 60000 "$PY $L/source/bb_runner_sweep3_20261113.py --family dinov3_l16t --fold 4 --seed 46 --ssl $RB/vitl16_avg_f4_s46/ssl/last.pt --epochs 12 --schedule-epochs 10 --micro 8 --workers 2 --ema 0.998,0.999 --swa-start 6 --tta-final --keep-epochs 8,9,10,11,12 --target-w 1.8,0.6,0.6 --root $L/arms/sweep_vitl_cdw3_f4_s46" > $LOG/sweep_vitl_cdw3_f4.log 2>&1 < /dev/null &)
echo "$(date -u +%FT%TZ) ViT-L cdw3 f0 / f4 handed to the guard (GPUs 3 / 0)" >> $LOG/auto_vitl_cdw3.log
t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt 14400 ]; do
  grep -q launching $LOG/sweep_vitl_cdw3_f0.log 2>/dev/null && grep -q launching $LOG/sweep_vitl_cdw3_f4.log 2>/dev/null && break; sleep 60
done
touch $L/VITL_CDW3_DECIDED; echo "$(date -u +%FT%TZ) both launched (or 4 h passed); VITL_CDW3_DECIDED written" >> $LOG/auto_vitl_cdw3.log
