#!/bin/bash
# Coordinator 02:16 EST 11-14 rule: when the cdw3 arm's fold-0 e16-20 four-flip TTA file exists, its member CD on fold 0 decides: if at least
# 0.03 below control P's 6.4053 (<= 6.3753), launch ViT-L folds 0 and 4 with --target-w 1.8,0.6,0.6, encoder LR 1e-5, like for like with the
# plain ViT-L runs (bb_runner_sweep3 = backbone runner + --target-w; 12 epochs, micro 8, swa6, ema as each plain run, the fold's own SSL),
# on GPUs 0 and 3 (guard, 60 GB), ahead of S2b and the generic items; otherwise no launch (CD-weighted loss closed). Writes VITL_CDW3_DECIDED.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; LOG=$L/logs; PY=/workspace/venv/bin/python; G=experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh; RB=experiments/lane_backbone_20261108/runs
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG/auto_vitl_cdw3.log; }
say "start pid $$"; t0=$(date +%s)
while [ ! -f $L/tta/sweep_cdw3_f0_tta4_e16_20.npz ] && [ $(( $(date +%s) - t0 )) -lt 14400 ]; do sleep 60; done
cdv=$(cd $L/source && $PY -c "
import numpy as np
from cd_common import *
import cd_read_20261108 as R
z = np.load(LANE / 'tta/sweep_cdw3_f0_tta4_e16_20.npz'); v = np.full(len(yc), np.nan); v[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]
print(round(cd(v, fold == 0), 4))" 2>/dev/null | tail -1)
say "cdw3 member CD fold 0 (e16-20 TTA) = $cdv (threshold 6.3753 = P 6.4053 - 0.03)"
if [ -n "$cdv" ] && python3 -c "import sys; sys.exit(0 if float('$cdv') <= 6.3753 else 1)"; then
  (OMP_NUM_THREADS=2 setsid nohup bash $G 3 60000 "$PY $L/source/bb_runner_sweep3_20261113.py --family dinov3_l16t --fold 0 --seed 46 --ssl $RB/vitl16_avg_f0_s46/ssl/last.pt --epochs 12 --schedule-epochs 10 --micro 8 --workers 2 --ema 0.999,0.9995 --swa-start 6 --tta-final --keep-epochs 8,9,10,11,12 --target-w 1.8,0.6,0.6 --root $L/arms/sweep_vitl_cdw3_f0_s46" > $LOG/sweep_vitl_cdw3_f0.log 2>&1 < /dev/null &)
  (OMP_NUM_THREADS=2 setsid nohup bash $G 0 60000 "$PY $L/source/bb_runner_sweep3_20261113.py --family dinov3_l16t --fold 4 --seed 46 --ssl $RB/vitl16_avg_f4_s46/ssl/last.pt --epochs 12 --schedule-epochs 10 --micro 8 --workers 2 --ema 0.998,0.999 --swa-start 6 --tta-final --keep-epochs 8,9,10,11,12 --target-w 1.8,0.6,0.6 --root $L/arms/sweep_vitl_cdw3_f4_s46" > $LOG/sweep_vitl_cdw3_f4.log 2>&1 < /dev/null &)
  say "PASS: ViT-L cdw3 folds 0 / 4 handed to the guard on GPUs 3 / 0"; sleep 600
else
  say "FAIL: no launch; CD-weighted loss closed"
fi
touch $L/VITL_CDW3_DECIDED; say "decided"
