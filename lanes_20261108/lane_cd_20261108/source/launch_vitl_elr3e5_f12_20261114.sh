#!/bin/bash
# Coordinator 15:12 EST 11-14: ViT-L s46 encoder LR 3e-5 on folds 1 and 2, same recipe as the folds 0 / 4 launched at 14:48 and as the 2e-5
# fold runs of those folds (bb_runner_20261108, the fold's own SSL arms/vitl16_f<k>_s46, 12 epochs, schedule 10, micro 8, workers 2,
# EMA 0.998/0.999 as the plain / 2e-5 fold 1-3 runs, SWA from 6, keep 8-12, --tta-final, seed 46). Fold 1 on GPU 1 after the hard-image
# fold-0 run there has complete.json; fold 2 on GPU 3 after the hard-image fold-4 run has complete.json; then through the 60 GB guard.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; PY=/workspace/venv/bin/python; R=experiments/lane_backbone_20261108/source/bb_runner_20261108.py; G=experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh
C="--family dinov3_l16t --seed 46 --epochs 12 --schedule-epochs 10 --micro 8 --workers 2 --ema 0.998,0.999 --swa-start 6 --tta-final --keep-epochs 8,9,10,11,12 --enc-lr 3e-5"
say() { echo "$(date) $*" >> $L/logs/launch_vitl_elr3e5_f12.log; }
go() {  # $1 fold, $2 gpu, $3 hard2 fold whose run must finish first
  t0=$(date +%s); until [ -f $L/arms/sweep_vitl_elr2e5_hard2_f$3_s46/raw/runs/fold$3/dinov3_l16t/complete.json ]; do [ $(( $(date +%s) - t0 )) -gt 14400 ] && { say "hard2 f$3 not complete in 4 h; f$1 not launched"; return; }; sleep 30; done
  sleep 60
  (OMP_NUM_THREADS=2 setsid nohup bash $G $2 60000 "$PY $R $C --fold $1 --ssl $L/arms/vitl16_f$1_s46/ssl/last.pt --root $L/arms/sweep_vitl_elr3e5_f$1_s46" > $L/logs/sweep_vitl_elr3e5_f$1.log 2>&1 < /dev/null &)
  say "f$1 handed to the guard on GPU $2 (hard2 f$3 complete)"
}
say "start pid $$"
go 1 1 0 &
go 2 3 4 &
wait; say "done"
