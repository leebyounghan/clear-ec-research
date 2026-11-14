#!/bin/bash
# Coordinator 06:27 EST 11-14: (1) DINOv3-B encoder LR 2e-5 full-data build (9,000 rows, fu2 fifth job) on GPU 2 now, next to recipe A f4;
# (2) S2b on GPU 1 once recipe A fold 0 has written epoch 12. auto_master9 (which held both for GPUs 0 / 3) was stopped first, so GPUs 0 / 3
# are not refilled when the 2e-5 folds finish (ViT-L keeps them). Same commands as auto_master9's items.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; A=$L/arms; LOG=$L/logs; PY=/workspace/venv/bin/python
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG/auto_master9.log; }
SW="--enc-lr 2e-5 --epochs 20 --schedule-epochs 10 --micro 16 --workers 2"
echo "fu2_elr2e5_f99|launch0627|gpu2|$(date -u +%FT%TZ)" >> $L/sweep_done_20261113.txt
(CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 setsid nohup timeout 43200 $PY $L/source/bb_runner_full9000_20261113.py --family dinov3_b16 --fold -1 --seed 46 --ssl $A/dinov3_b16_f99_s46/ssl/last.pt --keep-epochs 16,17,18,19,20 --root $A/fu2_elr2e5_f99_s46 $SW > $LOG/fu2_elr2e5_f99.log 2>&1 < /dev/null &)
say "launch0627: fu2_elr2e5_f99 launched on GPU 2 (coordinator 06:27)"
t0=$(date +%s)
until [ -f $A/sweep_ra_elr2e5_f0_s46/raw/runs/fold0/cpdino_b8/predictions_epoch_12.npz ]; do
  [ $(( $(date +%s) - t0 )) -gt 10800 ] && { say "launch0627: recipe A f0 epoch 12 not seen in 3 h; S2b not launched"; exit 1; }; sleep 60
done
echo "sslS2b|launch0627|gpu1|$(date -u +%FT%TZ)" >> $L/sweep_done_20261113.txt
(CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 setsid nohup timeout 36000 $PY experiments/lane_backbone_20261108/source/bb_runner_20261108.py --family dinov3_b16 --fold 0 --seed 46 --ssl $A/screen_sslS2_f0_s46/ssl/last.pt --schedule-epochs 10 --micro 16 --workers 2 --tta-final --epochs 20 --keep-epochs 16,17,18,19,20 --root $A/screen_sslS2b_f0_s46 > $LOG/sweep_sslS2b.log 2>&1 < /dev/null &)
say "launch0627: sslS2b launched on GPU 1 (recipe A f0 epoch 12 written)"
