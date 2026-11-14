#!/bin/bash
# D-CD-PRETRAIN-SCREEN: one SSL arm (S1 or S2) on GPU G with W loader workers, then the plain recipe fine-tune with the backbone-lane runner
# (seed 46, 20 epochs, keep 16-20, --tta-final; identical to control P except the SSL init). usage: auto_pretrain_screen_20261113.sh S1|S2 GPU W
cd /workspace/clear-ec-research || exit 1
V=$1; GPU=$2; W=$3; A=experiments/lane_cd_20261108/arms; LOG=experiments/lane_cd_20261108/logs; PY=/workspace/venv/bin/python
OUT=$A/screen_ssl${V}_f0_s46/ssl
export OMP_NUM_THREADS=$W
echo "$(date -u +%FT%TZ) SSL $V start GPU $GPU workers $W" >> $LOG/auto_pretrain_screen.log
CUDA_VISIBLE_DEVICES=$GPU timeout 14400 $PY experiments/lane_cd_20261108/source/ssl_variant_20261113.py $V $OUT $W > $LOG/screen_ssl${V}_ssl.log 2>&1
rc=$?; echo "$(date -u +%FT%TZ) SSL $V rc $rc" >> $LOG/auto_pretrain_screen.log
[ $rc = 0 ] && [ -f $OUT/last.pt ] || exit 1
CUDA_VISIBLE_DEVICES=$GPU timeout 14400 $PY experiments/lane_backbone_20261108/source/bb_runner_20261108.py --family dinov3_b16 --fold 0 --seed 46 --ssl $OUT/last.pt \
  --schedule-epochs 10 --micro 16 --workers $W --tta-final --epochs 20 --keep-epochs 16,17,18,19,20 --root $A/screen_ssl${V}_f0_s46 > $LOG/screen_ssl${V}_ft.log 2>&1
echo "$(date -u +%FT%TZ) fine-tune $V rc $?" >> $LOG/auto_pretrain_screen.log
