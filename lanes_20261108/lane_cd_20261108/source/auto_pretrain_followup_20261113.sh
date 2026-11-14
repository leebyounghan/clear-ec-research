#!/bin/bash
# D-CD-PRETRAIN-SCREEN follow-up for a winning SSL arm V (S1|S2): per fold f = 1..4 on GPU f-1 its own SSL (fold f) then the plain fine-tune;
# full data: SSL on fold 99 (8952 rows) then the fine-tune with --fold -1, on GPU GF. W loader workers each.
# usage: auto_pretrain_followup_20261113.sh V GF W
cd /workspace/clear-ec-research || exit 1
V=$1; GF=$2; W=$3; A=experiments/lane_cd_20261108/arms; LOG=experiments/lane_cd_20261108/logs; PY=/workspace/venv/bin/python
SV=experiments/lane_cd_20261108/source/ssl_variant_fold_20261113.py; RUN=experiments/lane_backbone_20261108/source/bb_runner_20261108.py; RUNF=experiments/lane_cd_20261108/source/bb_runner_full9000_20261113.py
export OMP_NUM_THREADS=$W
one() {  # gpu sslfold ftfold root
  g=$1; sf=$2; ff=$3; r=$4
  CUDA_VISIBLE_DEVICES=$g timeout 14400 $PY $SV $V $r/ssl $W $sf > $LOG/$(basename $r)_ssl.log 2>&1 || { echo "$(date -u +%FT%TZ) $r SSL failed" >> $LOG/auto_pretrain_followup.log; return; }
  extra="--keep-epochs 16,17,18,19,20 --tta-final"; [ $ff = -1 ] && extra="--keep-epochs 16,17,18,19,20"
  R_=$RUN; [ $ff = -1 ] && R_=$RUNF; CUDA_VISIBLE_DEVICES=$g timeout 14400 $PY $R_ --family dinov3_b16 --fold $ff --seed 46 --ssl $r/ssl/last.pt --schedule-epochs 10 --micro 16 --workers $W --epochs 20 $extra --root $r > $LOG/$(basename $r)_ft.log 2>&1
  echo "$(date -u +%FT%TZ) $r fine-tune rc $?" >> $LOG/auto_pretrain_followup.log
}
for f in 1 2 3 4; do (one $((f - 1)) $f $f $A/screen_ssl${V}_f${f}_s46) & done
(one $GF 99 -1 $A/screen_ssl${V}_f99_s46) &
echo "$(date -u +%FT%TZ) followup $V launched" >> $LOG/auto_pretrain_followup.log
wait
