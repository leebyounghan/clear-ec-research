#!/bin/bash
# v4 (03:35 EST 11-14): link epoch_20.pt -> final.pt before TTA.
# v3 (02:20 EST 11-14): TTA on GPUs 0 and 3 only (GPUs 1 / 2 hold recipe A).
# v2 (01:46 EST 11-14): S2 is read from the re-queued fine-tune screen_sslS2b_f0_s46 (the first S2 fine-tune was stopped to free GPU 1 for recipe A); 7 h limit.
# D-CD-PRETRAIN-SCREEN: wait for the S1 / S2 fine-tunes (and control P), four-flip TTA over epochs 16-20 on the arm's own GPU, then the screen read again.
cd /workspace/clear-ec-research || exit 1
A=experiments/lane_cd_20261108/arms; LOG=experiments/lane_cd_20261108/logs; PY=/workspace/venv/bin/python; T=experiments/lane_cd_20261108/source/family_tta_20261108.py
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG/auto_pretrain_read4.log; }
say "start pid $$"; t0=$(date +%s)
d() { [ -f $A/screen_$1_f0_s46/raw/runs/fold0/dinov3_b16/predictions_final_tta.npz ]; }
while [ $(( $(date +%s) - t0 )) -lt 25200 ]; do
  d sslS1 && d sslS2b && d plain && break; sleep 60
done
say "fine-tunes complete; TTA"
for n in sslS1 sslS2b; do dd=$A/screen_${n}_f0_s46/raw/runs/fold0/dinov3_b16; [ -e $dd/epoch_20.pt ] || ln -s final.pt $dd/epoch_20.pt; done   # v4: epoch 20 is final.pt
(CUDA_VISIBLE_DEVICES=0 timeout 2400 $PY $T $A/screen_sslS1_f0_s46/raw dinov3_b16 0 46 screen_sslS1_f0 > $LOG/screen_tta_sslS1.log 2>&1) &
(CUDA_VISIBLE_DEVICES=3 timeout 2400 $PY $T $A/screen_sslS2b_f0_s46/raw dinov3_b16 0 46 screen_sslS2b_f0 > $LOG/screen_tta_sslS2b.log 2>&1) &
wait
while [ ! -f experiments/lane_cd_20261108/tta/screen_plain_f0_tta4_e16_20.npz ] && [ $(( $(date +%s) - t0 )) -lt 14400 ]; do sleep 30; done
cd experiments/lane_cd_20261108/source && OMP_NUM_THREADS=3 timeout 1200 $PY cd_screen_read_20261113.py > ../logs/screen_read_pretrain_20261113.log 2>&1
say "read rc $?"
