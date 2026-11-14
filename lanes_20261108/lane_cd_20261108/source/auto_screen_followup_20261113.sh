#!/bin/bash
# D-CD-TRAIN-SCREEN follow-up: an arm that beat the plain control on fold 0 goes to folds 1-4 (GPUs 0-3) and the full-data build
# (--fold -1 with bb_runner_full9000 = the 9,000-frame folder, 8,998 rows; shares GPU G_FULL); folds with bb_runner_sweep3 (all sweep switches); SSL inits of the plain DINOv3-B s46 runs.
# usage: auto_screen_followup_20261113.sh NAME G_FULL 'ARM SWITCHES'   e.g.  auto_screen_followup_20261113.sh wd005 0 '--epochs 20 --keep-epochs 16,17,18,19,20 --wd 0.05'
cd /workspace/clear-ec-research || exit 1
NAME=$1; GF=$2; SW=$3
G=experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh; PY=/workspace/venv/bin/python
RUN=experiments/lane_cd_20261108/source/bb_runner_sweep3_20261113.py; RUNF=experiments/lane_cd_20261108/source/bb_runner_full9000_20261113.py; A=experiments/lane_cd_20261108/arms; LOG=experiments/lane_cd_20261108/logs
declare -A SSL=([1]=$A/dinov3_b16_f1_s46/ssl/last.pt [2]=$A/dinov3_b16_f2_s46/ssl/last.pt [3]=$A/dinov3_b16_f3_s46/ssl/last.pt
                [4]=experiments/lane_gpu3_20261106/runs/dinov3_b16_f4_s46/ssl/last.pt [-1]=$A/dinov3_b16_f99_s46/ssl/last.pt)
export OMP_NUM_THREADS=2
for f in 1 2 3 4; do
  g=$((f - 1)); r=$A/screen_${NAME}_f${f}_s46
  (setsid nohup bash $G $g 40000 "$PY $RUN --family dinov3_b16 --fold $f --seed 46 --ssl ${SSL[$f]} --schedule-epochs 10 --micro 16 --workers 2 --tta-final --root $r $SW" > $LOG/screen_${NAME}_f${f}.log 2>&1 < /dev/null &)
done
sleep 30
(setsid nohup bash $G $GF 40000 "$PY $RUNF --family dinov3_b16 --fold -1 --seed 46 --ssl ${SSL[-1]} --schedule-epochs 10 --micro 16 --workers 2 --root $A/screen_${NAME}_f99_s46 $SW" > $LOG/screen_${NAME}_f99.log 2>&1 < /dev/null &)
echo "followup $NAME launched $(date -u +%FT%TZ)"
