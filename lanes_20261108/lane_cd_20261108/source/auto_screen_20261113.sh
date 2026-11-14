#!/bin/bash
# D-CD-TRAIN-SCREEN (coordinator 22:50 EST 11-13, PI): DINOv3-B/16 fold 0 s46 training-setting screen, one arm per GPU, same SSL init as the
# plain fold-0 run (lane_gpu3_20261106/runs/dinov3_b16_f0_s46/ssl/last.pt), backbone-lane runner bb_runner_20261108.py (all switches off =
# recipe parity), through the backbone lane's GPU guard. Arms: A llrd 0.75; B wd 0.05; C appear 0.5; D 12 epochs + SWA from 6; P plain
# control with the same runner (shares GPU 3 with D) so every arm is compared like for like.
cd /workspace/clear-ec-research || exit 1
G=experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh; PY=/workspace/venv/bin/python
RUN=experiments/lane_backbone_20261108/source/bb_runner_20261108.py; A=experiments/lane_cd_20261108/arms; LOG=experiments/lane_cd_20261108/logs
SSL=experiments/lane_gpu3_20261106/runs/dinov3_b16_f0_s46/ssl/last.pt
COMMON="--family dinov3_b16 --fold 0 --seed 46 --ssl $SSL --schedule-epochs 10 --micro 16 --workers 4 --tta-final"
export OMP_NUM_THREADS=4
launch() { gpu=$1; name=$2; shift 2; (setsid nohup bash $G $gpu 40000 "$PY $RUN $COMMON --root $A/screen_${name}_f0_s46 $*" > $LOG/screen_${name}.log 2>&1 < /dev/null &); }
launch 0 llrd075 --epochs 20 --keep-epochs 16,17,18,19,20 --llrd 0.75
launch 1 wd005 --epochs 20 --keep-epochs 16,17,18,19,20 --wd 0.05
launch 2 appear05 --epochs 20 --keep-epochs 16,17,18,19,20 --appear 0.5
launch 3 swa6e12 --epochs 12 --keep-epochs 8,9,10,11,12 --swa-start 6
sleep 20
launch 3 plain --epochs 20 --keep-epochs 16,17,18,19,20
echo "launched $(date -u +%FT%TZ)"
