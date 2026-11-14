#!/bin/bash
# Follow-up of sweep arm encoder LR 2e-5 (coordinator 00:15 EST 11-14; interim fold-0 stack lower than with P): folds 1-4 with their own fold
# SSL inits (bb_runner_sweep3, plain recipe except --enc-lr 2e-5, 20 epochs, keep 16-20, --tta-final) and the full-data build on all 9,000 rows
# (bb_runner_full9000, SSL init of the full-data DINOv3-B s46, which was trained on 8,952 rows). Each launch goes through the GPU guard
# (>= 40 GB free): f1, f3 on GPU 2 (f3 starts when appear05 ends), f2, f4 on GPU 3 (f4 when control P ends), full data on GPU 1 (when wd005 ends).
cd /workspace/clear-ec-research || exit 1
G=experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh; PY=/workspace/venv/bin/python; L=experiments/lane_cd_20261108; A=$L/arms; LOG=$L/logs
RUN=$L/source/bb_runner_sweep3_20261113.py; RUNF=$L/source/bb_runner_full9000_20261113.py
declare -A SSL=([1]=$A/dinov3_b16_f1_s46/ssl/last.pt [2]=$A/dinov3_b16_f2_s46/ssl/last.pt [3]=$A/dinov3_b16_f3_s46/ssl/last.pt [4]=experiments/lane_gpu3_20261106/runs/dinov3_b16_f4_s46/ssl/last.pt)
SW="--enc-lr 2e-5 --epochs 20 --schedule-epochs 10 --micro 16 --workers 2"
export OMP_NUM_THREADS=2
go() { (setsid nohup bash $G $1 40000 "$PY $RUN --family dinov3_b16 --fold $2 --seed 46 --ssl ${SSL[$2]} --tta-final --keep-epochs 16,17,18,19,20 --root $A/fu_elr2e5_f$2_s46 $SW" > $LOG/fu_elr2e5_f$2.log 2>&1 < /dev/null &); }
go 2 1; go 3 2; sleep 5; go 2 3; go 3 4
(setsid nohup bash $G 1 40000 "$PY $RUNF --family dinov3_b16 --fold -1 --seed 46 --ssl $A/dinov3_b16_f99_s46/ssl/last.pt --keep-epochs 16,17,18,19,20 --root $A/fu_elr2e5_f99_s46 $SW" > $LOG/fu_elr2e5_f99.log 2>&1 < /dev/null &)
echo "$(date -u +%FT%TZ) follow-up elr2e5 launched (guarded)" >> $LOG/auto_followup.log
