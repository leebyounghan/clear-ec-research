#!/bin/bash
# CV lane: full-data (fold 99) builds of the order-loss DINOv3 B/16 seeds for the 3-seed rule (s46 is built). usage: GPU=<id> SEED=<47|49> bash experiments/lane_cv_20261108/source/qF99o.sh
# One seed per call; two calls may run side by side (~36 GiB each). SSL = the CD lane's full-data DINOv3 SSL (read only). Writes only under lane_cv_20261108.
cd /workspace/clear-ec-research
L=experiments/lane_cv_20261108; PY=/workspace/venv/bin/python; G=${GPU:?set GPU}; SD=${SEED:?set SEED}
name=dinov3_order_f99_s$SD
if [ -f $L/arms/$name/runs/fold99/dinov3_b16/complete.json ]; then echo "$name already built"; exit 0; fi
echo "$(date -u) start $name"; T0=$(date +%s)
env CD_LOSS=order CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY $L/source/family_cv_20261108.py all --family dinov3_b16 --fold 99 --seed $SD --ssl-from experiments/lane_cd_20261108/arms/dinov3_b16_f99_s46/ssl/last.pt --root $L/arms/$name --workers 3 > $L/logs/$name.log 2>&1; rc=$?
echo "$(date -u) end $name rc $rc seconds $(( $(date +%s) - T0 ))"
