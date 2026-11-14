#!/bin/bash
# CV lane: full-data DINOv3 B/16 builds for the freeze candidate, seeds 47 and 49 (seed 46 is built by the CD lane); SSL = CD lane full-data SSL (read only)
cd /workspace/clear-ec-research
L=experiments/lane_cv_20261108; PY=/workspace/venv/bin/python
SSL=experiments/lane_cd_20261108/arms/dinov3_b16_f99_s46/ssl/last.pt
for SD in 47 49; do
  name=dinov3_b16_f99_s$SD
  echo "$(date -u) start $name"; T0=$(date +%s)
  env CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY $L/source/family_cv_20261108.py all --family dinov3_b16 --fold 99 --seed $SD --ssl-from $SSL --root $L/arms/$name --workers 3 > $L/logs/$name.log 2>&1
  echo "$(date -u) end $name rc $? seconds $(( $(date +%s) - T0 ))"
done
echo QFDONE
