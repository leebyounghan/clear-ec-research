#!/bin/bash
# CPU pod (10:35 EST 11-14): rerun the ViT-L s47 2e-5 member table whenever another s47 fold has its final TTA (coordinator 10:36).
cd /workspace/clear-ec-research/experiments/lane_cd_20261108/source || exit 1
PY=/workspace/venv/bin/python; say() { echo "$(date -u +%FT%TZ) $*" >> ../logs/auto_vitl47_members.log; }
say "start pid $$"; t0=$(date +%s); last=0
while [ $(( $(date +%s) - t0 )) -lt 57600 ] && [ $last -lt 5 ]; do
  n=0; for f in 0 1 2 3 4; do [ -f ../arms/vitl47fu_elr2e5_f${f}_s47/raw/runs/fold$f/dinov3_l16t/predictions_final_tta.npz ] && n=$((n + 1)); done
  if [ $n -gt $last ]; then sleep 30; OMP_NUM_THREADS=2 nice -n 19 timeout 900 $PY cd_vitl47_members_20261114.py > ../logs/vitl47_members.log 2>&1; say "member table with $n folds rc $?"; last=$n; fi
  sleep 300
done
say "exit"
