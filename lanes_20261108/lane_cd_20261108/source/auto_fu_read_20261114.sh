#!/bin/bash
# Follow-up read automation: wait for fu_<NAME>_f{1..4} (final TTA written) and the fold-0 arm's e16-20 TTA, run family_tta for folds 1-4 on
# the card with the fewest jobs (one at a time, 1 worker), then the strict five-fold read. usage: auto_fu_read_20261114.sh NAME F0ROOTNAME
cd /workspace/clear-ec-research || exit 1
NAME=$1; F0=$2; L=experiments/lane_cd_20261108; A=$L/arms; LOG=$L/logs; PY=/workspace/venv/bin/python; T=$L/source/family_tta_20261108.py
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG/auto_fu_read_$NAME.log; }
say "start pid $$"; t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt 21600 ]; do
  for f in 1 2 3 4; do
    r=$A/fu_${NAME}_f${f}_s46; d=$r/raw/runs/fold$f/dinov3_b16; out=$L/tta/fu_${NAME}_f${f}_tta4_e16_20.npz
    if [ -f $d/predictions_final_tta.npz ] && [ -f $d/complete.json ] && [ ! -f $out ]; then
      best=0; bestn=99
      while IFS=, read -r gi gu; do gu=$(echo $gu | tr -d ' '); c=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$gu"); [ "$c" -lt "$bestn" ] && { best=$gi; bestn=$c; }; done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
      CUDA_VISIBLE_DEVICES=$best timeout 1800 $PY $T $r/raw dinov3_b16 $f 46 fu_${NAME}_f${f} > $LOG/fu_tta_${NAME}_f$f.log 2>&1; say "TTA f$f rc $? GPU $best"
    fi
  done
  n=0; for f in 1 2 3 4; do [ -f $L/tta/fu_${NAME}_f${f}_tta4_e16_20.npz ] && n=$((n + 1)); done
  if [ $n = 4 ] && [ -f $L/tta/${F0}_tta4_e16_20.npz ]; then
    (cd $L/source && NAME=$NAME F0NAME=$F0 OMP_NUM_THREADS=2 timeout 1800 $PY cd_fu_read_20261114.py > ../logs/fu_read_${NAME}_20261114.log 2>&1); say "read rc $?"; exit 0
  fi
  sleep 120
done
say "gave up"
