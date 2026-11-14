#!/bin/bash
# D-CD-SWEEP rolling read: every 5 min, each finished sweep arm (arms/sweep_*_f0_s46, raw final TTA written) without its e16-20 TTA file gets
# family_tta_20261108.py on the card with the fewest jobs (one at a time, 1 loader worker); after each new TTA the screen reader runs again
# (it reads every arm present). Stops after 7 h.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; LOG=$L/logs; PY=/workspace/venv/bin/python; T=$L/source/family_tta_20261108.py
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG/auto_sweep_read.log; }
say "start pid $$"; t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt 25200 ]; do
  new=0
  for r in $L/arms/sweep_*_f0_s46; do
    n=$(basename $r); out=$L/tta/${n%_s46}_tta4_e16_20.npz; d=$r/raw/runs/fold0/dinov3_b16
    [ -f $d/predictions_final_tta.npz ] && [ -f $d/complete.json ] && [ ! -f $out ] || continue
    best=0; bestn=99
    while IFS=, read -r gi gu; do gu=$(echo $gu | tr -d ' '); c=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$gu"); [ "$c" -lt "$bestn" ] && { best=$gi; bestn=$c; }; done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
    CUDA_VISIBLE_DEVICES=$best timeout 1800 $PY $T $r/raw dinov3_b16 0 46 ${n%_s46} > $LOG/sweep_tta_${n}.log 2>&1; say "TTA $n rc $? on GPU $best"; new=1
  done
  if [ $new = 1 ]; then (cd $L/source && OMP_NUM_THREADS=2 timeout 1200 $PY cd_screen_read_20261113.py > ../logs/screen_read_rolling_20261113.log 2>&1); say "read rc $?"; fi
  sleep 300
done
