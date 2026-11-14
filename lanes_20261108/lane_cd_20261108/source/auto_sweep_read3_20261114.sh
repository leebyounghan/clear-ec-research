#!/bin/bash
# v3 (02:50 EST 11-14): source specialists closed, their TTA skipped.
# Rolling TTA + read v2 (coordinator 02:16 EST 11-14): GPUs 0 and 3 only (GPUs 1 / 2 hold recipe A), closed arms skipped. Every 5 min each
# open fold-0 arm with its final TTA written and no e16-20 TTA file gets family_tta_20261108.py on GPU 0 or 3 (fewer jobs); source specialists
# (fold dirs 301 / 302) are reached through a link dir raw_f0link/runs/fold0 -> raw/runs/fold30x (their validation rows are the fold-0 rows);
# after each new TTA the screen reader runs again. Stops after 8 h.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; LOG=$L/logs; PY=/workspace/venv/bin/python; T=$L/source/family_tta_20261108.py
CLOSED="sweep_elr2e5_f0_s46 sweep_bs32_f0_s46 sweep_dp01_f0_s46 sweep_srcold_f0_s46 sweep_srcnew_f0_s46"
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG/auto_sweep_read3.log; }
say "start pid $$"; t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt 28800 ]; do
  new=0
  for r in $L/arms/sweep_*_f0_s46; do
    n=$(basename $r); case " $CLOSED " in *" $n "*) continue;; esac
    case $n in *vitl*) continue;; esac
    fd=fold0; case $n in *srcold*) fd=fold301;; *srcnew*) fd=fold302;; esac
    d=$r/raw/runs/$fd/dinov3_b16; out=$L/tta/${n%_s46}_tta4_e16_20.npz
    [ -f $d/predictions_final_tta.npz ] && [ -f $d/complete.json ] && [ ! -f $out ] || continue
    root=$r/raw
    if [ $fd != fold0 ]; then mkdir -p $r/raw_f0link/runs/fold0; [ -e $r/raw_f0link/runs/fold0/dinov3_b16 ] || ln -s ../../../raw/runs/$fd/dinov3_b16 $r/raw_f0link/runs/fold0/dinov3_b16; root=$r/raw_f0link; fi
    best=0; bestn=99
    while IFS=, read -r gi gu; do gi=$(echo $gi | tr -d ' '); [ "$gi" = 0 ] || [ "$gi" = 3 ] || continue; gu=$(echo $gu | tr -d ' '); c=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$gu"); [ "$c" -lt "$bestn" ] && { best=$gi; bestn=$c; }; done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
    CUDA_VISIBLE_DEVICES=$best timeout 2400 $PY $T $root dinov3_b16 0 46 ${n%_s46} > $LOG/sweep_tta2_${n}.log 2>&1; say "TTA $n rc $? on GPU $best"; new=1
  done
  if [ $new = 1 ]; then (cd $L/source && OMP_NUM_THREADS=2 nice -n 10 timeout 1200 $PY cd_screen_read_20261113.py > ../logs/screen_read_rolling_20261113.log 2>&1); say "read rc $?"; fi
  sleep 300
done
