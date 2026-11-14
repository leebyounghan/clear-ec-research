#!/bin/bash
# GPU pod (04:20 EST 11-14): e16-20 four-flip TTA for the restarted DINOv3-B encoder LR 2e-5 follow-up (fu2_elr2e5_f1..f4) as each fold
# finishes; GPUs 0 / 3 only (fewest compute processes first). bb_runner* save epoch 20 as final.pt, so epoch_20.pt -> final.pt is linked
# first. Outputs tta/fu2_elr2e5_f<fold>_tta4_e16_20.npz (fold 0 = the sweep arm's file, linked). Stops after 14 h or when all four are done.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; A=$L/arms; LOG=$L/logs; PY=/workspace/venv/bin/python; T=$L/source/family_tta_20261108.py
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG/auto_fu2_tta.log; }
[ -e $L/tta/fu2_elr2e5_f0_tta4_e16_20.npz ] || ln -s sweep_elr2e5_f0_tta4_e16_20.npz $L/tta/fu2_elr2e5_f0_tta4_e16_20.npz
say "start pid $$"; t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt 50400 ]; do
  left=0
  for f in 1 2 3 4; do
    d=$A/fu2_elr2e5_f${f}_s46/raw/runs/fold$f/dinov3_b16; out=$L/tta/fu2_elr2e5_f${f}_tta4_e16_20.npz
    [ -f $out ] && continue; left=1
    [ -f $d/complete.json ] && [ -f $d/final.pt ] || continue
    [ -e $d/epoch_20.pt ] || ln -s final.pt $d/epoch_20.pt
    best=0; bestn=99
    while IFS=, read -r gi gu; do gi=$(echo $gi | tr -d ' '); [ "$gi" = 0 ] || [ "$gi" = 3 ] || continue; gu=$(echo $gu | tr -d ' '); c=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$gu"); [ "$c" -lt "$bestn" ] && { best=$gi; bestn=$c; }; done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
    CUDA_VISIBLE_DEVICES=$best OMP_NUM_THREADS=2 timeout 2400 $PY $T $A/fu2_elr2e5_f${f}_s46/raw dinov3_b16 $f 46 fu2_elr2e5_f$f > $LOG/fu2_tta_f$f.log 2>&1
    say "TTA fu2_elr2e5_f$f rc $? on GPU $best"
  done
  [ $left = 0 ] && { say "all four folds done"; break; }
  sleep 120
done
say "exit"
