#!/bin/bash
# e16-20 four-flip TTA for runner arms (bb_runner* save the last epoch as final.pt, not epoch_20.pt, so family_tta_20261108.py stopped at
# epoch 20; earlier screen / sweep reads fell back to the final-epoch TTA). Adds a link epoch_20.pt -> final.pt in each raw run dir (no file
# changed) and runs family_tta on GPU 3 or 0 (fewer jobs), one arm at a time. usage: tta_fix_20261114.sh ROOTNAME [ROOTNAME ...]
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; LOG=$L/logs; PY=/workspace/venv/bin/python; T=$L/source/family_tta_20261108.py
for n in "$@"; do
  d=$L/arms/$n/raw/runs/fold0/dinov3_b16; [ -f $d/final.pt ] || { echo "$(date -u +%FT%TZ) $n no final.pt" >> $LOG/tta_fix.log; continue; }
  [ -e $d/epoch_20.pt ] || ln -s final.pt $d/epoch_20.pt
  best=3; bestn=99
  while IFS=, read -r gi gu; do gi=$(echo $gi | tr -d ' '); [ "$gi" = 0 ] || [ "$gi" = 3 ] || continue; gu=$(echo $gu | tr -d ' '); c=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$gu"); [ "$c" -lt "$bestn" ] && { best=$gi; bestn=$c; }; done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
  CUDA_VISIBLE_DEVICES=$best timeout 2400 $PY $T $L/arms/$n/raw dinov3_b16 0 46 ${n%_s46} > $LOG/tta_fix_$n.log 2>&1
  echo "$(date -u +%FT%TZ) $n TTA rc $? GPU $best" >> $LOG/tta_fix.log
done
