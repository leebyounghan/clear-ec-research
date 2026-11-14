#!/bin/bash
# CV lane: the single launcher of order-loss ViT-L (qVLO.sh) on GPU 0 (the CD lane withdrew its queue line in favour of this; 2026-11-10).
# Start when lane_cd queue_done.txt has "vitl16_f99_s46<TAB>0" and GPU 0 has no compute process in two checks 2 min apart.
# If vitl16_f99_s46 finished with rc != 0: HOLD (do not take GPU 0); the CV lane messages the CD lane.
cd /workspace/clear-ec-research
C=experiments/lane_cd_20261108; L=experiments/lane_cv_20261108; free=0
while true; do
  line=$(grep -P "^vitl16_f99_s46\t" $C/queue_done.txt 2>/dev/null | tail -1)
  if [ -n "$line" ]; then
    rc=$(echo "$line" | cut -f2)
    if [ "$rc" != "0" ]; then echo "$(date -u) HOLD: vitl16_f99_s46 rc=$rc, not taking GPU 0"; exit 1; fi
    n=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    if [ "$n" = "0" ]; then free=$((free + 1)); else free=0; fi
    if [ $free -ge 2 ]; then echo "$(date -u) vitl16_f99_s46 rc 0 and GPU 0 free twice; starting qVLO.sh on GPU 0"; break; fi
  fi
  sleep 120
done
GPU=0 bash $L/source/qVLO.sh
