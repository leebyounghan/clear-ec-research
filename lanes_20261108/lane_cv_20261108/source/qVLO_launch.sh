#!/bin/bash
# CV lane: start order-loss ViT-L (qVLO.sh) on GPU 0 when the CD lane hands it over (CD lane message 2026-11-10): (a) queue_done.txt has
# "vitl16_f99_s46<TAB>0", (b) its queue.log ends with "STOP3 file, exit" or pid 192116 is gone, (c) no compute process on GPU 0.
# If vitl16_f99_s46 finished with rc != 0: hold (do not take GPU 0) and log HOLD; the CV lane then messages the CD lane.
cd /workspace/clear-ec-research
C=experiments/lane_cd_20261108; L=experiments/lane_cv_20261108
while true; do
  line=$(grep -P "^vitl16_f99_s46\t" $C/queue_done.txt 2>/dev/null | tail -1)
  if [ -n "$line" ]; then
    rc=$(echo "$line" | cut -f2)
    if [ "$rc" != "0" ]; then echo "$(date -u) HOLD: vitl16_f99_s46 rc=$rc, not taking GPU 0"; exit 1; fi
    b=0; { tail -1 $C/logs/queue.log 2>/dev/null | grep -q "STOP3 file, exit"; } && b=1; [ -e /proc/192116 ] || b=1
    n=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    if [ $b = 1 ] && [ "$n" = "0" ]; then echo "$(date -u) handover conditions met, starting qVLO.sh on GPU 0"; break; fi
  fi
  sleep 120
done
GPU=0 bash $L/source/qVLO.sh
