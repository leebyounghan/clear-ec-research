#!/bin/bash
# GPU pod (14:27 EST 11-14, coordinator 14:25 item 3): family-level check of the ViT-L s47 2e-5 full-data file against its own fold family
# (cd_family_level_20261114.py vitl_s47_2e5, 500 training rows), once the s47 full-data build has complete.json, on GPU 0 or 2 when one of
# them has no compute process. Stops after 4 h.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; say() { echo "$(date) $*" >> $L/logs/auto_fam47.log; }
say "start pid $$"; t0=$(date +%s)
until [ -f $L/arms/vitl47fu_elr2e5_f99_s47/raw/runs/fold-1/dinov3_l16t/complete.json ]; do [ $(( $(date +%s) - t0 )) -gt 14400 ] && { say "no complete.json in 4 h"; exit 1; }; sleep 60; done
say "s47 full-data complete.json present"
while :; do
  g=""; while IFS=, read -r gi gu; do gi=$(echo $gi | tr -d ' '); gu=$(echo $gu | tr -d ' '); [ "$gi" = 0 ] || [ "$gi" = 2 ] || continue
    c=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$gu"); [ "$c" = 0 ] && { g=$gi; break; }; done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
  [ -n "$g" ] && break; [ $(( $(date +%s) - t0 )) -gt 14400 ] && { say "no free card 0 / 2 in 4 h"; exit 1; }; sleep 60
done
say "running on GPU $g"
CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 nice -n 10 timeout 3600 /workspace/venv/bin/python $L/source/cd_family_level_20261114.py vitl_s47_2e5 > $L/logs/family_level_vitl47.log 2>&1
say "rc $?"
