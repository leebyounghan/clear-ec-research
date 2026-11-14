#!/bin/bash
# v8 (03:35 EST 11-14): S2b only; nothing else is launched on GPUs 0 / 3 until the ViT-L interim read.
# v7 (02:50 EST 11-14): GPU 3 only until S1 has finished (then GPU 0 too), so ViT-L fold 4 keeps GPU 0 with S1 only.
# v6 (02:20 EST 11-14): srcnew already launched by v5; S2b and the generic items wait for the ViT-L CD-weighted-loss decision (flag VITL_CDW3_DECIDED, written by auto_vitl_cdw3_20261114.sh), so ViT-L cdw3 goes first if it passes.
# v5 (01:46 EST 11-14): PRL already launched by v4; S2 fine-tune re-queued from its SSL checkpoint (screen_sslS2b_f0_s46, same settings as control P / S1) after srcnew; GPUs 0 and 3 only.
# Coordinator 01:37 EST 11-14: after ViT-L 2e-5 folds 0 / 4 (GPUs 3 / 0), on GPUs 0 and 3 only: PRL probe, srcnew, then the remaining generic
# sweep items (head LR 3e-3, no clip, encoder LR 5e-6, head LR 3e-4, clip 3, batch 8, LLRD 0.85). GPUs 1 and 2 stay with recipe A.
# Launch rule: card 0 or 3 with >= 40 GB free; PRL / srcnew at <= 13 jobs on the pod, generic items at <= 11 (they run only when the pod is
# lighter); 240 s between launches. The pod is CPU-saturated, so no throttle condition is used (it would never pass).
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; LOG=$L/logs; PY=/workspace/venv/bin/python; SSL0=experiments/lane_gpu3_20261106/runs/dinov3_b16_f0_s46/ssl/last.pt
COM="--family dinov3_b16 --seed 46 --ssl $SSL0 --schedule-epochs 10 --micro 16 --workers 2 --tta-final --epochs 20 --keep-epochs 16,17,18,19,20"
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG/auto_master8.log; }
ITEMS=("sslS2b|13|$PY experiments/lane_backbone_20261108/source/bb_runner_20261108.py --family dinov3_b16 --fold 0 --seed 46 --ssl $L/arms/screen_sslS2_f0_s46/ssl/last.pt --schedule-epochs 10 --micro 16 --workers 2 --tta-final --epochs 20 --keep-epochs 16,17,18,19,20 --root $L/arms/screen_sslS2b_f0_s46")
# v8: generic DINOv3-B items dropped (coordinator 03:27); S2b only
say "start pid $$ items ${#ITEMS[@]}"; t0=$(date +%s); i=0
while [ $i -lt ${#ITEMS[@]} ] && [ $(( $(date +%s) - t0 )) -lt 28800 ]; do
  sleep 60; IFS='|' read -r name maxj cmd <<< "${ITEMS[$i]}"
  [ -f $L/VITL_CDW3_DECIDED ] || { say "hold $name: waiting for the ViT-L CD-weighted-loss decision (VITL_CDW3_DECIDED)"; continue; }
  nj=$(ps -eo pid=,ppid=,args= | awk '/python/ && /(bb_runner|ssl_variant|family_tta)/ && !/(gpu_launch|bash -c|timeout|awk)/ {p[$1]=$2} END {n=0; for (k in p) if (!(p[k] in p)) n++; print n}')
  allow3only=1; [ -f $L/arms/screen_sslS1_f0_s46/raw/runs/fold0/dinov3_b16/complete.json ] && allow3only=0   # GPU 0 stays with ViT-L f4 and S1 until S1 ends
  read gi fr <<< "$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v o=$allow3only '$1==3 || ($1==0 && o==0)' | sort -t, -k2 -n -r | head -1 | tr ',' ' ')"
  [ "$nj" -le "$maxj" ] && [ "$fr" -ge 40000 ] || { say "wait $name: jobs $nj, best free on 0/3 $fr (GPU $gi)"; continue; }
  echo "$name|master4|gpu$gi|$(date -u +%FT%TZ)" >> $L/sweep_done_20261113.txt
  (CUDA_VISIBLE_DEVICES=$gi OMP_NUM_THREADS=2 setsid nohup timeout 36000 $cmd > $LOG/sweep_${name}.log 2>&1 < /dev/null &)
  say "launched $name on GPU $gi (jobs $nj, free $fr)"; i=$((i + 1)); sleep 240
done
say "done"
