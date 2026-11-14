#!/bin/bash
# GPU pod (08:20 EST 11-14, coordinator 08:17): ViT-L s47 encoder LR 2e-5 runs behind the ViT-L s46 follow-up, not beside it. auto_vitl47
# (pid 756114) and the first s47 f0 / f4 runs were stopped (no epoch written; they are relaunched into the same roots: the runner refuses
# only a root with history.json, and those dirs hold just launch.json). A card takes an s47 job only when no s46 follow-up job (vitlfu_*) runs
# on it, it has fewer than 2 ViT-L jobs, and >= 60 GB is free; most free first, 300 s apart. Order f0, f4, f1, f2, f3, full data. Same
# commands and settings as auto_vitl47 (like for like with the plain s47 runs; workers 2). Stops after 16 h.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; A=$L/arms; LOG=$L/logs; PY=/workspace/venv/bin/python; RB=experiments/lane_backbone_20261108/runs
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG/auto_vitl47.log; }
VS="--seed 47 --epochs 12 --schedule-epochs 10 --micro 8 --workers 2 --ema 0.998,0.999 --swa-start 6 --keep-epochs 8,9,10,11,12 --enc-lr 2e-5"
declare -A SSL=([0]=$RB/vitl16_avg_f0_s46/ssl/last.pt [1]=$A/vitl16_f1_s46/ssl/last.pt [2]=$A/vitl16_f2_s46/ssl/last.pt [3]=$A/vitl16_f3_s46/ssl/last.pt [4]=$RB/vitl16_avg_f4_s46/ssl/last.pt)
ITEMS=()
for f in 0 4 1 2 3; do ITEMS+=("vitl47fu_elr2e5_f$f|$PY experiments/lane_backbone_20261108/source/bb_runner_20261108.py --family dinov3_l16t --fold $f --ssl ${SSL[$f]} --tta-final $VS --root $A/vitl47fu_elr2e5_f${f}_s47"); done
ITEMS+=("vitl47fu_elr2e5_f99|$PY $L/source/bb_runner_full9000_20261113.py --family dinov3_l16t --fold -1 --ssl $A/vitl16_f99_s46/ssl/last.pt $VS --root $A/vitl47fu_elr2e5_f99_s47")
pick() {  # prints "<card> <free>" of the best eligible card, or nothing
  declare -A U2I; while IFS=, read -r gi gu; do U2I[$(echo $gu | tr -d ' ')]=$(echo $gi | tr -d ' '); done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
  declare -A S46 NVL; while IFS=, read -r p gu; do c=${U2I[$(echo $gu | tr -d ' ')]}; a=$(ps -o args= -p $(echo $p | tr -d ' ') 2>/dev/null)
    case "$a" in *vitlfu_elr2e5_f*) S46[$c]=$(( ${S46[$c]:-0} + 1 ));; esac; case "$a" in *dinov3_l16t*) NVL[$c]=$(( ${NVL[$c]:-0} + 1 ));; esac
  done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | while IFS=, read -r gi fr; do gi=$(echo $gi | tr -d ' '); fr=$(echo $fr | tr -d ' ')
    [ ${S46[$gi]:-0} = 0 ] && [ ${NVL[$gi]:-0} -lt 2 ] && [ $fr -ge 60000 ] && echo "$gi $fr"; done | sort -k2 -n -r | head -1
}
say "v2 (behind s46) start pid $$ items ${#ITEMS[@]}"; t0=$(date +%s); i=0; last=0
while [ $i -lt ${#ITEMS[@]} ] && [ $(( $(date +%s) - t0 )) -lt 57600 ]; do
  IFS='|' read -r name cmd <<< "${ITEMS[$i]}"; r=$(pick)
  if [ -z "$r" ]; then [ $(( $(date +%s) - last )) -ge 1800 ] && { say "wait $name: no eligible card (s46 job on it, 2 ViT-L, or < 60 GB free)"; last=$(date +%s); }; sleep 60; continue; fi
  read gi fr <<< "$r"
  echo "$name|vitl47b|gpu$gi|$(date -u +%FT%TZ)" >> $L/sweep_done_20261113.txt
  (CUDA_VISIBLE_DEVICES=$gi OMP_NUM_THREADS=2 setsid nohup timeout 57600 $cmd > $LOG/$name.log 2>&1 < /dev/null &)
  say "launched $name on GPU $gi (free $fr)"; i=$((i + 1)); sleep 300
done
say "v2 done"
