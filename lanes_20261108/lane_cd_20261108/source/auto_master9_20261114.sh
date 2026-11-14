#!/bin/bash
# v9 (04:05 EST 11-14, coordinator 03:58): DINOv3-B encoder LR 2e-5 follow-up restarted (folds 1-4 with their own fold SSL inits, then the
# full-data build on all 9,000 rows), then S2b. GPUs 0 and 3 only (GPUs 1 / 2 stay with recipe A until its epoch-12 read). ViT-L with the
# CD-weighted loss was stopped (coordinator's decision), so S2b no longer waits for VITL_CDW3_DECIDED. New roots fu2_elr2e5_* (the closed
# fu_elr2e5_* roots are kept as they are). Settings are the 00:15 follow-up's: bb_runner_sweep3, plain recipe except --enc-lr 2e-5, 20
# epochs, keep 16-20, --tta-final; full data through bb_runner_full9000 with the full-data DINOv3-B s46 SSL init.
# Launch rule: card 0 or 3 with >= 40 GB free (a DINOv3-B run peaks at ~36.6 GB), the card with the most free memory first, 300 s between
# launches so the previous run's memory is counted. f1 / f2 fit now; f3 fits when the PRL probe frees GPU 3; f4, full data, S2b as slots free.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; A=$L/arms; LOG=$L/logs; PY=/workspace/venv/bin/python
RUN=$L/source/bb_runner_sweep3_20261113.py; RUNF=$L/source/bb_runner_full9000_20261113.py
declare -A SSL=([1]=$A/dinov3_b16_f1_s46/ssl/last.pt [2]=$A/dinov3_b16_f2_s46/ssl/last.pt [3]=$A/dinov3_b16_f3_s46/ssl/last.pt [4]=experiments/lane_gpu3_20261106/runs/dinov3_b16_f4_s46/ssl/last.pt)
SW="--enc-lr 2e-5 --epochs 20 --schedule-epochs 10 --micro 16 --workers 2"
ITEMS=()
for f in 1 2 3 4; do ITEMS+=("fu2_elr2e5_f$f|$PY $RUN --family dinov3_b16 --fold $f --seed 46 --ssl ${SSL[$f]} --tta-final --keep-epochs 16,17,18,19,20 --root $A/fu2_elr2e5_f${f}_s46 $SW"); done
ITEMS+=("fu2_elr2e5_f99|$PY $RUNF --family dinov3_b16 --fold -1 --seed 46 --ssl $A/dinov3_b16_f99_s46/ssl/last.pt --keep-epochs 16,17,18,19,20 --root $A/fu2_elr2e5_f99_s46 $SW")
ITEMS+=("sslS2b|$PY experiments/lane_backbone_20261108/source/bb_runner_20261108.py --family dinov3_b16 --fold 0 --seed 46 --ssl $A/screen_sslS2_f0_s46/ssl/last.pt --schedule-epochs 10 --micro 16 --workers 2 --tta-final --epochs 20 --keep-epochs 16,17,18,19,20 --root $A/screen_sslS2b_f0_s46")
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG/auto_master9.log; }
say "start pid $$ items ${#ITEMS[@]}"; t0=$(date +%s); i=0
while [ $i -lt ${#ITEMS[@]} ] && [ $(( $(date +%s) - t0 )) -lt 43200 ]; do
  IFS='|' read -r name cmd <<< "${ITEMS[$i]}"
  read gi fr <<< "$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$1==0 || $1==3' | sort -t, -k2 -n -r | head -1 | tr ',' ' ')"
  [ "$fr" -ge 40000 ] || { say "wait $name: best free on 0/3 $fr (GPU $gi)"; sleep 60; continue; }
  echo "$name|master9|gpu$gi|$(date -u +%FT%TZ)" >> $L/sweep_done_20261113.txt
  (CUDA_VISIBLE_DEVICES=$gi OMP_NUM_THREADS=2 setsid nohup timeout 43200 $cmd > $LOG/$name.log 2>&1 < /dev/null &)
  say "launched $name on GPU $gi (free $fr)"; i=$((i + 1)); sleep 300
done
say "done"
