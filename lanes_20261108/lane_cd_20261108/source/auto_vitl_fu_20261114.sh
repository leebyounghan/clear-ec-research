#!/bin/bash
# GPU pod (06:45 EST 11-14). ViT-L encoder LR 2e-5 follow-up: the rule fixed by the coordinator fired at the e8-10 interim read (kept-constants
# stack lower than plain on both folds: fold 0 -0.0148, fold 4 -0.0103; also lower than the incumbent on both). Folds 1-3 (bb_runner_20261108,
# the plain ViT-L fold runs' settings and SSL inits, --enc-lr 2e-5) and the full-data build on all 9,000 rows (bb_runner_full9000, the plain
# full-data ViT-L s46 SSL init), on GPUs 0 / 3 only. Coordinator 06:27 item 4: freed slots on GPUs 0 / 3 are not refilled until the ViT-L
# arms f0 / f4 have finished, so nothing launches before both have complete.json. Then: card 0 or 3 with >= 65 GB free (a ViT-L run peaks
# at ~54.6 GB), most free first, 300 s apart. Order f1, f2, f3, full data. Stops after 14 h.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; A=$L/arms; LOG=$L/logs; PY=/workspace/venv/bin/python
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG/auto_vitl_fu.log; }
VS="--epochs 12 --schedule-epochs 10 --micro 8 --workers 2 --ema 0.998,0.999 --swa-start 6 --keep-epochs 8,9,10,11,12 --enc-lr 2e-5"
ITEMS=()
for f in 1 2 3; do ITEMS+=("vitlfu_elr2e5_f$f|$PY experiments/lane_backbone_20261108/source/bb_runner_20261108.py --family dinov3_l16t --fold $f --seed 46 --ssl $A/vitl16_f${f}_s46/ssl/last.pt --tta-final $VS --root $A/vitlfu_elr2e5_f${f}_s46"); done
ITEMS+=("vitlfu_elr2e5_f99|$PY $L/source/bb_runner_full9000_20261113.py --family dinov3_l16t --fold -1 --seed 46 --ssl $A/vitl16_f99_s46/ssl/last.pt $VS --root $A/vitlfu_elr2e5_f99_s46")
say "start pid $$ items ${#ITEMS[@]}"; t0=$(date +%s)
until [ -f $A/sweep_vitl_elr2e5_f0_s46/raw/runs/fold0/dinov3_l16t/complete.json ] && [ -f $A/sweep_vitl_elr2e5_f4_s46/raw/runs/fold4/dinov3_l16t/complete.json ]; do
  [ $(( $(date +%s) - t0 )) -gt 50400 ] && { say "ViT-L arms not finished in 14 h; nothing launched"; exit 1; }; sleep 60
done
say "ViT-L arms f0 / f4 finished; launching the follow-up"; i=0
while [ $i -lt ${#ITEMS[@]} ] && [ $(( $(date +%s) - t0 )) -lt 50400 ]; do
  IFS='|' read -r name cmd <<< "${ITEMS[$i]}"
  read gi fr <<< "$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$1==0 || $1==3' | sort -t, -k2 -n -r | head -1 | tr ',' ' ')"
  [ "$fr" -ge 65000 ] || { say "wait $name: best free on 0/3 $fr (GPU $gi)"; sleep 60; continue; }
  echo "$name|vitl_fu|gpu$gi|$(date -u +%FT%TZ)" >> $L/sweep_done_20261113.txt
  (CUDA_VISIBLE_DEVICES=$gi OMP_NUM_THREADS=2 setsid nohup timeout 50400 $cmd > $LOG/$name.log 2>&1 < /dev/null &)
  say "launched $name on GPU $gi (free $fr)"; i=$((i + 1)); sleep 300
done
say "done"
