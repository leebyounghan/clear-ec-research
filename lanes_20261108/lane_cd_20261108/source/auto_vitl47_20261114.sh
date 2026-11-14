#!/bin/bash
# GPU pod (07:10 EST 11-14, coordinator 07:06 rule). When the true-form read of ViT-L s46 encoder LR 2e-5 folds 0 / 4 is out
# (reads/final_V_20261114.json: final.pt four-flip TTA, arm in V's s46 half vs plain s46 in the same form, kept-constants stack):
# if arm minus plain < 0 on both folds, launch ViT-L s47 at encoder LR 2e-5 on folds 0-4 and the 9,000-row full-data build, like for like
# with the plain s47 runs (bb_runner_20261108, 12 epochs, schedule 10, micro 8, EMA 0.998 / 0.999 on every fold, SWA from 6, keep 8-12,
# --tta-final, seed 47, the s46 SSL inits; workers 2 under the pod CPU rule where the plain runs had 4), behind the s46 follow-up jobs f3 / f99
# (waits until auto_vitl_fu has launched f99). Any card 0-3 with >= 60 GB free, most free first, 300 s apart; order f0, f4, f1, f2, f3, f99.
# If not lower on both folds, nothing is launched (logged). Stops after 16 h.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; A=$L/arms; LOG=$L/logs; PY=/workspace/venv/bin/python; RB=experiments/lane_backbone_20261108/runs
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG/auto_vitl47.log; }
VS="--seed 47 --epochs 12 --schedule-epochs 10 --micro 8 --workers 2 --ema 0.998,0.999 --swa-start 6 --keep-epochs 8,9,10,11,12 --enc-lr 2e-5"
declare -A SSL=([0]=$RB/vitl16_avg_f0_s46/ssl/last.pt [1]=$A/vitl16_f1_s46/ssl/last.pt [2]=$A/vitl16_f2_s46/ssl/last.pt [3]=$A/vitl16_f3_s46/ssl/last.pt [4]=$RB/vitl16_avg_f4_s46/ssl/last.pt)
ITEMS=()
for f in 0 4 1 2 3; do ITEMS+=("vitl47fu_elr2e5_f$f|$PY experiments/lane_backbone_20261108/source/bb_runner_20261108.py --family dinov3_l16t --fold $f --ssl ${SSL[$f]} --tta-final $VS --root $A/vitl47fu_elr2e5_f${f}_s47"); done
ITEMS+=("vitl47fu_elr2e5_f99|$PY $L/source/bb_runner_full9000_20261113.py --family dinov3_l16t --fold -1 --ssl $A/vitl16_f99_s46/ssl/last.pt $VS --root $A/vitl47fu_elr2e5_f99_s47")
say "start pid $$ items ${#ITEMS[@]}"; t0=$(date +%s); R=$L/reads/final_V_20261114.json
until [ -f $R ]; do [ $(( $(date +%s) - t0 )) -gt 57600 ] && { say "no true-form read in 16 h"; exit 1; }; sleep 60; done
sleep 30
v=$($PY -c "import json; d=json.load(open('$R')); a=[d[f'fold{f}']['stack']['arm_minus_plain'] for f in (0, 4)]; print(a[0], a[1], int(a[0] < 0 and a[1] < 0))")
say "true-form read: arm minus plain fold 0 / fold 4 / both lower = $v"
[ "${v##* }" = 1 ] || { say "not lower on both folds in the true form: nothing launched"; exit 0; }
until grep -q "launched vitlfu_elr2e5_f99" $LOG/auto_vitl_fu.log; do [ $(( $(date +%s) - t0 )) -gt 57600 ] && { say "s46 f99 never launched; stop"; exit 1; }; sleep 60; done
say "s46 follow-up f3 / f99 launched; s47 queue starts"; i=0
while [ $i -lt ${#ITEMS[@]} ] && [ $(( $(date +%s) - t0 )) -lt 57600 ]; do
  IFS='|' read -r name cmd <<< "${ITEMS[$i]}"
  read gi fr <<< "$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -1 | tr ',' ' ')"
  [ "$fr" -ge 60000 ] || { sleep 60; continue; }
  echo "$name|vitl47|gpu$gi|$(date -u +%FT%TZ)" >> $L/sweep_done_20261113.txt
  (CUDA_VISIBLE_DEVICES=$gi OMP_NUM_THREADS=2 setsid nohup timeout 57600 $cmd > $LOG/$name.log 2>&1 < /dev/null &)
  say "launched $name on GPU $gi (free $fr)"; i=$((i + 1)); sleep 300
done
say "done"
