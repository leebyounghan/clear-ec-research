#!/bin/bash
# sweep_elr3e5 watcher, third version (02:35 EST 11-14). v1 never matched (tilde expanded on the Mac); v2 pulled small files first, which would
# let the CD lane's rolling runner (auto_sweep_read2_20261114.sh: TTA on GPU 0/3 for any open arms/sweep_*_f0_s46 with predictions_final_tta.npz
# + complete.json and no e16-20 file) start on a copy without weights, racing our own TTA. Here: wait for EXIT on the box; ONE detached pull
# with WITH_PT=1 (rsync writes files in name order, so predictions_final_tta.npz lands after epoch_*.pt / final.pt); the CD runner then does
# the TTA (GPU 0/3). Fallback: if after 15 min there is no TTA output, no CD-runner log for this arm and no family_tta process for it, run
# family_tta ourselves on GPU 0 or 3 through the guard (8000 MiB). Waits for experiments/lane_cd_20261108/tta/sweep_elr3e5_f0_tta4_e16_20.npz.
S=/private/tmp/claude-501/-Users-byounghanlee-Desktop-k8s-server/3fe760b8-2aec-41ed-af70-e1afef1b7f06/scratchpad
A=sweep_elr3e5; O=${A}_f0; H=workspace/clear-ec/train_20261113/arms/${A}_f0_s46; D=experiments/lane_cd_20261108/arms/${A}_f0_s46
LG=experiments/lane_cellroute_20261109/logs; OUT=experiments/lane_cd_20261108/tta/${O}_tta4_e16_20.npz
line=""; seen=""
for i in $(seq 1 180); do
  r=$(ssh -o BatchMode=yes -o ConnectTimeout=15 qudgks@210.107.192.239 "test -f $H/train.log && echo LOG; grep '^EXIT' $H/train.log" < /dev/null 2>/dev/null)
  case "$r" in LOG*) seen=1;; esac
  line=$(printf '%s\n' "$r" | grep '^EXIT' | tail -1)
  [ -n "$line" ] && break
  [ $i -eq 3 ] && [ -z "$seen" ] && { echo "log not visible on the box after 3 tries: stop"; exit 1; }
  sleep 60
done
echo "end: ${line:-none} at $(TZ=America/New_York date +%H:%M)"
[ -n "$line" ] || exit 1
$S/kx 30 "WITH_PT=1 setsid nohup bash experiments/lane_cellroute_20261109/inference/boxtrain_20261113/pull_arm_20261113.sh $A > $LG/boxtrain_pullpt_$A.log 2>&1 < /dev/null & echo pull_started"
for i in $(seq 1 120); do
  n=$($S/kx 20 "ls $D/raw/runs/fold0/dinov3_b16/epoch_{16,17,18,19,20}.pt $D/raw/runs/fold0/dinov3_b16/predictions_final_tta.npz 2>/dev/null | wc -l; pgrep -f '[p]ull_arm_20261113.sh $A' > /dev/null && echo RUN || echo DONE" 2>/dev/null | tr '\n' ' ')
  case "$n" in "6 DONE"*) break;; esac
  case "$n" in *DONE*) [ $i -gt 2 ] && break;; esac
  sleep 15
done
echo "pull: $n at $(TZ=America/New_York date +%H:%M)"; $S/kx 20 "tail -3 $LG/boxtrain_pullpt_$A.log; tail -1 $D/train.log"
[ "$line" = "EXIT 0" ] || { echo "not EXIT 0: no TTA"; exit 1; }
case "$n" in "6 DONE"*) ;; *) echo "weights incomplete: no TTA"; exit 1;; esac
started=""
for i in $(seq 1 30); do
  st=$($S/kx 20 "test -f $OUT && echo OUT; test -f experiments/lane_cd_20261108/logs/sweep_tta2_${A}_f0_s46.log && echo CDLOG; pgrep -f '[f]amily_tta_20261108.py .*${A}_f0_s46' > /dev/null && echo PROC" 2>/dev/null | tr '\n' ' ')
  case "$st" in *OUT*) echo "TTA output present ($st) at $(TZ=America/New_York date +%H:%M)"; started=done; break;; *CDLOG*|*PROC*) started=cd;; esac
  [ -n "$started" ] && break
  sleep 30
done
if [ -z "$started" ]; then
  G=$($S/kx 20 "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | grep -E '^ *(0|3),' | sort -t, -k2 -nr | head -1 | cut -d, -f1" | tail -1 | tr -d ' ')
  echo "no CD-runner TTA after 15 min: own TTA on pod GPU $G at $(TZ=America/New_York date +%H:%M)"
  $S/kx 30 "RAMP=5 setsid nohup bash experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh $G 8000 '/workspace/venv/bin/python experiments/lane_cd_20261108/source/family_tta_20261108.py $D/raw dinov3_b16 0 46 $O' > $LG/boxtrain_tta_$A.log 2>&1 < /dev/null & echo tta_started"
else echo "TTA by the CD runner ($started) at $(TZ=America/New_York date +%H:%M)"; fi
for i in $(seq 1 120); do
  $S/kx 20 "ls $OUT" > /dev/null 2>&1 && break
  sleep 30
done
$S/kx 20 "ls -la experiments/lane_cd_20261108/tta/${O}_tta4_*; tail -2 experiments/lane_cd_20261108/logs/sweep_tta2_${A}_f0_s46.log $LG/boxtrain_tta_$A.log 2>/dev/null"
echo "WATCH_DONE $(TZ=America/New_York date +%H:%M)"
