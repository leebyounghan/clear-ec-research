#!/bin/bash
# GPU 0/3 copy of tta_after_pull.sh (coordinator 02:16 EST 11-14: recipe A runs alone on GPUs 1 and 2); guard needs 8000 MiB free (GPUs 0/3 have ~15.4 GB free beside ViT-L; batch-16 bf16 inference). CD-lane read form for a box arm (2026-11-13): after the chain has pulled the arm (pod folder train.log ends 'EXIT 0'), pull its kept
# epoch_16-20.pt too (WITH_PT=1, detached on the pod), then run the CD lane's four-flip TTA (family_tta_20261108.py, 1 loader worker) on the
# pod GPU with the most free memory through gpu_launch_20261111.sh; wait for experiments/lane_cd_20261108/tta/<OUT>_tta4_e16_20.npz.
# usage: tta_after_pull.sh ARM OUTNAME
S=/private/tmp/claude-501/-Users-byounghanlee-Desktop-k8s-server/3fe760b8-2aec-41ed-af70-e1afef1b7f06/scratchpad
A=$1; O=$2; D=experiments/lane_cd_20261108/arms/${A}_f0_s46; LG=experiments/lane_cellroute_20261109/logs
st=""
for i in $(seq 1 360); do
  st=$($S/kx 20 "tail -1 $D/train.log 2>/dev/null" 2>/dev/null | tail -1)
  [ "$st" = "EXIT 0" ] && break
  sleep 60
done
echo "pod copy of $A: '${st}' at $(TZ=America/New_York date +%H:%M)"; [ "$st" = "EXIT 0" ] || exit 1
$S/kx 30 "WITH_PT=1 setsid nohup bash experiments/lane_cellroute_20261109/inference/boxtrain_20261113/pull_arm_20261113.sh $A > $LG/boxtrain_pullpt_$A.log 2>&1 < /dev/null & echo pull_started"
for i in $(seq 1 120); do
  n=$($S/kx 20 "ls $D/raw/runs/fold0/dinov3_b16/epoch_{16,17,18,19,20}.pt 2>/dev/null | wc -l; pgrep -f '[p]ull_arm_20261113.sh $A' > /dev/null && echo RUN || echo DONE" 2>/dev/null | tr '\n' ' ')
  case "$n" in "5 DONE"*) break;; esac
  sleep 15
done
echo "weights on pod: $n"; $S/kx 20 "tail -3 $LG/boxtrain_pullpt_$A.log"
case "$n" in "5 DONE"*) ;; *) echo "weights incomplete"; exit 1;; esac
G=$($S/kx 20 "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | grep -E '^ *(0|3),' | sort -t, -k2 -nr | head -1 | cut -d, -f1" | tail -1 | tr -d ' ')
echo "TTA on pod GPU $G"
$S/kx 30 "RAMP=5 setsid nohup bash experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh $G 8000 '/workspace/venv/bin/python experiments/lane_cd_20261108/source/family_tta_20261108.py $D/raw dinov3_b16 0 46 $O' > $LG/boxtrain_tta_$A.log 2>&1 < /dev/null & echo tta_started"
for i in $(seq 1 120); do
  $S/kx 20 "ls experiments/lane_cd_20261108/tta/${O}_tta4_e16_20.npz" > /dev/null 2>&1 && break
  sleep 20
done
$S/kx 20 "ls -la experiments/lane_cd_20261108/tta/${O}_tta4_*; tail -3 $LG/boxtrain_tta_$A.log"
echo "TTA_DONE $A $(TZ=America/New_York date +%H:%M)"
