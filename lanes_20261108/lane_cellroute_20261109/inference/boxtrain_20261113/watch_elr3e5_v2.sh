#!/bin/bash
# sweep_elr3e5 watcher, second version (02:16 EST 11-14). The first (watch_elr3e5.sh) never matched: H=~/... expanded the tilde on the Mac,
# so the ssh grep looked for /Users/... on the box. Here the box path is relative to the box home. Waits for the EXIT line, pulls to the pod,
# then the CD-lane four-flip TTA on pod GPU 0 or 3 only (tta_after_pull_g03.sh). The epoch-12 pull is done by hand before this starts.
S=/private/tmp/claude-501/-Users-byounghanlee-Desktop-k8s-server/3fe760b8-2aec-41ed-af70-e1afef1b7f06/scratchpad
A=sweep_elr3e5; H=workspace/clear-ec/train_20261113/arms/${A}_f0_s46
PULL="bash experiments/lane_cellroute_20261109/inference/boxtrain_20261113/pull_arm_20261113.sh $A"
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
$S/kx 300 "$PULL"
[ "$line" = "EXIT 0" ] || { echo "not EXIT 0: no TTA"; exit 1; }
bash $S/boxtrain/tta_after_pull_g03.sh $A ${A}_f0
echo "WATCH_DONE $(TZ=America/New_York date +%H:%M)"
