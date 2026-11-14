#!/bin/bash
# Pull one box arm's small outputs back to the pod for the CD lane's read (deploy lane, 2026-11-13, A-DEPLOY-BOX-TRAIN). Run on the main pod.
# Copies predictions_*.npz, history.json, result.json, box_launch.json, smoke.log and train.log (no .pt weights; those stay on the box unless
# asked) from qudgks@210.107.192.239:workspace/clear-ec/train_20261113/arms/<ARM>_f0_s46/ into DEST (default
# experiments/lane_cd_20261108/arms/<ARM>_f0_s46, the same tree layout as the pod arms: raw/runs/fold0/dinov3_b16/...), with the key the PI
# registered. Refuses an existing DEST unless the run there came from the box (box_launch.json present), so a pod arm is never overwritten.
# WITH_PT=1 also pulls the kept epoch_XX.pt and final.pt (5 + 1 x 343 MB) for the CD lane's four-flip TTA read of epochs 16-20 on the pod.
# usage: [WITH_PT=1] pull_arm_20261113.sh ARM [DEST]
set -euo pipefail
ARM=$1; cd /workspace/clear-ec-research; DEST=${2:-experiments/lane_cd_20261108/arms/${ARM}_f0_s46}
if [ -e "$DEST" ] && [ ! -f "$DEST/box_launch.json" ]; then echo "refusing: $DEST exists and is not a box arm"; exit 1; fi
mkdir -p "$DEST"
rsync -a --info=stats1 -e "ssh -o BatchMode=yes -o ConnectTimeout=15 -i /root/.ssh/buildbox_20261112" \
  --include='*/' --include='predictions_*.npz' --include='history.json' --include='result.json' --include='box_launch.json' \
  --include='smoke.log' --include='train.log' --include='complete.json' --include='exit.json' ${WITH_PT:+--include='epoch_*.pt' --include='final.pt'} --exclude='*' --prune-empty-dirs \
  "qudgks@210.107.192.239:workspace/clear-ec/train_20261113/arms/${ARM}_f0_s46/" "$DEST/"
tail -1 "$DEST/train.log" 2>/dev/null; find "$DEST" -name 'predictions_*.npz' | wc -l
