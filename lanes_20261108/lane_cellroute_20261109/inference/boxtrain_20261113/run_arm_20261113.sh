#!/bin/bash
# One bb_runner_20261108.py dinov3_b16 fold-0 seed-46 arm on the build box RTX 4090 (deploy lane, 2026-11-13, A-DEPLOY-BOX-TRAIN).
# The bundle (bundle_files_20261113.txt, rsynced by the PI to ~/workspace/clear-ec/train_20261113/) is mounted at /workspace/clear-ec-research
# inside the existing image clear-ec:20261112-freeze-9687076-p (torch 2.11.0+cu126; every import the runner needs is in it). Root in the
# container (outputs root-owned), --network none, one GPU, workers = the box's CPU count (WORKERS overrides), --shm-size 32g for the loaders.
# micro 8: the runner's DataLoader batch is fixed at 16 and each batch is split into --micro micro-batches with gradient accumulation, so the
# effective batch stays 16. Peak at micro 8: 18.53 GiB allocated on the H200 (smoke incl. TTA eval), 18.34 / 18.55 GiB allocated / reserved in
# the 4090 synthetic step benchmark (of 23.52 GiB); micro 16 runs out of memory on the 4090.
# Steps: (1) verify every bundle file (size; sha256 below 5 MB) against bundle_files_20261113.tsv; (2) refuse if the arm folder already has
# outputs or the GPU is in use; (3) smoke: --smoke 2 with the same switches in a fresh container (2 updates + TTA check, writes nothing);
# (4) the run, detached container cec_train_<ARM>, log arms/<ARM>_f0_s46/train.log (last line 'EXIT <rc>').
# usage: run_arm_20261113.sh ARM [extra runner switches]   e.g. run_arm_20261113.sh screen_lr2e5llrd075 --enc-lr 2e-5 --llrd 0.75
set -euo pipefail
ARM=$1; shift
B=$HOME/workspace/clear-ec/train_20261113; IMG=clear-ec:20261112-freeze-9687076-p; WORKERS=${WORKERS:-$(nproc)}; SMOKE=${SMOKE:-2}
ROOT=arms/${ARM}_f0_s46; NAME=cec_train_${ARM}
ARGS=(--family dinov3_b16 --fold 0 --seed 46 --ssl experiments/lane_gpu3_20261106/runs/dinov3_b16_f0_s46/ssl/last.pt --schedule-epochs 10
      --micro 8 --workers "$WORKERS" --tta-final --epochs 20 --keep-epochs 16,17,18,19,20 "$@")
echo "== verify bundle $(date)"
python3 - "$B" "$B/bundle_files_20261113.tsv" <<'EOF'
import os, sys, hashlib
b, tsv = sys.argv[1], sys.argv[2]; bad = []; n = tot = 0
for line in open(tsv):
    r, size, h = line.rstrip('\n').split('\t'); p = os.path.join(b, r); n += 1
    if not os.path.isfile(p): bad.append(('missing', r)); continue
    if os.path.getsize(p) != int(size): bad.append(('size', r)); continue
    if h and hashlib.sha256(open(p, 'rb').read()).hexdigest() != h: bad.append(('sha256', r))
    tot += int(size)
print(f'{n} files, {tot:,} bytes checked; problems: {len(bad)}', bad[:5]); sys.exit(1 if bad else 0)
EOF
[ -e "$B/$ROOT/raw" ] && { echo "refusing: $B/$ROOT/raw exists (no overwrite)"; exit 1; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$used" -lt 1000 ] || { echo "refusing: GPU in use ($used MiB)"; exit 1; }
docker ps --format '{{.Names}}' | grep -qx "$NAME" && { echo "refusing: container $NAME running"; exit 1; }
mkdir -p "$B/$ROOT"
DOCKER=(docker run --gpus all --network none --user 0:0 --shm-size 32g -e PYTHONUNBUFFERED=1
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w /workspace/clear-ec-research -v "$B":/workspace/clear-ec-research --entrypoint bash)
RUNNER=experiments/lane_backbone_20261108/source/bb_runner_20261108.py
if [ "$SMOKE" -gt 0 ]; then
  echo "== smoke ($SMOKE updates, writes nothing) $(date)"
  "${DOCKER[@]}" --rm "$IMG" -c "python -u $RUNNER ${ARGS[*]} --root $ROOT/smoke_root --smoke $SMOKE" 2>&1 | tail -5 | tee "$B/$ROOT/smoke.log"
  grep -q '"smoke": "done"' "$B/$ROOT/smoke.log" || { echo "smoke failed, see $B/$ROOT/smoke.log"; exit 1; }
fi
printf '{"arm": "%s", "image": "%s", "image_id": "%s", "workers": %s, "args": "%s", "started": "%s"}\n' "$ARM" "$IMG" \
  "$(docker image inspect $IMG --format '{{.Id}}')" "$WORKERS" "${ARGS[*]} --root $ROOT" "$(date -u +%FT%TZ)" > "$B/$ROOT/box_launch.json"
echo "== run $(date)"
"${DOCKER[@]}" -d --rm --name "$NAME" "$IMG" -c "python -u $RUNNER ${ARGS[*]} --root $ROOT > $ROOT/train.log 2>&1; echo EXIT \$? >> $ROOT/train.log"
echo "started container $NAME; log $B/$ROOT/train.log; outputs $B/$ROOT/raw/runs/fold0/dinov3_b16/"
