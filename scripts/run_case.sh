#!/usr/bin/env bash
# Run the container on one image the way Grand Challenge does (one image per fresh container, no network, read-only weights).
# Usage: scripts/run_case.sh <image.mha> <output_dir> [--t4path] [--image TAG] [--checkpoints DIR]
#   --t4path  force the code path the platform T4 takes (fp16 autocast, one detector copy). On a T4 it is selected automatically.
set -euo pipefail
cd "$(dirname "$0")/.."
MHA=${1:?image.mha}; OUT=${2:?output_dir}; shift 2
TAG=clear-ec-p3b:local; CKPT=$PWD/checkpoints; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --t4path) EXTRA+=(-e CLEAR_EC_AUTOCAST=fp16 -e CLEAR_EC_DET_COPIES=1);;
    --image) TAG=$2; shift;;
    --checkpoints) CKPT=$2; shift;;
    *) echo "unknown option $1"; exit 2;;
  esac; shift
done
[ -d "$CKPT/cpsam_models" ] || { echo "checkpoints not found in $CKPT (see checkpoints/README.md)"; exit 1; }
[ -e "$OUT/cell-density.json" ] && { echo "$OUT already holds a prediction"; exit 1; }
IN=$(mktemp -d)
mkdir -p "$IN/images/corneal-specular-microscopy-image" "$OUT"
cp examples/inputs.json "$IN/inputs.json"
cp "$MHA" "$IN/images/corneal-specular-microscopy-image/"
chmod -R a+rX "$IN"; chmod a+rwx "$OUT"
docker run --rm --gpus all --network none --memory 16g --shm-size 1g "${EXTRA[@]}" \
  -v "$CKPT":/opt/ml/model:ro -v "$IN":/input:ro -v "$(cd "$OUT" && pwd)":/output "$TAG" > "$OUT/run_log.json"
rm -rf "$IN"
echo "$(basename "$MHA"): CD $(cat "$OUT/cell-density.json")  CV $(cat "$OUT/coefficient-of-variation.json")  HEX $(cat "$OUT/hexagonality.json")"
