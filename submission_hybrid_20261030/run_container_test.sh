#!/bin/bash
# Container smoke test on the build box: one authorised MHA through the hybrid image with the model mount.
# Usage: ./run_container_test.sh <case.mha> [out_dir]   — emulates the Grand Challenge layout and a T4 memory bound.
set -euo pipefail
MHA=${1:?path to a uint8 972x1296 .mha}; OUT=${2:-$HOME/workspace/clear-ec/container_test_$(date -u +%Y%m%dT%H%M%SZ)}
MODEL=$HOME/workspace/clear-ec/model
mkdir -p "$OUT/input/images/corneal-specular-microscopy-image" "$OUT/output"
cp "$MHA" "$OUT/input/images/corneal-specular-microscopy-image/case.mha"
cat > "$OUT/input/inputs.json" <<'JSON'
[{"socket": {"slug": "corneal-specular-microscopy-image"}}]
JSON
chmod -R a+rX "$OUT/input"; chmod a+rwX "$OUT/output"
# --gpus with a 16 GiB-class bound is not available on a 4090; the entrypoint itself asserts the SDPA kernel and the
# integration test measured 2.6 GiB peak, so here we check correctness, wall time and that nothing reads the network.
/usr/bin/time -v docker run --rm --gpus all --network none --memory 16g --shm-size 1g \
  -v "$MODEL":/opt/ml/model:ro -v "$OUT/input":/input:ro -v "$OUT/output":/output \
  clear-ec-hybrid:20261030 2> "$OUT/time.txt" | tee "$OUT/stdout.json"
echo "--- outputs"; for f in cell-density coefficient-of-variation hexagonality; do printf '%s = ' "$f"; cat "$OUT/output/$f.json"; echo; done
grep -E "Elapsed|Maximum resident" "$OUT/time.txt"
