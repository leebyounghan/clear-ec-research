#!/bin/bash
# Container test of one tier image as Grand Challenge runs it (deploy lane, 2026-11-13): RTX 4090 (sm89), --network none, --gpus all, --memory 16g,
# --shm-size 1g, the model archive EXTRACTED (test_model_<tier>, written by pack_tier_20261113.sh) mounted read-only at /opt/ml/model, input
# read-only, one job per fresh container, the image's own user (uid 10001).
#   default : the image as built (sm89 -> bf16, 4 detector copies): the 5 Colab frames, the 3 platform debug cases, a garbage file (fallback)
#   t4path  : the T4 container default forced on the 4090 (CLEAR_EC_AUTOCAST=fp16 + CLEAR_EC_DET_COPIES=1 = the branch pack_hooks takes below
#             compute capability 8.0): the same 5 frames and 3 debug cases
# The first job of each mode starts with the model files evicted from the page cache (posix_fadvise DONTNEED, best effort, no root) = cold;
# the rest are warm. GPU memory: nvidia-smi every 250 ms during the job (device total, baseline ~16 MiB). Writes test_20261113/<tier>/.
# usage: test_tier_20261113.sh TIER [MODES]   (MODES default "default t4path")
set -uo pipefail
T=$1; MODES=${2:-default t4path}; B=$HOME/workspace/clear-ec/package_20261112; IMG=clear-ec:20261112-freeze-9687076-$T
MODEL=$B/test_model_$T; R=$B/test_20261113/$T; FR=$B/frames_colab5; CASES=$HOME/workspace/clear-ec/guard_test_20261031/cases
VAR=$HOME/workspace/clear-ec/guard_test_20261031/variants
[ -d "$MODEL" ] || { echo "no extracted model $MODEL"; exit 1; }
evict() { python3 - "$MODEL" <<'EOF'
import os, sys
n = b = 0
for dp, dn, fn in os.walk(sys.argv[1]):
    for f in fn:
        p = os.path.join(dp, f); fd = os.open(p, os.O_RDONLY)
        try: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); n += 1; b += os.fstat(fd).st_size
        finally: os.close(fd)
print(f'evicted {n} files {b/1e9:.2f} GB from the page cache (best effort)')
EOF
}
run() {  # mode name mha [docker env args...]
  local mode=$1 name=$2 mha=$3; shift 3
  local out=$R/$mode/$name
  rm -rf "$out"; mkdir -p "$out/input/images/corneal-specular-microscopy-image" "$out/output"
  cp "$mha" "$out/input/images/corneal-specular-microscopy-image/$(basename "$mha")"
  echo '[{"socket": {"slug": "corneal-specular-microscopy-image"}}]' > "$out/input/inputs.json"
  chmod -R a+rX "$out/input"; chmod a+rwX "$out/output"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 250 > "$out/gpu_mem.txt" 2>/dev/null &
  local smi=$!
  local s=$(date +%s.%N)
  docker run --rm --gpus all --network none --memory 16g --shm-size 1g "$@" \
    -v "$MODEL":/opt/ml/model:ro -v "$out/input":/input:ro -v "$out/output":/output "$IMG" > "$out/stdout.json" 2> "$out/stderr.txt"
  local rc=$? e=$(date +%s.%N)
  kill $smi 2>/dev/null; wait $smi 2>/dev/null
  local peak=$(sort -n "$out/gpu_mem.txt" | tail -1)
  printf '%s %s %s rc=%s wall=%.1fs gpu_peak_mib=%s outputs=' "$T" "$mode" "$name" "$rc" "$(echo "$e - $s" | bc)" "$peak"
  for f in cell-density coefficient-of-variation hexagonality; do printf '%s,' "$(cat "$out/output/$f.json" 2>/dev/null || echo MISSING)"; done
  echo
  echo "{\"tier\": \"$T\", \"mode\": \"$mode\", \"name\": \"$name\", \"rc\": $rc, \"wall_s\": $(echo "$e - $s" | bc), \"gpu_device_peak_mib\": ${peak:-null}}" > "$out/job.json"
}
echo "start $(date) image $(docker image inspect $IMG --format '{{.Id}}')"
for m in $MODES; do
  ENVA=(); [ "$m" = t4path ] && ENVA=(-e CLEAR_EC_AUTOCAST=fp16 -e CLEAR_EC_DET_COPIES=1)
  evict
  first=cold
  for f in "$FR"/*.mha; do run "$m" "$(basename "$f" .mha)_$first" "$f" "${ENVA[@]}"; first=warm; done
  for c in 0001-19 0001-20 0001-21; do run "$m" "debug_$c" "$CASES/$c.mha" "${ENVA[@]}"; done
  [ "$m" = default ] && run "$m" fault_garbage "$VAR/garbage.mha"
done
echo "== stdout summary"
for d in "$R"/*/*; do printf '%s ' "${d#$R/}"; python3 -c "
import json,sys
try:
    j=json.loads(open('$d/stdout.json').read().strip().splitlines()[-1]); h=j.get('hooks') or {}
    print(j['mode'], 'seconds', j['seconds'], 'setup', j.get('setup_s'), 'ram', j.get('ram_peak_gb'), 'gpu_alloc', j.get('gpu_peak_gb'), 'autocast', h.get('autocast'), 'copies', h.get('det_copies'), 't4_default', h.get('t4_default'), 'read_s', round(h.get('read_s', 0), 1), 'errors', json.dumps(j.get('errors'))[:200])
except Exception as e: print('NO JSON', e, open('$d/stderr.txt').read()[-300:])
"; done
echo "TEST_DONE $T $(date)"
