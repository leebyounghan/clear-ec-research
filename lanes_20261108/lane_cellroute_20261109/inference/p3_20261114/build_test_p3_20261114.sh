#!/bin/bash
# Tier p3 / p3b on the build box (deploy lane, prepared 2026-11-14 under D-COORD-P3-PREP; run only on the PI's go). The variant is the name of the
# folder this script sits in (package_20261114/<V>/box/), so the same script serves p3 and p3b. Expects the PI's rsync of the pod folder
# package_20261114/<V> (context/, model_add/, model_swap.tsv, box/, laptop/) to ~/workspace/clear-ec/package_20261114/<V>. Steps:
#   1 build   docker build --build-arg TIER=<V> -t clear-ec:<date>-freeze-9687076-<V> (date = today, US Eastern; never 'latest'); refuses an existing tag
#   2 model   the model tree the container test mounts, package_20261114/test_model_<V>: hard links to every file of the extracted tier-p tree
#             (package_20261112/test_model_p) except the old paths of model_swap.tsv, plus hard links to the new files in model_add/ (sha256 checked
#             against model_swap.tsv). Hard links: no data is copied and nothing in test_model_p is changed. Refuses an existing test_model_<V>.
#             Then the per-file sha256 list of that tree and the patch folder for the laptop: upload_20261114/<V>_model_patch/ = model_swap.tsv,
#             model_tree_sha256_<V>.txt, rebuild_model_archive_20261114.py, files/<new path> (hard links)
#   3 test    as tier p's box test (test_tier_20261113.sh): RTX 4090, --network none, --gpus all, --memory 16g, uid 10001, one job per fresh container,
#             test_model_<V> mounted read-only; the 5 Colab frames + 3 platform debug cases in modes default (bf16) and t4path (fp16 + 1 detector
#             copy forced), plus a garbage file (fallback). Refuses if the GPU is in use.
#   4 compare with tier p's box test of the same jobs (package_20261112/test_20261113/p): CV and HEX must agree within 1e-6 relative (the GPU's
#             index_add_ is not deterministic: tier p itself moves by ~1e-9 between runs); prints the CD difference and the seconds
#   5 save    docker save | pigz -> upload_20261114/clear-ec_<date>-freeze-9687076-<V>.tar.gz + SHA256SUMS_<V>.txt
#   archive   (not in the default steps; option (a) only) the full model archive from test_model_<V> packed as tier p's was (pack_tier_20261113.sh:
#             --sort=name, owner 0:0, a+rX, pigz), its per-file list compared with model_tree_sha256_<V>.txt, sha256 into SHA256SUMS_<V>.txt
# No registry push, no upload. usage: build_test_p3_20261114.sh [STEPS]   (default "build model test compare save")
set -uo pipefail
STEPS=${1:-build model test compare save}
P=$(cd "$(dirname "$0")/.." && pwd); V=$(basename "$P")
case "$V" in p3|p3b) ;; *) echo "unexpected variant folder $P"; exit 1;; esac
D=${D:-$(TZ=America/New_York date +%Y%m%d)}; IMG=clear-ec:$D-freeze-9687076-$V
W=$HOME/workspace/clear-ec; B=$W/package_20261112; MODEL_P=$B/test_model_p; MODEL=$W/package_20261114/test_model_$V
R=$W/package_20261114/test_$V; U=$W/package_20261114/upload_20261114; RP=$B/test_20261113/p; PATCH=$U/${V}_model_patch
FR=$B/frames_colab5; CASES=$W/guard_test_20261031/cases; VAR=$W/guard_test_20261031/variants
RB=$P/laptop/rebuild_model_archive_20261114.py
case " $STEPS " in *" build "*)
  docker image inspect "$IMG" > /dev/null 2>&1 && { echo "refusing: $IMG already exists"; exit 1; }
  test -f "$P/context/code/experiments/lane_cellroute_20261109/inference/manifests/deploy_$V.json" || { echo "no $V context at $P (rsync first)"; exit 1; }
  echo "== build $IMG $(date)"
  docker build --build-arg TIER=$V -t "$IMG" "$P/context" > "$P/build_${V}_$D.log" 2>&1; rc=$?
  tail -3 "$P/build_${V}_$D.log"; [ $rc -eq 0 ] || { echo "build failed rc $rc, see $P/build_${V}_$D.log"; exit 1; }
  docker image inspect "$IMG" --format 'image id {{.Id}} size {{.Size}}'
  docker run --rm --entrypoint cat "$IMG" /opt/build/TIER;;
esac
case " $STEPS " in *" model "*)
  [ -d "$MODEL_P" ] || { echo "no extracted tier-p model $MODEL_P"; exit 1; }
  [ -e "$MODEL" ] && { echo "refusing: $MODEL exists"; exit 1; }
  [ -e "$PATCH" ] && { echo "refusing: $PATCH exists"; exit 1; }
  echo "== model tree $MODEL $(date)"
  python3 - "$MODEL_P" "$P/model_add" "$P/model_swap.tsv" "$MODEL" "$PATCH" <<'PY' || exit 1
import os, sys, stat, hashlib
src, add, swap, dst, patch = sys.argv[1:6]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
rows = [l.rstrip('\n').split('\t') for l in open(swap) if l.strip() and not l.startswith('#')]
rm = {o for o, n, s in rows}; seen = set(); n = b = 0
for o, nw, s in rows:
    f = os.path.join(add, nw); assert sha(f) == s, ('new file does not match model_swap.tsv', f)
for dp, dn, fn in os.walk(src):
    for x in fn:
        p = os.path.join(dp, x); rel = os.path.relpath(p, src); assert stat.S_ISREG(os.lstat(p).st_mode), ('not a regular file', p)
        if rel in rm: seen.add(rel); continue
        q = os.path.join(dst, rel); os.makedirs(os.path.dirname(q), exist_ok=True); os.link(p, q); n += 1; b += os.path.getsize(p)
assert seen == rm, ('old paths not in the tier-p tree', rm - seen)
for o, nw, s in rows:
    for base in (dst, os.path.join(patch, 'files')):
        q = os.path.join(base, nw); os.makedirs(os.path.dirname(q), exist_ok=True); os.link(os.path.join(add, nw), q)
    n += 1; b += os.path.getsize(os.path.join(add, nw))
print(f'{dst}: {n} files {b:,} bytes ({len(rm)} old path(s) left out, {len(rows)} new file(s) linked)')
PY
  chmod -R a+rX "$PATCH"
  cp "$P/model_swap.tsv" "$RB" "$PATCH/"
  python3 "$RB" --list-dir "$MODEL" "$PATCH/model_tree_sha256_$V.txt" || { echo "model tree check failed"; exit 1; }
  (cd "$PATCH" && sha256sum model_tree_sha256_$V.txt model_swap.tsv rebuild_model_archive_20261114.py && find files -type f | LC_ALL=C sort | xargs -d '\n' sha256sum) > "$U/SHA256SUMS_${V}_model_patch.txt"
  cat "$U/SHA256SUMS_${V}_model_patch.txt";;
esac
case " $STEPS " in *" test "*)
  [ -d "$MODEL" ] || { echo "no model tree $MODEL (run the model step)"; exit 1; }
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$used" -lt 1000 ] || { echo "refusing: GPU in use ($used MiB)"; exit 1; }
  evict() { python3 - "$MODEL" <<'PY'
import os, sys
n = b = 0
for dp, dn, fn in os.walk(sys.argv[1]):
    for f in fn:
        p = os.path.join(dp, f); fd = os.open(p, os.O_RDONLY)
        try: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); n += 1; b += os.fstat(fd).st_size
        finally: os.close(fd)
print(f'evicted {n} files {b/1e9:.2f} GB from the page cache (best effort)')
PY
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
    printf '%s %s rc=%s wall=%.1fs gpu_peak_mib=%s outputs=' "$mode" "$name" "$rc" "$(echo "$e - $s" | bc)" "$peak"
    for f in cell-density coefficient-of-variation hexagonality; do printf '%s,' "$(cat "$out/output/$f.json" 2>/dev/null || echo MISSING)"; done
    echo
    echo "{\"tier\": \"$V\", \"mode\": \"$mode\", \"name\": \"$name\", \"rc\": $rc, \"wall_s\": $(echo "$e - $s" | bc), \"gpu_device_peak_mib\": ${peak:-null}}" > "$out/job.json"
  }
  echo "== test $(date) image $(docker image inspect $IMG --format '{{.Id}}')"
  for m in default t4path; do
    ENVA=(); [ "$m" = t4path ] && ENVA=(-e CLEAR_EC_AUTOCAST=fp16 -e CLEAR_EC_DET_COPIES=1)
    evict; first=cold
    for f in "$FR"/*.mha; do run "$m" "$(basename "$f" .mha)_$first" "$f" "${ENVA[@]}"; first=warm; done
    for c in 0001-19 0001-20 0001-21; do run "$m" "debug_$c" "$CASES/$c.mha" "${ENVA[@]}"; done
    [ "$m" = default ] && run "$m" fault_garbage "$VAR/garbage.mha"
  done;;
esac
case " $STEPS " in *" compare "*)
  echo "== compare with tier p's box test ($RP)"
  python3 - "$R" "$RP" "$V" <<'PY'
import json, os, sys
R, RP, V = sys.argv[1], sys.argv[2], sys.argv[3]; bad = []; worst = 0.0
def outs(d): return [json.load(open(f'{d}/output/{k}.json')) if os.path.exists(f'{d}/output/{k}.json') else None for k in ('cell-density', 'coefficient-of-variation', 'hexagonality')]
def info(d):
    try: j = json.loads(open(f'{d}/stdout.json').read().strip().splitlines()[-1]); return j.get('seconds'), j.get('ram_peak_gb'), (j.get('hooks') or {}).get('ro_hits'), j.get('mode')
    except Exception: return None, None, None, None
for mode in ('default', 't4path'):
    for name in sorted(os.listdir(f'{R}/{mode}')):
        a, b = outs(f'{R}/{mode}/{name}'), outs(f'{RP}/{mode}/{name}') if os.path.isdir(f'{RP}/{mode}/{name}') else [None] * 3
        sa, ra, ha, ma = info(f'{R}/{mode}/{name}'); sb, rb, _, mb = info(f'{RP}/{mode}/{name}')
        rel = [abs(x - y) / abs(y) if None not in (x, y) and y else (0.0 if x == y else float('inf')) for x, y in zip(a[1:], b[1:])]
        ok = all(r <= 1e-6 for r in rel); worst = max([worst] + [r for r in rel if r != float('inf')])
        cd = (a[0] - b[0]) if None not in (a[0], b[0]) else None
        if not ok and name != 'fault_garbage': bad.append(f'{mode}/{name}')
        print(f'{mode:8s} {name:28s} {V} {a}  p {b}  CD diff {cd}  CV/HEX max rel diff {max(rel) if rel else None:.2g}  seconds {V} {sa} p {sb}  ram {V} {ra} p {rb}  mode {ma}/{mb}')
print(f'CV/HEX beyond 1e-6 relative on: {bad or "none"}; largest CV/HEX relative difference {worst:.2g}')
PY
  ;;
esac
case " $STEPS " in *" save "*)
  mkdir -p "$U"; IMGF=$U/clear-ec_$D-freeze-9687076-$V.tar.gz
  [ -e "$IMGF" ] && { echo "refusing: $IMGF exists"; exit 1; }
  echo "== save $(date)"
  docker save "$IMG" | pigz -p 12 > "$IMGF.part" && mv "$IMGF.part" "$IMGF"
  (cd "$U" && sha256sum "$(basename $IMGF)" | tee SHA256SUMS_$V.txt); ls -l "$U";;
esac
case " $STEPS " in *" archive "*)
  MODF=$U/model_freeze-9687076-${V}_$D.tar.gz
  [ -e "$MODF" ] && { echo "refusing: $MODF exists"; exit 1; }
  echo "== archive (option a) $(date)"
  tar -C "$MODEL" --sort=name --owner=0 --group=0 --numeric-owner --mode=a+rX --hard-dereference -cf - . | pigz -p 12 > "$MODF.part" && mv "$MODF.part" "$MODF"
  python3 "$RB" --list-tar "$MODF" "$MODF.files.txt" && cmp "$MODF.files.txt" "$PATCH/model_tree_sha256_$V.txt" && echo "archive contents = tested tree" || { echo "archive contents differ"; exit 1; }
  (cd "$U" && sha256sum "$(basename $MODF)" | tee -a SHA256SUMS_$V.txt);;
esac
echo "${V}_DONE [$STEPS] $(date)"
