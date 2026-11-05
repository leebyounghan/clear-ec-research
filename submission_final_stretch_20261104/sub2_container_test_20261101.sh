#!/bin/bash
# Container test of the submission-2 image (hybrid B + hex6) against the submission-1 image (2026-11-01).
set -uo pipefail
BASE=$HOME/workspace/clear-ec/sub2_test_20261101
G1=$HOME/workspace/clear-ec/guard_test_20261031
MODEL=$HOME/workspace/clear-ec/model
run() {  # image name mha [docker env args...]
  local image=$1 name=$2 mha=$3; shift 3
  local out=$BASE/runs/${image//[:\/]/_}/$name
  rm -rf "$out"; mkdir -p "$out/input/images/corneal-specular-microscopy-image" "$out/output"
  cp "$mha" "$out/input/images/corneal-specular-microscopy-image/case.mha"
  echo '[{"socket": {"slug": "corneal-specular-microscopy-image"}}]' > "$out/input/inputs.json"
  chmod -R a+rX "$out/input"; chmod a+rwX "$out/output"
  local s=$(date +%s.%N)
  docker run --rm --gpus all --network none --memory 16g --shm-size 1g "$@" \
    -v "$MODEL":/opt/ml/model:ro -v "$out/input":/input:ro -v "$out/output":/output \
    $image > "$out/stdout.json" 2> "$out/stderr.txt"
  local rc=$? e=$(date +%s.%N)
  printf '%s %s rc=%s wall=%.1fs outputs=' "$image" "$name" "$rc" "$(echo "$e - $s" | bc)"
  for f in cell-density coefficient-of-variation hexagonality; do printf '%s,' "$(cat "$out/output/$f.json" 2>/dev/null || echo MISSING)"; done
  echo
}
S2=clear-ec-hybrid-hex6:20261101
echo "== normal cases (sub2) vs sub1 outputs from guard_test_20261031"
for c in 0001-19 0001-20 0001-21 0234-20; do run $S2 $c "$G1/cases/$c.mha"; done
echo "== faults"
for v in rgb crop960x1280 uint16 zeros garbage; do run $S2 fault_$v "$G1/variants/$v.mha"; done
run $S2 inject_hex6 "$G1/cases/0001-19.mha" -e CLEAR_EC_FAULT_INJECT=hex6
run $S2 inject_recipea "$G1/cases/0001-19.mha" -e CLEAR_EC_FAULT_INJECT=recipea
echo "== checks"
python3 - "$BASE" "$G1" <<'PY'
import json, sys, pathlib
base, g1 = map(pathlib.Path, sys.argv[1:])
def out(d):
    return [open(d / 'output' / f'{k}.json').read().strip() for k in ('cell-density', 'coefficient-of-variation', 'hexagonality')]
def log(d):
    return json.loads(open(d / 'stdout.json').read().strip().splitlines()[-1])
r2 = base / 'runs' / 'clear-ec-hybrid-hex6_20261101'
ok = True
for c in ('0001-19', '0001-20', '0001-21', '0234-20'):
    o1, o2, j = out(g1 / 'runs' / '20261031' / c), out(r2 / c), log(r2 / c)
    same_cdcv = o1[:2] == o2[:2]
    before_eq = abs(j['hex6']['hex_before'] - float(o1[2])) == 0.0
    print(c, 'CD/CV byte-identical:', same_cdcv, '| hex before == sub1 HEX:', before_eq, '| hex6 applied:', j['hex6']['applied'],
          '| HEX sub1 -> sub2: %s -> %s (%.2f %%)' % (o1[2], o2[2], (float(o2[2]) / float(o1[2]) - 1) * 100), '| seconds %.1f' % j['seconds'])
    ok &= same_cdcv and before_eq and j['hex6']['applied'] and j['mode'] == 'hybrid' and not j['errors']
for name in ('inject_hex6', 'inject_recipea', 'fault_rgb', 'fault_crop960x1280', 'fault_uint16', 'fault_zeros', 'fault_garbage'):
    j = log(r2 / name)
    print(name, j['mode'], 'hex6', json.dumps(j.get('hex6'))[:120], 'errors', json.dumps(j['errors'])[:150])
j = log(r2 / 'inject_hex6'); o1 = out(g1 / 'runs' / '20261031' / '0001-19'); o2 = out(r2 / 'inject_hex6')
print('inject_hex6 equals sub1 output byte for byte:', o1 == o2); ok &= o1 == o2
print('ALL NORMAL + HEX6-FAULT CHECKS PASSED' if ok else 'CHECK FAILED')
PY
echo DONE
