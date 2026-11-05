#!/bin/bash
# Container test of the final image (hybrid B + hex6 + stretch) against the 2026-11-01 hex6 image's outputs (2026-11-04).
set -uo pipefail
BASE=$HOME/workspace/clear-ec/final_test_20261104
G1=$HOME/workspace/clear-ec/guard_test_20261031
S2=$HOME/workspace/clear-ec/sub2_test_20261101/runs/clear-ec-hybrid-hex6_20261101
MODEL=$HOME/workspace/clear-ec/model
IMG=clear-ec-hybrid-final:20261104
run() {
  local name=$1 mha=$2; shift 2
  local out=$BASE/runs/$name
  mkdir -p "$out/input/images/corneal-specular-microscopy-image" "$out/output"
  cp "$mha" "$out/input/images/corneal-specular-microscopy-image/case.mha"
  echo '[{"socket": {"slug": "corneal-specular-microscopy-image"}}]' > "$out/input/inputs.json"
  chmod -R a+rX "$out/input"; chmod a+rwX "$out/output"
  local s=$(date +%s.%N)
  docker run --rm --gpus all --network none --memory 16g --shm-size 1g "$@" \
    -v "$MODEL":/opt/ml/model:ro -v "$out/input":/input:ro -v "$out/output":/output \
    $IMG > "$out/stdout.json" 2> "$out/stderr.txt"
  local rc=$? e=$(date +%s.%N)
  printf '%s rc=%s wall=%.1fs outputs=' "$name" "$rc" "$(echo "$e - $s" | bc)"
  for f in cell-density coefficient-of-variation hexagonality; do printf '%s,' "$(cat "$out/output/$f.json" 2>/dev/null || echo MISSING)"; done
  echo
}
for c in 0001-19 0001-20 0001-21 0234-20; do run $c "$G1/cases/$c.mha"; done
for v in rgb crop960x1280 uint16 zeros garbage; do run fault_$v "$G1/variants/$v.mha"; done
run inject_hex6 "$G1/cases/0001-19.mha" -e CLEAR_EC_FAULT_INJECT=hex6
run inject_recipea "$G1/cases/0001-19.mha" -e CLEAR_EC_FAULT_INJECT=recipea
run inject_stretch "$G1/cases/0001-19.mha" -e CLEAR_EC_FAULT_INJECT=stretch
python3 - "$BASE/runs" "$S2" <<'PY'
import json, math, pathlib, sys
new, old = map(pathlib.Path, sys.argv[1:])
K = ('cell-density', 'coefficient-of-variation', 'hexagonality')
WITH = ((1.075, 0.9996324118578636, 7.916392631948902), (1.025, 0.9999294860299096, -1.0007504165302752), (1.175, 0.9979300718545588, -0.6242508502436247))
NOHEX = (1.075, 1.0000583223332744, -0.6266355317536868)
def out(d): return [float((d / 'output' / f'{k}.json').read_text()) for k in K]
def log(d): return json.loads((d / 'stdout.json').read_text().strip().splitlines()[-1])
def st(v, hex_applied):
    p = list(WITH)
    if not hex_applied: p[2] = NOHEX
    return [c * math.exp(g * (math.log(x) - mu) + mu) for x, (g, c, mu) in zip(v, p)]
ok = True
for c in ('0001-19', '0001-20', '0001-21', '0234-20'):
    o2, o3, j = out(old / c), out(new / c), log(new / c)
    want = st(o2, True); same = all(abs(a - b) / b < 1e-5 for a, b in zip(o3, want))
    print(c, 'hex6-image output + stretch == final output:', same, '| change %', [round((a / b - 1) * 100, 2) for a, b in zip(o3, o2)], '| mode', j['mode'], '| errors', j['errors'], '| seconds %.1f' % j['seconds'], '| out', o3)
    ok &= same and j['mode'] == 'hybrid' and not j['errors'] and j['stretch']['applied'] and j['hex6']['applied']
for name in ('inject_hex6', 'inject_recipea', 'inject_stretch', 'fault_rgb', 'fault_crop960x1280', 'fault_uint16', 'fault_zeros', 'fault_garbage'):
    j = log(new / name)
    print(name, j['mode'], 'hex6', j['hex6'].get('applied'), 'stretch', j['stretch'].get('applied'), 'errors', json.dumps(j['errors'])[:130], 'out', [round(v, 4) for v in j['output']])
a = log(new / 'inject_stretch'); b = out(old / '0001-19')
same = all(abs(x - y) / y < 1e-5 for x, y in zip(a['output'], b)); print('inject_stretch equals hex6 image output:', same); ok &= same
g = log(new / 'fault_garbage'); ok &= g['mode'] == 'constant' and not g['stretch']['applied']
print('ALL CHECKS PASSED' if ok else 'CHECK FAILED')
PY
echo DONE
