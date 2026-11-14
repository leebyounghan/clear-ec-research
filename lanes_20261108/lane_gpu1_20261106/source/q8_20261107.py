from pathlib import Path
L = Path('/workspace/clear-ec-research/experiments/lane_gpu1_20261106')
p = L/'source/run_e2e_v2_20261107.sh'
s = p.read_text()
for a, b in (
 ("""CUDA_VISIBLE_DEVICES= PYTHONPATH=$PP $PY $L/source/$W prepare $A || fail prepare $?
""", """if [ -f $ROOT/manifest.json ]; then echo "manifest exists, not prepared again"; else CUDA_VISIBLE_DEVICES= PYTHONPATH=$PP $PY $L/source/$W prepare $A || fail prepare $?; fi
"""),
 ("""CUDA_VISIBLE_DEVICES= PYTHONPATH=$RT $PY $FZ cpu-tests $A || fail cpu-tests $?
""", """if [ -f $ROOT/cpu_tests.json ]; then echo "cpu tests already passed for this manifest"; else CUDA_VISIBLE_DEVICES= PYTHONPATH=$RT $PY $FZ cpu-tests $A || fail cpu-tests $?; fi
"""),
 ("""CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$RT $PY $FZ smoke $A || fail smoke $?
""", """if [ -f $ROOT/smoke.json ]; then echo "smoke already passed for this manifest"; else CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$RT $PY $FZ smoke $A || fail smoke $?; fi
"""),
 ("""GPU=${CUDA_VISIBLE_DEVICES:-1}
""", """GPU=${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to this lane's GPU}
"""),
):
    assert a in s, a
    s = s.replace(a, b)
p.write_text(s)
print(s)
