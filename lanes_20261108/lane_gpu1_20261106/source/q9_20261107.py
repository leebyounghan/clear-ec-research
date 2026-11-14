import subprocess
from pathlib import Path
L = Path('/workspace/clear-ec-research/experiments/lane_gpu1_20261106')
run = subprocess.run(['pgrep','-f','run_e2e_v2_20261107'],capture_output=True,text=True).stdout.split()
assert not run, f'v2 is running: {run}; not editing'
p = L/'source/run_e2e_v2_20261107.sh'
s = p.read_text()
a = """if [ -f $SSL/run/complete.json ]; then echo "regression_input exists, not rewritten"; else"""
b = """# Keep regression_input only if it was built after this arm's SSL finished: GPU 2 built the fold 0 one from epoch 1 of 5 by mistake (2026-11-07).
[ -f $L/ssl/$ARM/run/complete.json ] || fail ssl-not-complete 3
if [ -f $SSL/run/complete.json ] && [ $SSL/run/complete.json -nt $L/ssl/$ARM/run/complete.json ]; then echo "regression_input exists and is newer than the SSL, not rewritten"; else"""
assert a in s
p.write_text(s.replace(a,b))
print('v2 guarded')
