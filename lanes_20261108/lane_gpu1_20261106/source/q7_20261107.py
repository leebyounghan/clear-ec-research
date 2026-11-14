from pathlib import Path
L = Path('/workspace/clear-ec-research/experiments/lane_gpu1_20261106')
# run_e2e.sh is being executed by GPU 2, so it is not edited. New file: the same steps, but regression_input is built once and never rewritten, because another lane's running job checks its hashes.
s = (L/'source/run_e2e.sh').read_text()
a = """step input
CUDA_VISIBLE_DEVICES= PYTHONPATH=$PP $PY $L/source/$W make_input $ARM || fail input $?
"""
b = """step input
if [ -f $SSL/run/complete.json ]; then echo "regression_input exists, not rewritten"; else CUDA_VISIBLE_DEVICES= PYTHONPATH=$PP $PY $L/source/$W make_input $ARM || fail input $?; fi
"""
assert a in s
(L/'source/run_e2e_v2_20261107.sh').write_text(s.replace(a, b))
q = (L/'queue.tsv').read_text().splitlines()
n = 0
for i, r in enumerate(q):
    if r.startswith('e2e_cropopt_both_f4\t'):
        q[i] = f"e2e_cropopt_both_f4\tbash {L}/source/run_e2e_v2_20261107.sh cropopt_both_f4 4 46"; n += 1
assert n == 1
t = L/'queue.tmp'; t.write_text('\n'.join(q)+'\n'); t.replace(L/'queue.tsv')
print('queue line swapped')
