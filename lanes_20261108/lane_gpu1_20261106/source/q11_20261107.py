import os, signal, subprocess, time
from pathlib import Path
L = Path('/workspace/clear-ec-research/experiments/lane_gpu1_20261106')
# The bug: the apostrophe in "lane's" inside ${VAR:?message} opened a quote that swallowed lines 18-33 into one word, so the script jumped to "step train". bash -n passes it. New file without it; v2 is left in place.
s = (L/'source/run_e2e_v2_20261107.sh').read_text()
a = "GPU=${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to this lane's GPU}"
assert a in s
s = s.replace(a, 'GPU=${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}')
new = L/'source/run_e2e_v3_20261107.sh'; new.write_text(s); os.chmod(new, 0o555)
q = (L/'queue.tsv').read_text().splitlines()
names = [r.split('\t')[0] for r in q]
i = names.index('cropopt_both_f1_r')
line = q[i].split('\t',1)[1]
q.insert(i+1, f"e2e_cropopt_both_f4_v3\tbash {new} cropopt_both_f4 4 46")
q.insert(i+2, 'cropopt_both_f1_r2\t' + line)
t = L/'queue.tmp'; t.write_text('\n'.join(q)+'\n'); t.replace(L/'queue.tsv')
for r in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():
    if 'ssl_views_20261106.py cropopt_both_f1' in r and 'bash -c' not in r:
        os.kill(int(r.split()[0]), signal.SIGTERM)
time.sleep(40)
print(open(L/'done.tsv').read()[-170:])
print(subprocess.run(f"nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i 1; nvidia-smi -i 1 --query-compute-apps=pid,used_memory --format=csv,noheader; grep '^###\\|regression_input' {L}/logs/e2e_cropopt_both_f4_v3.log | cut -c1-120; ls {L}/e2e/cropopt_both_f4/f4_s46", shell=True, text=True, capture_output=True).stdout)
