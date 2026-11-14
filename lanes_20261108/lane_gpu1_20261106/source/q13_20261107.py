import os, signal, subprocess, time
from pathlib import Path
L = Path('/workspace/clear-ec-research/experiments/lane_gpu1_20261106')
RT = '/workspace/clear-ec-research/experiments/phase2_foundation_20261002/runtime'
job = f"dinov2_b14_f0_s46\tPYTHONPATH={RT} /workspace/venv/bin/python experiments/lane_gpu3_20261106/source/family_recipe_20261106.py all --family dinov2_b14 --fold 0 --seed 46 --root experiments/lane_gpu3_20261106/runs/dinov2_b14_f0_s46"
q = (L/'queue.tsv').read_text().splitlines()
i = [r.split('\t')[0] for r in q].index('cropopt_both_seed1004_f4')
q.insert(i+1, job)
t = L/'queue.tmp'; t.write_text('\n'.join(q)+'\n'); t.replace(L/'queue.tsv')
assert not Path('/workspace/clear-ec-research/experiments/lane_gpu3_20261106/runs/dinov2_b14_f0_s46/runs').exists()
for r in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():
    if 'ssl_views_20261106.py cropopt_both_seed1004_f4' in r and 'bash -c' not in r:
        os.kill(int(r.split()[0]), signal.SIGTERM)
time.sleep(45)
print(open(L/'done.tsv').read()[-110:])
print(subprocess.run(f"nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i 1; nvidia-smi -i 1 --query-compute-apps=pid,used_memory --format=csv,noheader; tail -2 {L}/logs/dinov2_b14_f0_s46.log | cut -c1-200; ps -eo pid,args | grep '[f]amily_recipe.*dinov2' | cut -c1-140; python3 -c \"import json; [print(j['name']) for j in json.load(open('{L}/../lane_gpu3_20261106/jobs.json')) if 'dinov2' in j['name']]\"", shell=True, text=True, capture_output=True).stdout)
