import os, signal, subprocess, time
from pathlib import Path
L = Path('/workspace/clear-ec-research/experiments/lane_gpu1_20261106'); PY = '/workspace/venv/bin/python'; R = 'experiments/lane_gpu1_20261106'
(L/'tokens').mkdir(exist_ok=True)
G = 'experiments/meta_ceiling_20261103/sweep/cell_graphs_pre48_xy.npz'
S = 'experiments/meta_ceiling_20261103/source/cell_graph_sweep3_20261105.py'
base = '--d 64 --layers 5 --seeds 3 --no-ctx --graphs cell_graphs_pre48_xy.npz'
new = [
 f"tok_crop_pre48\t{PY} -u {R}/source/node_tokens_20261107.py {R}/ssl/cropopt_both_f4/run/last.pt {G} {R}/tokens/crop_pre48.npz",
 f"tok_ssl5_pre48\t{PY} -u {R}/source/node_tokens_20261107.py experiments/fulldata_recipeA_20261028/ssl5_s1003/run/last.pt {G} {R}/tokens/ssl5_pre48.npz",
 f"g1_tk_crop8\t{PY} -u {S} g1_tk_crop8 {base} --tokens {L}/tokens/crop_pre48.npz --token-dims 8",
 f"g1_tk_ssl8\t{PY} -u {S} g1_tk_ssl8 {base} --tokens {L}/tokens/ssl5_pre48.npz --token-dims 8",
 f"g1_tk_crop32\t{PY} -u {S} g1_tk_crop32 {base} --tokens {L}/tokens/crop_pre48.npz --token-dims 32",
 f"g1_tk_crop8_reg\t{PY} -u {S} g1_tk_crop8_reg {base} --tokens {L}/tokens/crop_pre48.npz --token-dims 8 --drop 0.3 --wd 0.1",
 f"read_node_tokens\t{PY} -u {R}/source/node_token_read_20261107.py g1_tk_crop8 g1_tk_ssl8 g1_tk_crop32 g1_tk_crop8_reg",
]
q = (L/'queue.tsv').read_text().splitlines()
done = {r.split('\t')[0] for r in (L/'done.tsv').read_text().splitlines()}
names = [r.split('\t')[0] for r in q]
head = [r for r in q if r.split('\t')[0] in done or r.split('\t')[0] == 'cropopt_both_f1_r2']
pend = {r.split('\t')[0]: r for r in q if r not in head}
tail = [pend[n] for n in ('cropopt_both_seed1004_f4', 'cropxfer_both_f4', 'cropopt_mean_f0', 'cropoptlr3_both_f4', 'cropopt20_both_f4', 'cropopt_both_f3', 'cropopt_both_f2', 'cropopt_both_all')]
t = L/'queue.tmp'; t.write_text('\n'.join(head + new + tail) + '\n'); t.replace(L/'queue.tsv')
for r in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():
    if 'ssl_views_20261106.py cropopt_both_f1' in r and 'bash -c' not in r:
        os.kill(int(r.split()[0]), signal.SIGTERM)
time.sleep(30)
print(open(L/'done.tsv').read()[-120:])
print(subprocess.run(f"nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i 1; tail -2 {L}/logs/tok_crop_pre48.log | cut -c1-160", shell=True, text=True, capture_output=True).stdout)
