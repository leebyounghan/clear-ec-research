"""Is the control's epoch-1 gap a runner difference or path noise? Step-level gradient norms against the two-rank log, and epoch-1 spread of two-rank runs."""
import json, re, glob
import numpy as np
def steps(path):
    out = {}
    for line in open(path, errors='ignore'):
        if line.startswith('{"status": "training"') and '"epoch": 1,' in line:
            try:
                d = json.loads(line)
            except Exception:
                continue
            out[d['updates']] = (d['gradient_norm'], d['encoder_lr'], d['head_lr'], d['seen'])
    return out
c = steps('experiments/lane_gpu0_20261106/logs/control_single_gpu_f0_s46.log')
candidates = glob.glob('experiments/phase2_pooling_replication_20261026/f0_s46_std/full.log') + glob.glob('experiments/phase2_pooling_replication_20261026/*f0*s46*.log')
print('reference logs', candidates)
r = steps(candidates[0]) if candidates else {}
print('update control two-rank ratio')
for u in sorted(set(c) & set(r)):
    print(u, round(c[u][0], 4), round(r[u][0], 4), round(c[u][0] / r[u][0], 4), 'lr equal', c[u][1:3] == r[u][1:3], 'seen', c[u][3], r[u][3])
print('\nepoch-1 and epoch-20 validation mean of two-rank runs: spread across seeds of the same fold')
X = 'experiments/'
RUNS = {'f0 s46': 'phase2_pooling_replication_20261026/f0_s46_std', 'f0 s47': 'phase2_pooling_replication_20261026/f0_s47_std', 'f0 s48': 'oof_recipeA_20261027/fold0/regression_std_s48', 'f4 s46': 'phase2_pooling_20261020/arms/std', 'f4 s47': 'phase2_pooling_replication_20261026/f4_s47_std', 'f4 s48': 'oof_recipeA_20261027/fold4/regression_std_s48'}
for k, v in RUNS.items():
    h = json.load(open(glob.glob(X + v + '/runs/*/*/history.json')[0]))
    print(k, 'first three', [round(e['score']['mean'], 3) for e in h[:3]], 'loss1', round(h[0]['online_train_loss'], 4), 'gradnorm1', round(h[0]['gradient_norm_mean'], 3), 'last five mean', round(np.mean([e['score']['mean'] for e in h[-5:]]), 4))
