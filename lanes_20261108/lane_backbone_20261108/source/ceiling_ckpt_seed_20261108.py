"""CPU read from stored predictions (lane backbone, 2026-11-08): how much do checkpoint averaging (EMA/SWA proxy)
and seed averaging (distillation proxy) buy recipe A on folds 0 and 4? Member read = mean of per-epoch errors."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
X = Path('/workspace/clear-ec-research/experiments')
RUNS = {0: {46: 'phase2_pooling_replication_20261026/f0_s46_std', 47: 'phase2_pooling_replication_20261026/f0_s47_std', 48: 'oof_recipeA_20261027/fold0/regression_std_s48'},
        4: {46: 'phase2_pooling_20261020/arms/std', 47: 'phase2_pooling_replication_20261026/f4_s47_std', 48: 'oof_recipeA_20261027/fold4/regression_std_s48'}}
b = E.load_base(); gp = E.donors(b['ids']); pos = {int(i): n for n, i in enumerate(b['idx'])}
def load(root, f):
    d = sorted((X / root / 'runs' / f'fold{f}').glob('*'))[0]; out = {}
    for p in sorted(d.glob('predictions_epoch_*.npz')):
        with np.load(p, allow_pickle=True) as z:
            out[int(p.stem[-2:])] = (z['indices'].astype(int), z['pred'].astype(np.float64), z['target'].astype(np.float64))
    return out
res = {}; allE = {}
for f, seeds in RUNS.items():
    P = {s: load(r, f) for s, r in seeds.items()}
    ix, _, y = P[46][20]; keep = np.array([int(i) in pos for i in ix]); y = y[keep]
    for s in P:
        assert all(np.array_equal(P[s][e][0], ix) for e in P[s])
    def mem(s, ep):
        return np.mean([E.ape(P[s][e][1][keep], y) for e in ep], 0)
    def ens(pairs):
        return E.ape(np.exp(np.mean([np.log(P[s][e][1][keep]) for s, e in pairs], 0)), y)
    r = {}
    for s in P:
        if len(P[s]) < 20: r[f's{s}'] = f'only {len(P[s])} epochs'; continue
        r[f's{s}'] = {k: round(E.official(v)['mean'], 4) for k, v in dict(
            member_16_20=mem(s, range(16, 21)), member_8_12=mem(s, range(8, 13)), member_6_10=mem(s, range(6, 11)),
            ens_16_20=ens([(s, e) for e in range(16, 21)]), ens_8_12=ens([(s, e) for e in range(8, 13)]),
            ens_6_20=ens([(s, e) for e in range(6, 21)]), ens_8_20=ens([(s, e) for e in range(8, 21)]), ens_11_20=ens([(s, e) for e in range(11, 21)])).items()}
    full = [s for s in P if len(P[s]) >= 20]
    r['seed_ens_last5'] = round(E.official(ens([(s, e) for s in full for e in range(16, 21)]))['mean'], 4)
    r['seed_ens_8_20'] = round(E.official(ens([(s, e) for s in full for e in range(8, 21)]))['mean'], 4)
    r['seeds_full'] = full
    rows = np.array([pos[int(i)] for i in ix[keep]])
    r['A_tta_scalar_hybrid_col'] = round(E.official(E.ape(b['A'][rows], y))['mean'], 4)
    r['B_hybrid'] = round(E.official(E.ape(b['B'][rows], y))['mean'], 4)
    res[f'fold{f}'] = r
print(json.dumps(res, indent=1))
