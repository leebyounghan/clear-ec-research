"""Pre-registered comparison for masked_pool_regression_20261031 (see protocol.json): masked vs standard std pooling,
same fold / seed / SSL encoder, terminal-epoch single-view validation predictions; member level and hybrid level
(0.5 * member x the fold's LOFO recipe-A scalars + 0.5 * Phase-1 five), paired donor-cluster bootstrap."""
import json, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E

X = '/workspace/clear-ec-research/experiments/'
REF = {4: 'phase2_pooling_20261020/arms/std', 0: 'phase2_pooling_replication_20261026/f0_s46_std'}
f = int(sys.argv[1])
masked_root = X + f'masked_pool_regression_20261031/f{f}_s46_stdmasked'
b = E.load_base(); y = b['y']; pos = {int(i): n for n, i in enumerate(b['idx'])}
rows = np.where(b['fold'] == f)[0]
rp = {int(b['idx'][r]): n for n, r in enumerate(rows)}

def load(root):
    with np.load(root + f'/runs/fold{f}/cpdino_vitb/predictions.npz') as z:
        p = np.full((len(rows), 3), np.nan)
        for i, v in zip(z['indices'].astype(int), z['pred'].astype(np.float64)):
            p[rp[int(i)]] = v
    assert np.isfinite(p).all()
    return p

std, msk = load(X + REF[f]), load(masked_root)
yy = y[rows]; frame = E.image_stats(b)['black_frac'][rows] > 0.01
g = np.unique(E.donors(b['ids'])[rows], return_inverse=True)[1]
sc = np.array(json.load(open(X + 'recipeA_analysis_20261028/hybrid_oof.json'))['recipeA_lofo_scalars'][str(f)])
P1 = b['P1'][rows]
eS, eM = E.ape(std, yy), E.ape(msk, yy)
hS, hM = E.ape(0.5 * std * sc + 0.5 * P1, yy), E.ape(0.5 * msk * sc + 0.5 * P1, yy)
def pb(e1, e0, m):
    return E.paired_bootstrap(e1[m], e0[m], np.unique(g[m], return_inverse=True)[1])
allm = np.ones(len(rows), bool)
res = dict(fold=f, n=int(len(rows)), n_frame=int(frame.sum()),
           member=dict(standard={k: round(E.official(eS, m)['mean'], 4) for k, m in (('all', allm), ('frame', frame), ('no_frame', ~frame))},
                       masked={k: round(E.official(eM, m)['mean'], 4) for k, m in (('all', allm), ('frame', frame), ('no_frame', ~frame))},
                       delta_all=pb(eM, eS, allm), delta_frame=pb(eM, eS, frame), delta_no_frame=pb(eM, eS, ~frame)),
           hybrid=dict(delta_all=pb(hM, hS, allm), delta_frame=pb(hM, hS, frame)))
d = res['member']
res['continue_rule_pass'] = bool(d['delta_frame']['delta'] <= -0.15 and d['delta_no_frame']['delta'] <= 0.05)
res['rollout_rule_this_fold'] = bool(res['continue_rule_pass'] and d['delta_all']['delta'] <= -0.05 and res['hybrid']['delta_all']['delta'] <= -0.03)
out = X + f'masked_pool_regression_20261031/compare_fold{f}.json'
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps(res, indent=1))
