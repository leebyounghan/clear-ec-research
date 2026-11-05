"""Bridge-capacity read-out (2026-11-04): fold 0, seed 46, against the matched recipe A member. Validation per checkpoint,
train-probe error from the runner's history, paired donor-cluster bootstrap on the terminal epoch (one view), and the hybrid
with the seed-46 member swapped, one view (seed 47 and Phase-1 unchanged). Also the snapshot mean of epochs 11-20 per arm."""
import json, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
E = C.E; K = C.K
b = E.load_base(); y, fold = b['y'], b['fold']; pos = {int(i): k for k, i in enumerate(b['idx'])}
rows = np.where(fold == 0)[0]; yy = y[rows]; g = np.unique(E.donors(b['ids'])[rows], return_inverse=True)[1]
sc = np.array(json.load(open(C.X / 'recipeA_analysis_20261028/hybrid_oof.json'))['recipeA_lofo_scalars']['0'])
ROOT = dict(recipeA=C.X / 'phase2_pooling_replication_20261026/f0_s46_std', enc_lr5x=C.X / 'meta_ceiling_20261103/bridge2/f0_enc_lr5x', token_mlp=C.X / 'meta_ceiling_20261103/bridge2/f0_token_mlp')
seeds, single, _ = C.load_members(b)
def pred(name, e):
    with np.load(ROOT[name] / f'runs/fold0/cpdino_vitb/predictions_epoch_{e:02d}.npz') as z:
        p = np.full((len(b['y']), 3), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred']
    return p[rows]
out = {}
hist = {k: json.load(open(ROOT[k] / 'runs/fold0/cpdino_vitb/history.json')) for k in ROOT}
print('history keys', list(hist['recipeA'][-1])[:30])
for name in ROOT:
    out[name] = dict(val={e: {k: round(v, 3) for k, v in C.mape(pred(name, e), yy).items() if k != 'n'} for e in (1, 5, 10, 15, 20)})
    snap = np.exp(np.mean([np.log(pred(name, e)) for e in range(11, 21)], 0))
    out[name]['snapshot_11_20'] = {k: round(v, 3) for k, v in C.mape(snap, yy).items() if k != 'n'}
    print(name, 'e20', out[name]['val'][20], 'snap', out[name]['snapshot_11_20'], flush=True)
    H = hist[name]
    tr = [(h.get('epoch'), {k_: h[k_] for k_ in h if 'train' in k_ and not isinstance(h[k_], (list, dict))}) for h in H if h.get('epoch') in (1, 5, 10, 15, 20)]
    out[name]['train_probe'] = tr; print('  train probe', json.dumps(tr[-1])[:500], flush=True)
base = pred('recipeA', 20)
for name in ('enc_lr5x', 'token_mlp'):
    p = pred(name, 20)
    out[name]['paired_member'] = E.paired_bootstrap(E.ape(p, yy), E.ape(base, yy), g)
    s47 = single[47][rows]; P1 = b['P1'][rows]
    h0 = 0.5 * (0.5 * (base + s47) * sc) + 0.5 * P1; h1 = 0.5 * (0.5 * (p + s47) * sc) + 0.5 * P1
    out[name]['paired_hybrid_swap_s46'] = E.paired_bootstrap(E.ape(h1, yy), E.ape(h0, yy), g)
    out[name]['err_corr_with_recipeA'] = {k: round(float(np.corrcoef(np.log(p[:, j] / yy[:, j])[yy[:, j] > 0], np.log(base[:, j] / yy[:, j])[yy[:, j] > 0])[0, 1]), 4) for j, k in enumerate(K)}
    h2 = 0.5 * (0.5 * (0.5 * (p + base) + s47) * sc) + 0.5 * P1
    out[name]['paired_hybrid_add_as_member'] = E.paired_bootstrap(E.ape(h2, yy), E.ape(h0, yy), g)
    print(name, json.dumps({k: out[name][k] for k in ('paired_member', 'paired_hybrid_swap_s46', 'paired_hybrid_add_as_member', 'err_corr_with_recipeA')}), flush=True)
json.dump(out, open(C.X / 'meta_ceiling_20261103/bridge2/compare_fold0.json', 'w'), indent=1, default=float)
