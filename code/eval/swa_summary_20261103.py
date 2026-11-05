"""Five-fold summary of the train-fit and SWA inference (2026-11-04): member level for seed 46 (epoch 20 vs SWA of epochs 11-20,
one view and four flips) and hybrid level with the seed-46 member replaced by SWA (seed 47 unchanged, so this understates a
full swap), LOFO recipe A scalars and Phase-1 five unchanged. Paired donor-cluster bootstrap against hybrid B."""
import json, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
E = C.E; K = C.K
b = E.load_base(); y, fold = b['y'], b['fold']; pos = {int(i): k for k, i in enumerate(b['idx'])}
T = C.X / 'meta_ceiling_20261103/trainfit'
seeds, single, hist = C.load_members(b)
sc = np.array([json.load(open(C.X / 'recipeA_analysis_20261028/hybrid_oof.json'))['recipeA_lofo_scalars'][str(f)] for f in fold])
swa1, swa4 = np.full_like(b['B'], np.nan), np.full_like(b['B'], np.nan)
rows = []
for f in range(5):
    z = np.load(T / f'fold{f}_swa11_20.npz')
    r = np.array([pos[int(i)] for i in z['val_indices']])
    swa1[r], swa4[r] = z['val_pred_single'], z['val_pred_tta4']
    R = json.load(open(T / f'fold{f}_report.json')); m = json.load(open(C.X / str(R['root']).split('experiments/')[-1] / 'manifest.json'))['recipe']
    fm = fold == f
    rows.append(dict(fold=f, runner_head_lr=m['head_lr'], schedule_epochs=m['schedule_epochs'], train_e20={k: round(R['e20']['train'][k], 2) for k in K}, val_e20={k: round(R['e20']['val'][k], 2) for k in K},
                     e20_single=round(C.mape(single[46][fm], y[fm])['mean'], 3), swa_single=round(C.mape(swa1[fm], y[fm])['mean'], 3),
                     e20_tta=round(C.mape(seeds[46][fm], y[fm])['mean'], 3), swa_tta=round(C.mape(swa4[fm], y[fm])['mean'], 3)))
    print(rows[-1], flush=True)
g = E.donors(b['ids']); out = dict(per_fold=rows)
def rep(name, p, ref):
    r = dict(score={k: round(v, 4) for k, v in C.mape(p, y).items() if k != 'n'}, paired=E.paired_bootstrap(E.ape(p, y), E.ape(ref, y), g),
             per_fold={f: round(C.mape(p[fold == f], y[fold == f])['mean'] - C.mape(ref[fold == f], y[fold == f])['mean'], 4) for f in range(5)})
    out[name] = r; print(name, r['score'], round(r['paired']['delta'], 4), [round(x, 4) for x in r['paired']['ci95']], r['per_fold'], flush=True)
rep('member_swa_single_vs_e20_single', swa1, single[46])
rep('member_swa_tta_vs_e20_tta', swa4, seeds[46])
A_new = 0.5 * (swa4 + seeds[47]) * sc
rep('A_with_s46_swa_vs_A', A_new, b['A'])
rep('B_with_s46_swa_vs_B', 0.5 * A_new + 0.5 * b['P1'], b['B'])
json.dump(out, open(T / 'swa_summary.json', 'w'), indent=1, default=float)
