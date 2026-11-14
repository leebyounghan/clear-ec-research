"""How much does best-epoch selection on the validation fold flatter each metric? (HEX lane, 2026-11-08)
Phase-1 members (half of hybrid B) were saved at the epoch with the best validation mean; submission 1 lost 0.40 on test HEX
but gained 0.06 on CD and CV. Test with recipe A per-epoch validation predictions: split each run's validation rows in two by
donor, pick the epoch with the best mean on one half, score both halves at that epoch. Optimism per metric =
E[score on the other half - score on the picking half]. Also the same for picking by each single metric."""
import glob, sys, json
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
E = C.E; X = C.X; b = E.load_base(); don = dict(zip(b['idx'].tolist(), E.donors(b['ids']).tolist()))
res = []
for s, d in C.RECIPE.items():
    for f, p in d.items():
        fs = sorted(glob.glob(str(X / p / 'runs' / f'fold{f}' / '*' / 'predictions_epoch_*.npz')))
        if len(fs) < 10: print('skip', s, f, len(fs)); continue
        Z = [np.load(q, allow_pickle=True) for q in fs]; ix = Z[0]['indices']; t = Z[0]['target']; P = np.stack([z['pred'] for z in Z])  # E x n x 3
        ok = t > 0; A = np.where(ok[None], np.abs(P - t[None]) / np.where(ok, t, 1)[None] * 100, np.nan)   # E x n x 3
        g = np.array([don.get(int(i), -1) for i in ix]); u = np.unique(g); rng = np.random.default_rng(f * 100 + s)
        opt = []; opt_single = []
        for _ in range(300):
            h = np.isin(g, rng.choice(u, len(u) // 2, replace=False)); Sm = np.nanmean(A[:, h], 1); Vm = np.nanmean(A[:, ~h], 1)   # E x 3
            for (S_, V_) in ((Sm, Vm), (Vm, Sm)):
                e = int(np.argmin(S_.mean(1))); opt.append(V_[e] - S_[e])
                opt_single.append([V_[int(np.argmin(S_[:, j])), j] - S_[int(np.argmin(S_[:, j])), j] for j in range(3)])
        opt = np.mean(opt, 0); opt_single = np.mean(opt_single, 0)
        full = np.nanmean(A, 1); eb = int(np.argmin(full.mean(1)))
        r = dict(seed=s, fold=f, epochs=len(fs), best_epoch=eb + 1, best=full[eb].round(3).tolist(), last5=full[-5:].mean(0).round(3).tolist(), optimism_half=opt.round(3).tolist(), optimism_pick_own_metric=opt_single.round(3).tolist(), sd_last10=full[-10:].std(0).round(3).tolist())
        print(json.dumps(r), flush=True); res.append(r)
o = np.array([r['optimism_half'] for r in res]); print('mean optimism, pick by mean, half-size picking set (CD, CV, HEX):', o.mean(0).round(3), 'SD over runs', o.std(0).round(3))
d = np.array([np.array(r['last5']) - np.array(r['best']) for r in res]); print('last-5 minus picked epoch on all rows (CD, CV, HEX):', d.mean(0).round(3))
