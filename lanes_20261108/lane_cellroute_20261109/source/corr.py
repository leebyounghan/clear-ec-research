"""Cell-route lane (2026-11-09): nested five-fold correction of the CV and HEX lane bests with cell-route features (feats/cellroute_feats.npz).
Ridge (alpha fixed, standardized on training rows) on the log residual of the lane best, fit on folds != f and applied to fold f; rows with
< MINC interior cells in the set's 'full' region keep the lane best. MAPE-optimal scale refit on the training rows (as konan_corr_20261109.py).
One recipe on all folds. Pre-registered (from the 25-frame gate): CV <- P1 (seeds3 t0.2 cleaned), HEX <- P2 (3-detector + mask union).
Lane bests: CV lane_cv_20261108/out/stack5_best_3seeds.npz (10.114), HEX lane_hex_20261108/preds/chain_cvw3.npz (9.738).
usage: corr.py [FEATFILE] [ALPHA] [MINC]"""
import sys, json
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import ceiling_meta_cpu_20261103 as C
ROOT = '/workspace/clear-ec-research/experiments/'
E = BP.E; y = BP.y; fold = BP.fold; idx = BP.idx.astype(int); good = BP.good
FF = sys.argv[1] if len(sys.argv) > 1 else ROOT + 'lane_cellroute_20261109/feats/cellroute_feats.npz'
ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0; MINC = int(sys.argv[3]) if len(sys.argv) > 3 else 50
z = np.load(FF); F = z['F'][idx]; NM = list(z['names'])
BASE = BP.REF_BEST.copy(); BASE[:, 1] = np.load(ROOT + 'lane_cv_20261108/out/stack5_best_3seeds.npz')['P'][:, 1]; BASE[:, 2] = np.load(ROOT + 'lane_hex_20261108/preds/chain_cvw3.npz')['final_hex']
lg = lambda a: np.log(np.maximum(a, 1e-4))


def design(sets, regions):
    cols, nref = [], None
    for s in sets:
        for rg in regions:
            if f'{s}_{rg}_cv_k2' not in NM: continue
            for c in ('cv_k0', 'cv_k2'): cols.append(lg(F[:, NM.index(f'{s}_{rg}_{c}')]))
            for c in ('hex_k0', 'hex_k2'): cols.append(F[:, NM.index(f'{s}_{rg}_{c}')])
            cols.append(np.log1p(np.nan_to_num(F[:, NM.index(f'{s}_{rg}_n')])))
        if nref is None: nref = np.nan_to_num(F[:, NM.index(f'{s}_full_n')])
    return np.column_stack(cols), nref


def correct(j, sets, regions, alpha=ALPHA, minc=MINC):
    X, n = design(sets, regions); X = np.c_[X, np.log(BASE[:, j])]; m = np.isfinite(X).all(1) & (n >= minc); ok = y[:, j] > 0
    res = np.log(np.where(ok, y[:, j], 1.0)) - np.log(BASE[:, j]); X = np.where(m[:, None], X, 0.0); P = BASE.copy()
    for f in range(5):
        tr = (fold != f) & good & m & ok; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Z = (X - mu) / sd; A = Z[tr]
        w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (res[tr] - res[tr].mean())); q = BASE[:, j] * np.exp(Z @ w)
        c = C.wmed(y[tr, j] / q[tr], q[tr] / y[tr, j]); c0 = C.wmed(y[tr, j] / BASE[tr, j], BASE[tr, j] / y[tr, j]); sel = (fold == f) & m; P[sel, j] = q[sel] * c / c0
    a, a0, ar = E.ape(P, y)[:, j], E.ape(BASE, y)[:, j], E.ape(BP.REF_BEST, y)[:, j]; f04 = (fold == 0) | (fold == 4)
    return P, dict(value=round(float(np.nanmean(a)), 4), lane_best=round(float(np.nanmean(a0)), 4), d_lane=round(float(np.nanmean(a) - np.nanmean(a0)), 4), d_best_pipeline=round(float(np.nanmean(a) - np.nanmean(ar)), 4),
                   folds_vs_lane=[round(float(np.nanmean(a[fold == f]) - np.nanmean(a0[fold == f])), 4) for f in range(5)], f04_vs_lane=round(float(np.nanmean(a[f04]) - np.nanmean(a0[f04])), 4), rows=int(m.sum()))


if __name__ == '__main__':
    out = {}
    # raw correlations (rows with >= MINC cells)
    ly = np.log(np.maximum(y, 1e-4))
    for s in ('P0', 'P1', 'P2', 'P3'):
        for rg in ('win', 'full', 'creg'):
            if f'{s}_{rg}_cv_k2' not in NM: continue
            n = np.nan_to_num(F[:, NM.index(f'{s}_full_n')]); line = []
            for c, j in (('cv_k0', 1), ('cv_k2', 1), ('hex_k0', 2), ('hex_k2', 2)):
                x = F[:, NM.index(f'{s}_{rg}_{c}')]; m = np.isfinite(x) & (x > 0) & (n >= MINC) & (y[:, j] > 0); xx = lg(x[m]) if c.startswith('cv') else x[m]
                rl = np.corrcoef(xx, ly[m, j])[0, 1]; rr = np.corrcoef(xx, ly[m, j] - np.log(BASE[m, j]))[0, 1]; line.append(f'{c} r {rl:.3f} / resid {rr:+.3f}')
            print(f'{s} {rg}'.ljust(9), ' | '.join(line), f'(rows {int(m.sum())})', flush=True)
    for tag, j, sets, regs in (('CV  pre-registered P1 win+full', 1, ['P1'], ['win', 'full']), ('HEX pre-registered P2 win+full', 2, ['P2'], ['win', 'full']),
                               ('CV  info P0', 1, ['P0'], ['win', 'full']), ('CV  info P2', 1, ['P2'], ['win', 'full']), ('CV  info P3', 1, ['P3'], ['win', 'full']), ('CV  info all sets', 1, ['P0', 'P1', 'P2', 'P3'], ['win', 'full']),
                               ('HEX info P0', 2, ['P0'], ['win', 'full']), ('HEX info P1', 2, ['P1'], ['win', 'full']), ('HEX info P3', 2, ['P3'], ['win', 'full']), ('HEX info P2 +creg', 2, ['P2'], ['win', 'full', 'creg']), ('HEX info all sets', 2, ['P0', 'P1', 'P2', 'P3'], ['win', 'full'])):
        P, o = correct(j, sets, regs); out[tag] = o; print(tag.ljust(32), json.dumps(o), flush=True)
        if 'pre-registered' in tag: np.savez_compressed(ROOT + f'lane_cellroute_20261109/out/corr_{tag.split()[0].lower()}_a{int(ALPHA)}.npz', P=P, note=tag)
    json.dump(out, open(ROOT + f'lane_cellroute_20261109/out/corr_a{int(ALPHA)}_m{MINC}.json', 'w'), indent=1)
