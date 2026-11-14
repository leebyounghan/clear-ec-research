"""Cell-route lane (2026-11-09): nested five-fold HEX (and CV, info) correction with the click-emulation features.
Blocks: P2raw = cellroute_feats P2 win+full (k0/k2 CV, HEX, count); P2c / P5c = click_feats (mean over 8 draws of click noise 0.16 s, windows):
k0/k2 CV, HEX, count, 5- and 7-sided shares. Ridge alpha 100 (fixed, as corr.py) on the log residual of the lane best, fit on folds != f.
Rows need finite features and >= MINC cells in every block, else they keep the lane best. Pre-registered HEX try: P2raw + P2c + P5c.
usage: corr2.py CLICKFILE [ALPHA] [MINC]"""
import sys, json
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import ceiling_meta_cpu_20261103 as C
ROOT = '/workspace/clear-ec-research/experiments/'; LANE = ROOT + 'lane_cellroute_20261109/'
E = BP.E; y = BP.y; fold = BP.fold; idx = BP.idx.astype(int); good = BP.good
ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0; MINC = int(sys.argv[3]) if len(sys.argv) > 3 else 50
z = np.load(LANE + 'feats/cellroute_feats.npz'); F1 = z['F'][idx]; N1 = list(z['names'])
z = np.load(sys.argv[1]); F2 = z['F'][idx]; N2 = list(z['names'])
BASE = BP.REF_BEST.copy(); BASE[:, 1] = np.load(ROOT + 'lane_cv_20261108/out/stack5_best_3seeds.npz')['P'][:, 1]; BASE[:, 2] = np.load(ROOT + 'lane_hex_20261108/preds/chain_cvw3.npz')['final_hex']
KEPT = BASE.copy(); KEPT[:, 2] = np.load(LANE + 'out/corr_hex_a100.npz')['P'][:, 2]
lg = lambda a: np.log(np.maximum(a, 1e-4))


def block(name):
    cols = []
    if name == 'P2raw':
        for rg in ('win', 'full'):
            g = lambda c: F1[:, N1.index(f'P2_{rg}_{c}')]
            cols += [lg(g('cv_k0')), lg(g('cv_k2')), g('hex_k0'), g('hex_k2'), np.log1p(np.nan_to_num(g('n')))]
        return np.column_stack(cols), np.nan_to_num(F1[:, N1.index('P2_full_n')])
    g = lambda c: F2[:, N2.index(f'{name}_win_{c}')]
    X = np.column_stack([lg(g('cv_k0')), lg(g('cv_k2')), g('hex_k0'), g('hex_k2'), np.log1p(np.nan_to_num(g('n'))), g('s5_k2'), g('s7_k2')])
    return X, np.nan_to_num(g('n'))


def correct(j, blocks, base, alpha=ALPHA, minc=MINC):
    Xs, m = [], np.ones(len(y), bool)
    for b in blocks:
        X, n = block(b); Xs.append(X); m &= np.isfinite(X).all(1) & (n >= minc)
    X = np.column_stack(Xs + [np.log(base[:, j])]); ok = y[:, j] > 0; res = np.log(np.where(ok, y[:, j], 1.0)) - np.log(base[:, j]); X = np.where(m[:, None], X, 0.0); P = base.copy()
    for f in range(5):
        tr = (fold != f) & good & m & ok; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Z = (X - mu) / sd; A = Z[tr]
        w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (res[tr] - res[tr].mean())); q = base[:, j] * np.exp(Z @ w)
        c = C.wmed(y[tr, j] / q[tr], q[tr] / y[tr, j]); c0 = C.wmed(y[tr, j] / base[tr, j], base[tr, j] / y[tr, j]); sel = (fold == f) & m; P[sel, j] = q[sel] * c / c0
    a, a0, al = E.ape(P, y)[:, j], E.ape(base, y)[:, j], E.ape(BASE, y)[:, j]; f04 = (fold == 0) | (fold == 4)
    return P, dict(value=round(float(np.nanmean(a)), 4), d_lane_best=round(float(np.nanmean(a) - np.nanmean(al)), 4), d_base=round(float(np.nanmean(a) - np.nanmean(a0)), 4),
                   folds_vs_lane=[round(float(np.nanmean(a[fold == f]) - np.nanmean(al[fold == f])), 4) for f in range(5)], f04_vs_lane=round(float(np.nanmean(a[f04]) - np.nanmean(al[f04])), 4), rows=int(m.sum()))


if __name__ == '__main__':
    ly = np.log(np.maximum(y, 1e-4)); out = {}
    for nm in ('P2c', 'P5c'):
        X, n = block(nm); m = np.isfinite(X).all(1) & (n >= MINC) & (y[:, 2] > 0)
        print(nm, 'hex_k2 r with log HEX %.3f, with chain residual %+.3f | hex_k0 %.3f / %+.3f | cv_k2 r with log CV %.3f, with CV stack residual %+.3f (rows %d)' % (
            np.corrcoef(X[m, 3], ly[m, 2])[0, 1], np.corrcoef(X[m, 3], ly[m, 2] - np.log(BASE[m, 2]))[0, 1], np.corrcoef(X[m, 2], ly[m, 2])[0, 1], np.corrcoef(X[m, 2], ly[m, 2] - np.log(BASE[m, 2]))[0, 1],
            np.corrcoef(X[m, 1], ly[m, 1])[0, 1], np.corrcoef(X[m, 1], ly[m, 1] - np.log(BASE[m, 1]))[0, 1], int(m.sum())), flush=True)
    tries = [('HEX pre-registered P2raw+P2c+P5c on lane best', 2, ['P2raw', 'P2c', 'P5c'], BASE), ('HEX info P2c+P5c on the kept P2raw step', 2, ['P2c', 'P5c'], KEPT),
             ('HEX info P2c', 2, ['P2c'], BASE), ('HEX info P5c', 2, ['P5c'], BASE), ('HEX info P2raw+P2c', 2, ['P2raw', 'P2c'], BASE), ('HEX info P2raw+P5c', 2, ['P2raw', 'P5c'], BASE),
             ('CV  info P2raw+P2c+P5c', 1, ['P2raw', 'P2c', 'P5c'], BASE), ('CV  info P5c', 1, ['P5c'], BASE)]
    for tag, j, bl, base in tries:
        P, o = correct(j, bl, base); out[tag] = o; print(tag.ljust(44), json.dumps(o), flush=True)
        if 'pre-registered' in tag: np.savez_compressed(LANE + f'out/corr2_hex_a{int(ALPHA)}.npz', P=P, note=tag)
    json.dump(out, open(LANE + f'out/corr2_a{int(ALPHA)}_m{MINC}.json', 'w'), indent=1)
