"""Cell-route lane (2026-11-09): nested five-fold reads of the operator-like cell selection features (feats/trim_feats.npz).
Block = (set, region, trim): log cv_k0, log cv_k2, hex_k0, hex_k2, log1p n. Ridge alpha 100 (fixed) on the log residual of the lane best (or of the kept
HEX step), fit on folds != f; rows with < MINC cells in the block keep the base. Pre-registered from the 25-frame gate: CV <- P2 box trim; HEX <- P2 box raw.
usage: corr3.py [ALPHA] [MINC]"""
import sys, json
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import ceiling_meta_cpu_20261103 as C
ROOT = '/workspace/clear-ec-research/experiments/'; LANE = ROOT + 'lane_cellroute_20261109/'
E = BP.E; y = BP.y; fold = BP.fold; idx = BP.idx.astype(int); good = BP.good
ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0; MINC = int(sys.argv[2]) if len(sys.argv) > 2 else 50
z = np.load(LANE + 'feats/trim_feats.npz'); F = z['F'][idx]; NM = list(z['names'])
z = np.load(LANE + 'feats/cellroute_feats.npz'); F1 = z['F'][idx]; N1 = list(z['names'])
BASE = BP.REF_BEST.copy(); BASE[:, 1] = np.load(ROOT + 'lane_cv_20261108/out/stack5_best_3seeds.npz')['P'][:, 1]; BASE[:, 2] = np.load(ROOT + 'lane_hex_20261108/preds/chain_cvw3.npz')['final_hex']
KEPT = BASE.copy(); KEPT[:, 2] = np.load(LANE + 'out/corr_hex_a100.npz')['P'][:, 2]
lg = lambda a: np.log(np.maximum(a, 1e-4))


def block(b):
    if b == 'P2raw':  # the kept step's features (cellroute_feats P2 win+full)
        cols = []
        for rg in ('win', 'full'):
            g = lambda c: F1[:, N1.index(f'P2_{rg}_{c}')]; cols += [lg(g('cv_k0')), lg(g('cv_k2')), g('hex_k0'), g('hex_k2'), np.log1p(np.nan_to_num(g('n')))]
        return np.column_stack(cols), np.nan_to_num(F1[:, N1.index('P2_full_n')])
    g = lambda c: F[:, NM.index(f'{b}_{c}')]
    return np.column_stack([lg(g('cv_k0')), lg(g('cv_k2')), g('hex_k0'), g('hex_k2'), np.log1p(np.nan_to_num(g('n')))]), np.nan_to_num(g('n'))


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
    for s in ('P2', 'P0'):
        for rg in ('box', 'p150', 'win', 'full'):
            for t in ('raw', 'trim'):
                b = f'{s}_{rg}_{t}'; X, n = block(b); m = np.isfinite(X).all(1) & (n >= MINC) & (y[:, 1] > 0) & (y[:, 2] > 0)
                print(b.ljust(14), 'cv_k2 r %.3f / resid %+.3f | cv_k0 %.3f / %+.3f | hex_k2 %.3f / %+.3f | hex_k0 %.3f / %+.3f (rows %d)' % (
                    np.corrcoef(X[m, 1], ly[m, 1])[0, 1], np.corrcoef(X[m, 1], ly[m, 1] - np.log(BASE[m, 1]))[0, 1], np.corrcoef(X[m, 0], ly[m, 1])[0, 1], np.corrcoef(X[m, 0], ly[m, 1] - np.log(BASE[m, 1]))[0, 1],
                    np.corrcoef(X[m, 3], ly[m, 2])[0, 1], np.corrcoef(X[m, 3], ly[m, 2] - np.log(BASE[m, 2]))[0, 1], np.corrcoef(X[m, 2], ly[m, 2])[0, 1], np.corrcoef(X[m, 2], ly[m, 2] - np.log(BASE[m, 2]))[0, 1], int(m.sum())), flush=True)
    tries = [('CV  pre-registered P2 box trim', 1, ['P2_box_trim'], BASE), ('HEX pre-registered P2 box raw', 2, ['P2_box_raw'], BASE), ('HEX pre-registered P2 box raw on the kept step', 2, ['P2_box_raw'], KEPT),
             ('CV  info P2 p150 trim', 1, ['P2_p150_trim'], BASE), ('CV  info P2 win trim', 1, ['P2_win_trim'], BASE), ('CV  info P2 full trim', 1, ['P2_full_trim'], BASE), ('CV  info P0 box trim', 1, ['P0_box_trim'], BASE),
             ('CV  info P2 box trim + P0 box trim', 1, ['P2_box_trim', 'P0_box_trim'], BASE),
             ('HEX info P2 p150 raw', 2, ['P2_p150_raw'], BASE), ('HEX info P2 box trim', 2, ['P2_box_trim'], BASE), ('HEX info P2raw + P2 box raw', 2, ['P2raw', 'P2_box_raw'], BASE)]
    for tag, j, bl, base in tries:
        P, o = correct(j, bl, base); out[tag] = o; print(tag.ljust(48), json.dumps(o), flush=True)
        if 'pre-registered' in tag: np.savez_compressed(LANE + 'out/corr3_' + tag.replace(' ', '_').replace('-', '') + '.npz', P=P, note=tag)
    json.dump(out, open(LANE + f'out/corr3_a{int(ALPHA)}_m{MINC}.json', 'w'), indent=1)
