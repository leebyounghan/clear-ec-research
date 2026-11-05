"""Clean nested check of 'hex6 then affine-in-log stretch' (2026-11-03). Per outer fold f: hex6 with the parameters fit
on folds != f (a, b by OLS of log HEX on f_active10, w by MAPE grid; = the exported LOFO params of fold f) applied to
every row; stretch (slope g on a grid, MAPE-optimal scalar c) fit on the rows of folds != f; both applied to fold f.
Mechanism probe: the same stretch on recipe A alone, on Phase-1 alone and on B (averaging two partly independent
predictors compresses the spread, so the hybrid should need it more than either member)."""
import json, os, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
E = C.E; K = C.K; r4 = C.r4
GRID = np.round(np.arange(0.85, 1.36, 0.025), 3)

def fit_stretch(p, y, rows):
    out = []
    for j in range(3):
        ok = rows & (y[:, j] > 0); lb = np.log(p[ok, j]); mu = lb.mean(); best = None
        for g in GRID:
            q = np.exp(g * (lb - mu) + mu); c = C.wmed(y[ok, j] / q, q / y[ok, j]); m = np.mean(np.abs(c * q - y[ok, j]) / y[ok, j])
            if best is None or m < best[0]: best = (m, g, c, mu)
        out.append(best[1:])
    return out

def apply_stretch(p, prm):
    q = p.copy()
    for j, (g, c, mu) in enumerate(prm):
        q[:, j] = c * np.exp(g * (np.log(p[:, j]) - mu) + mu)
    return q

def main():
    b = E.load_base(); y, B, fold = b['y'], b['B'], b['fold']
    gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True)
    prm = json.loads(str(gf['lofo_params_plain'])); fa = gf['f_active10']; have = gf['f_active10_valid'] & ~gf['seen_leaked'] & np.isfinite(fa)
    grp = E.donors(b['ids']); out = {}
    def rep(name, p, ref):
        r = dict(score={k: r4(v) for k, v in C.mape(p, y).items() if k != 'n'}, paired=E.paired_bootstrap(E.ape(p, y), E.ape(ref, y), grp),
                 per_fold={f: r4(C.mape(p[fold == f], y[fold == f])['mean'] - C.mape(ref[fold == f], y[fold == f])['mean']) for f in range(5)})
        out[name] = r; print(name, r['score'], round(r['paired']['delta'], 4), [round(x, 4) for x in r['paired']['ci95']], r['per_fold'], flush=True)
    for name, base in (('A', b['A']), ('P1', b['P1']), ('B', B)):
        p = np.full_like(base, np.nan); slopes = {}
        for f in range(5):
            s = fit_stretch(base, y, fold != f); slopes[f] = [r4(x[0]) for x in s]
            p[fold == f] = apply_stretch(base[fold == f], s)
        rep(f'stretch_{name}', p, base); out[f'stretch_{name}']['slopes'] = slopes
    hexB = B.copy(); p = np.full_like(B, np.nan); par = {}
    for f in range(5):
        a_, b_, w = prm[str(f)]['a'], prm[str(f)]['b'], prm[str(f)]['w']['0']
        h = B.copy(); h[have, 2] = np.exp((1 - w) * np.log(B[have, 2]) + w * (a_ + b_ * fa[have]))
        hexB[fold == f] = h[fold == f]
        s = fit_stretch(h, y, fold != f); par[f] = [[r4(v) for v in x] for x in s]
        p[fold == f] = apply_stretch(h[fold == f], s)
    rep('hex6_nested', hexB, B)
    rep('hex6_then_stretch_nested', p, B); out['hex6_then_stretch_nested']['params_g_c_mu'] = par
    rep('hex6_then_stretch_vs_hex6', p, hexB)
    (C.OUT / 'stretch_nested.json').write_text(json.dumps(out, indent=1, default=float))

if __name__ == '__main__':
    main()
