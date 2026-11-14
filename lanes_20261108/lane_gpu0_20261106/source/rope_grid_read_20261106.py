"""Read rope_grid: each variant against its own scale-1 prediction per member, the same stretch on both, fit on the other folds read. Usage: rope_grid_read_20261106.py <folds> <seeds> <variants>"""
import sys
from pathlib import Path
import numpy as np
sys.path[:0] = ['experiments/meta_ceiling_20261103/source', 'experiments/eval_harness_20261031/source']
import eval_oof_20261031 as E, stretch_nested_20261103 as N
base = E.load_base(); gp = E.donors(base['ids']); pos = {int(i): n for n, i in enumerate(base['idx'])}; Y = base['y']
G = Path('experiments/lane_gpu0_20261106/rope_grid')
folds = [int(v) for v in sys.argv[1].split(',')]; seeds = [int(v) for v in sys.argv[2].split(',')]; variants = sys.argv[3].split(',')


def load(f, s, v):
    z = np.load(G / f'f{f}_s{s}_{v}.npz'); return np.array([pos[int(i)] for i in z['indices']]), z['pred'].astype(np.float64)


for s in seeds + ['avg']:
    use = seeds if s == 'avg' else [s]
    P = {}
    for v in variants + ['1.0']:
        P[v] = {}
        for f in folds:
            ok = [q for q in use if (G / f'f{f}_s{q}_{v}.npz').exists()]
            if len(ok) != len(use):
                P[v][f] = None; continue
            rows = load(f, ok[0], v)[0]; P[v][f] = (rows, np.exp(np.mean([np.log(load(f, q, v)[1]) for q in ok], 0)))

    def cal(v):
        out = []
        for f in folds:
            oth = [o for o in folds if o != f]
            if any(P[v][q] is None for q in folds):
                return None
            rows = np.concatenate([P[v][o][0] for o in oth]); p = np.concatenate([P[v][o][1] for o in oth])
            st = N.fit_stretch(p, Y[rows], np.ones(len(rows), bool))
            out.append((P[v][f][0], N.apply_stretch(P[v][f][1], st)))
        return np.concatenate([a for a, _ in out]), np.concatenate([c for _, c in out])
    ref = cal('1.0')
    if ref is None:
        continue
    rows, pr = ref; s0 = E.official(E.ape(pr, Y[rows]))
    print(f'## seed {s}, folds {folds}, scale 1 calibrated {s0["mean"]:.4f} CD {s0["CD"]:.3f} CV {s0["CV"]:.3f} HEX {s0["HEX"]:.3f}')
    for v in variants:
        c = cal(v)
        if c is None:
            continue
        r2, p = c; assert np.array_equal(rows, r2)
        pb = E.paired_bootstrap(E.ape(p, Y[rows]), E.ape(pr, Y[rows]), gp[rows]); s1 = E.official(E.ape(p, Y[rows]))
        per = []
        i = 0
        for f in folds:
            n = len(P[v][f][0]); per.append(round(E.official(E.ape(p[i:i + n], Y[rows[i:i + n]]))['mean'] - E.official(E.ape(pr[i:i + n], Y[rows[i:i + n]]))['mean'], 3)); i += n
        print(v.ljust(14), 'mean', round(s1['mean'], 4), 'delta', round(pb['delta'], 4), 'CI', np.round(pb['ci95'], 4).tolist(), 'CD', round(s1['CD'] - s0['CD'], 4), 'CV', round(s1['CV'] - s0['CV'], 4), 'HEX', round(s1['HEX'] - s0['HEX'], 4), 'folds', per, flush=True)
