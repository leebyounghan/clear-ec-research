"""Frozen read, step 2 (lane GPU 1, 2026-11-06). Ridge on the pooled tokens of one encoder, no fine-tuning, deterministic.
Train rows: the fold's split file. Scored rows: the fold's validation rows in the shared OOF table. Ridge on standardised log targets, lambda per metric from a donor-grouped inner 5-fold on train rows only.
Usage: frozen_read_20261106.py <name> <fold> [--ref <name>] [--pool std,mean,both]
Prints delta against --ref with the donor bootstrap of the evaluation harness."""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E

LANE = Path('/workspace/clear-ec-research/experiments/lane_gpu1_20261106')
SPLITS = {0: 'ssl_screen_20260911/preflight/fold0.npz', 1: 'ssl_screen_20260911/preflight/fold1.npz', 2: 'overnight_target_20260912/preflight/fold2.npz',
          3: 'overnight_target_20260912/preflight/fold3.npz', 4: 'target_fold4_20260912/preflight/fold4.npz'}
K = ['CD', 'CV', 'HEX']
GRID = 10.0 ** np.arange(-5, 2.01, 0.5)
B = E.load_base()
GP = E.donors(B['ids'])
ROWMAP = np.full(8954, -1); ROWMAP[B['idx']] = np.arange(len(B['idx']))


def feats(name, pool):
    z = np.load(LANE / 'features' / f'{name}.npz')
    if pool == 'both':
        return np.concatenate([z['mean'], z['std']], 1).astype(np.float64)
    return z[pool].astype(np.float64)


def ridge_fit(X, Y, lams):
    """All lambdas at once through the eigendecomposition of X'X. Returns list of coefficient arrays."""
    w, V = np.linalg.eigh(X.T @ X)
    w = np.maximum(w, 0)
    XtY = V.T @ (X.T @ Y)
    return [V @ (XtY / (w + l)[:, None]) for l in lams]


def fit_predict(Xtr, Ytr, Xva, groups):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    ym, ys = Ytr.mean(0), Ytr.std(0)
    Z = (Ytr - ym) / ys
    n = len(Xtr)
    lams = GRID * n
    ug = np.unique(groups)
    inner = np.random.default_rng(0).integers(0, 5, len(ug))[np.searchsorted(ug, groups)]
    err = np.zeros((len(lams), 3))
    for k in range(5):
        a, b = inner != k, inner == k
        for li, c in enumerate(ridge_fit(Xtr[a], Z[a], lams)):
            p = Xtr[b] @ c
            err[li] += np.abs(np.expm1((p - Z[b]) * ys)).sum(0)
    best = err.argmin(0)
    co = ridge_fit(Xtr, Z, lams)
    pred = np.stack([(Xva @ co[best[m]])[:, m] for m in range(3)], 1)
    return np.exp(pred * ys + ym), [float(GRID[i]) for i in best]


def read(name, fold, pool):
    with np.load('/workspace/clear-ec-research/experiments/' + SPLITS[fold]) as z:
        tr = z['train'].copy()
    Y = np.full((8954, 3), np.nan); Y[B['idx']] = B['y']
    tr = tr[np.isfinite(Y[tr]).all(1) & (Y[tr] > 0).all(1)]
    rows = np.where(B['fold'] == fold)[0]; va = B['idx'][rows]
    X = feats(name, pool)
    g = np.full(8954, -1); g[B['idx']] = GP
    assert not np.intersect1d(tr, va).size and (g[tr] >= 0).all()
    p, lam = fit_predict(X[tr], np.log(Y[tr]), X[va], g[tr])
    return p, rows, lam


def main():
    name, fold = sys.argv[1], int(sys.argv[2])
    ref = sys.argv[sys.argv.index('--ref') + 1] if '--ref' in sys.argv else None
    pools = sys.argv[sys.argv.index('--pool') + 1].split(',') if '--pool' in sys.argv else ['std', 'mean', 'both']
    out = []
    for pool in pools:
        p, rows, lam = read(name, fold, pool)
        e = E.ape(p, B['y'][rows]); s = E.official(e)
        r = dict(name=name, fold=fold, pool=pool, n=len(rows), mean=round(s['mean'], 4), **{k: round(s[k], 4) for k in K}, lam=lam)
        if ref:
            p0, rows0, _ = read(ref, fold, pool)
            e0 = E.ape(p0, B['y'][rows0]); s0 = E.official(e0)
            pb = E.paired_bootstrap(e, e0, np.unique(GP[rows], return_inverse=True)[1])
            r.update(ref=ref, delta=round(pb['delta'], 4), ci95=[round(v, 4) for v in pb['ci95']], per={k: round(s[k] - s0[k], 4) for k in K})
        print(json.dumps(r), flush=True)
        out.append(r)
    d = LANE / 'reads'; d.mkdir(exist_ok=True)
    (d / f'frozen_{name}_f{fold}{"_vs_" + ref if ref else ""}.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
