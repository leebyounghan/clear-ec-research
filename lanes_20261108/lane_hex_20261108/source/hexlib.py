"""HEX lane helpers (2026-11-08). Reference = best pipeline (BP.REF_BEST). A HEX try is a log correction on top of BP.BEST,
one N-vector per outer fold, from a fit that did not use that fold's labels; BP.pipe then fits the stretch nested.
evaluate() reports HEX on folds 0+4 (screen) and on all five folds, paired donor bootstrap against the reference."""
import sys, json, time
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
E = BP.E; y, fold, idx, donors, good = BP.y, BP.fold, BP.idx, BP.donors, BP.good
N = len(y); LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
REF = BP.REF_BEST
HB = {f: BP.H[:, 2] * np.exp(BP.BEST[f][:, 2]) for f in range(5)}   # pre-stretch HEX of the reference, per outer fold model
S04 = (fold == 0) | (fold == 4)


def hexape(p, rows):
    ok = rows & (y[:, 2] > 0); return float(np.mean(np.abs(p[ok, 2] - y[ok, 2]) / y[ok, 2]) * 100)


def boot(p, rows, reps=2000, seed=0):
    ok = rows & (y[:, 2] > 0); e1 = np.abs(p[ok, 2] - y[ok, 2]) / y[ok, 2] * 100; e0 = np.abs(REF[ok, 2] - y[ok, 2]) / y[ok, 2] * 100
    g = np.asarray(donors)[ok]; u, inv = np.unique(g, return_inverse=True); d = np.bincount(inv, e1 - e0); c = np.bincount(inv)
    rng = np.random.default_rng(seed); s = []
    for _ in range(reps):
        k = rng.integers(0, len(u), len(u)); s.append(d[k].sum() / c[k].sum())
    return float((e1 - e0).mean()), [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def evaluate(name, corhex, log=None, note=''):
    cor = {f: BP.BEST[f] + np.c_[np.zeros(N), np.zeros(N), np.nan_to_num(corhex[f])] for f in range(5)}
    P = BP.pipe(cor)
    o = dict(name=name, hex04=round(hexape(P, S04), 4), ref04=round(hexape(REF, S04), 4), hex5=round(hexape(P, np.ones(N, bool)), 4), ref5=round(hexape(REF, np.ones(N, bool)), 4))
    o['d04'], o['ci04'] = boot(P, S04); o['d5'], o['ci5'] = boot(P, np.ones(N, bool))
    o['folds'] = [round(hexape(P, fold == f) - hexape(REF, fold == f), 3) for f in range(5)]
    o = {k: (round(v, 4) if isinstance(v, float) else ([round(q, 4) for q in v] if isinstance(v, list) else v)) for k, v in o.items()}
    print(json.dumps(o), flush=True)
    if log:
        with open(f'{LANE}/RESULTS.tsv', 'a') as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime())}Z\t{name}\t0,4|all5\t{o['hex04']}|{o['hex5']}\t{o['d04']:+.4f} {o['ci04']}|{o['d5']:+.4f} {o['ci5']}\t{note} folds {o['folds']}\n")
    return o, P


def nested(F, model='ridge', alpha=10.0, base_in=True, hgb=None):
    """F: N x k features aligned to base rows (nan allowed), or {fold: N x k} from per-fold models. Fits log y_hex - log HB[f] on rows of folds != f; returns {f: N}."""
    PF = F if isinstance(F, dict) else None; F = None if PF is not None else np.asarray(F, float).reshape(N, -1); out = {}
    for f in range(5):
        lb = np.log(HB[f]); X = (np.asarray(PF[f], float).reshape(N, -1) if PF is not None else F).copy(); miss = ~np.isfinite(X)
        tr = (fold != f) & good & (y[:, 2] > 0)
        med = np.nanmedian(X[tr], 0); X[miss] = np.take(med, np.where(miss)[1])
        if miss.any(): X = np.c_[X, miss.any(1).astype(float)]
        if base_in: X = np.c_[X, lb]
        t = np.log(np.where(y[:, 2] > 0, y[:, 2], 1)) - lb
        if model == 'ridge':
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Z = (X - mu) / sd
            A = Z[tr]; w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (t[tr] - t[tr].mean())); out[f] = Z @ w + t[tr].mean()
        elif model == 'l1':   # MAPE-ish: median regression via IRLS
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Z = np.c_[np.ones(N), (X - mu) / sd]; A = Z[tr]; tt = t[tr]; wt = np.ones(len(tt))
            for _ in range(30):
                W = A * wt[:, None]; b = np.linalg.solve(A.T @ W + alpha * np.diag([0] + [1] * (A.shape[1] - 1)), W.T @ tt); wt = 1 / np.maximum(np.abs(tt - A @ b), 1e-3)
            out[f] = Z @ b
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            m = HistGradientBoostingRegressor(loss='absolute_error', **(hgb or dict(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.0)), random_state=0)
            m.fit(X[tr], t[tr]); out[f] = m.predict(X)
    return out
