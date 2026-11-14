"""HEX stackers on top of the best pipeline (HEX lane, 2026-11-08). Each try: features -> nested fit per outer fold on the
other folds -> log correction on HEX -> BP.pipe (nested stretch) -> HEX vs best pipeline. Models: ridge on log residual;
'mape' = gradient boosting with absolute loss on the ratio y/HB weighted by HB/y, which is exactly MAPE, so it learns the
per-image MAPE-optimal shift as well as signal. usage: stack_hex.py TRY [TRY...]  (tries defined below)"""
import sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source')
import hexlib as L
BP = L.BP; N = L.N; y, fold, idx, good = L.y, L.fold, L.idx, L.good; X = BP.C.X; O = X / 'meta_ceiling_20261103/sweep'
from sklearn.ensemble import HistGradientBoostingRegressor


def lg(v): v = np.asarray(v, float); return np.where(v > 0, np.log(np.where(v > 0, v, 1)), np.nan)


def feats_base(f):
    P = BP.H * np.exp(BP.BEST[f]); return np.c_[lg(P[:, 0]), lg(P[:, 1]), lg(P[:, 2])]


def gfe():
    g = np.load(X / 'submission2_search_20261103/green_features.npz', allow_pickle=True)
    cols = [g['f_active10'], g['n_det'], g['black_frac']] + [g[k] for k in g.files if k.startswith('e2_') and not k.endswith('_valid')]
    return np.c_[tuple(np.asarray(c, float) for c in cols)]


def legible(): return np.load(O / 'legible_window.npz')['F'][idx].astype(float)


def wmulti(): z = np.load(O / 'window_multi.npz'); return z['F'][idx].reshape(len(idx), -1).astype(float)


def fused(path, keys=None):
    z = np.load(path); nm = [str(s) for s in z['names']]; F = z['F'][idx].astype(float)
    if keys: F = F[:, [nm.index(k) for k in keys if k in nm]]
    return F, nm


def mape_gbm(Fd, params=None, label=''):
    params = params or dict(max_iter=250, learning_rate=0.03, max_leaf_nodes=12, min_samples_leaf=100, l2_regularization=1.0)
    out = {}
    for f in range(5):
        Xf = Fd[f]; hb = L.HB[f]; tr = (fold != f) & good & (y[:, 2] > 0); t = y[:, 2] / hb; w = hb / np.where(y[:, 2] > 0, y[:, 2], 1)
        m = HistGradientBoostingRegressor(loss='absolute_error', random_state=0, **params); m.fit(Xf[tr], t[tr], sample_weight=w[tr]); out[f] = np.log(np.clip(m.predict(Xf), 0.5, 2.0))
    return out


TRY = {}
def T(fn): TRY[fn.__name__] = fn; return fn


@T
def mape_base():     # calibration + per-image shift from the pipeline's own outputs only
    return mape_gbm({f: feats_base(f) for f in range(5)})


@T
def mape_green():
    G = gfe(); return mape_gbm({f: np.c_[feats_base(f), G] for f in range(5)})


@T
def mape_green_win():
    G = np.c_[gfe(), legible(), wmulti()]; return mape_gbm({f: np.c_[feats_base(f), G] for f in range(5)})


@T
def ridge_win():
    G = np.c_[gfe(), legible()]; return L.nested({f: np.c_[feats_base(f)[:, :2], G] for f in range(5)}, alpha=30.0)


def fused_try(path, tag):
    Fz, nm = fused(path); h6 = [k for k in nm if k.endswith('_h6') or k.endswith('_wh6') or k.endswith('_h6c') or k.endswith('_wh6c') or k.endswith('_w1h6')]
    Fh, _ = fused(path, h6); Fh = lg(Fh)
    r1 = L.evaluate(f'{tag} ridge hex6 variants', L.nested({f: Fh for f in range(5)}, alpha=30.0), log=True, note=f'{path} {len(h6)} log hex6 variants, ridge on log residual')
    r2 = L.evaluate(f'{tag} mape-gbm all fused', mape_gbm({f: np.c_[feats_base(f), Fz] for f in range(5)}), log=True, note=f'{path} {Fz.shape[1]} fused feats + base preds')
    # single strongest variants one by one
    best = []
    for j, k in enumerate(h6):
        o, _ = L.evaluate(f'{tag} ridge {k}', L.nested({f: Fh[:, j] for f in range(5)}, alpha=10.0)); best.append((o['d5'], k))
    print(sorted(best)[:8], flush=True)


if __name__ == '__main__':
    for t in sys.argv[1:]:
        if t.startswith('fused:'):
            _, path, tag = t.split(':'); fused_try(path, tag); continue
        L.evaluate(t, TRY[t](), log=True, note=(TRY[t].__doc__ or '').strip()[:80])
