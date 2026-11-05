"""Read the multi-detector window statistics (2026-11-05). LOFO HGB on the log residual of B + hex6 + stretch, per metric, feature sets chosen before reading: single pre-only K = 3 (the reference line, -0.032 on HEX with the earlier statistics); all detectors mean of log, K = 3 and K = 5; with between-detector SD; with rich side-count statistics. Whole-frame controls come from window_hex_ablation. Gate for a new best: mean <= -0.02 against B + hex6 + stretch, CI upper bound < 0, at least 4 of 5 folds negative. Also a linear residual fit on hex6 and cell count for a deployable simple form."""
import json, sys, warnings
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import stretch_nested_20261103 as N
E = C.E; K = list(C.K); b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']; g = E.donors(b['ids']); O = C.X / 'meta_ceiling_20261103/sweep'
z = np.load(O / 'window_multi.npz'); F = z['F'][idx].astype(np.float64); fam = [str(x) for x in z['families']]; ks = z['ks'].tolist(); st = [str(x) for x in z['stats']]; print('shape', F.shape, 'finite share per family', np.round(np.isfinite(F[:, :, 0, 0]).mean(0), 3).tolist(), flush=True)
gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); a_, b_, w_ = -0.2084, 0.5989, 0.28; have = gf['f_active10_valid'] & ~gf['seen_leaked'] & np.isfinite(gf['f_active10']); good = ~gf['seen_leaked']; f6 = gf['f_active10'].astype(float); H = B.copy(); H[have, 2] = np.exp((1 - w_) * np.log(B[have, 2]) + w_ * (a_ + b_ * f6[have]))
def pipeline(base):
    o = np.full_like(base, np.nan)
    for f in range(5): o[fold == f] = N.apply_stretch(base[fold == f], N.fit_stretch(base, y, (fold != f) & good))
    return o
REF = pipeline(H); ly = np.log(np.where(y > 0, y, np.nan)); r = ly - np.log(REF)
def lg(X): return np.where(X > 0, np.log(np.where(X > 0, X, 1)), np.nan)
def mean_of(K_, stats, fams=None):
    fams = range(len(fam)) if fams is None else fams; b_i = ks.index(K_); return np.stack([np.nanmean(lg(F[:, list(fams), b_i, st.index(s)]), 1) for s in stats], 1)
def sd_of(K_, stats):
    b_i = ks.index(K_); return np.stack([np.nanstd(lg(F[:, :, b_i, st.index(s)]), 1) for s in stats], 1)
BASIC = ['hex6', 'cv_area', 'cv_dist', 'cd', 'n']; RICH = st
pre = [fam.index('preonly')]; pre3 = [fam.index(q) for q in ('preonly', 'preonly_s47', 'preonly_s48')]; sg = [fam.index(q) for q in fam if q.startswith('sg2')]
SETS = {'preonly K3 basic': mean_of(3, BASIC, pre), 'preonly K3 rich': mean_of(3, RICH, pre), 'preonly x3 K3 rich': mean_of(3, RICH, pre3), 'sg2 x5 K3 rich': mean_of(3, RICH, sg), 'all8 K3 basic': mean_of(3, BASIC), 'all8 K3 rich': mean_of(3, RICH), 'all8 K5 rich': mean_of(5, RICH), 'all8 K3+K5 rich': np.c_[mean_of(3, RICH), mean_of(5, RICH)], 'all8 K3 rich + detector SD': np.c_[mean_of(3, RICH), sd_of(3, ['hex6', 'cv_dist', 'cd'])], 'all8 K3 rich + preds + hex6': np.c_[mean_of(3, RICH), np.log(H), f6]}
def fill(X):
    X = X.copy(); med = np.nanmedian(X, 0); bad = ~np.isfinite(X); X[bad] = np.take(med, np.nonzero(bad)[1]); return X
def fit(X, ms):
    P = np.full_like(H, np.nan)
    for f in range(5):
        tr = (fold != f) & good; C_ = H.copy()
        for j in ms:
            q = tr & (y[:, j] > 0); t = np.log(y[q, j]) - np.log(H[q, j]); m = HGB(loss='absolute_error', max_depth=3, max_iter=200, learning_rate=0.03, min_samples_leaf=80, l2_regularization=5.0, random_state=0).fit(X[q], t); C_[:, j] = H[:, j] * np.exp(m.predict(X))
        P[fold == f] = N.apply_stretch(C_[fold == f], N.fit_stretch(C_, y, tr))
    return P
def report(nm, p):
    pb = E.paired_bootstrap(E.ape(p, y), E.ape(REF, y), g); s, s0 = E.official(E.ape(p, y)), E.official(E.ape(REF, y)); per = [round(E.official(E.ape(p[fold == f], y[fold == f]))['mean'] - E.official(E.ape(REF[fold == f], y[fold == f]))['mean'], 3) for f in range(5)]; o = dict(score=round(s['mean'], 3), per_delta={k: round(s[k] - s0[k], 3) for k in K}, delta=round(pb['delta'], 4), ci=[round(v, 4) for v in pb['ci95']], folds=per); print(nm, json.dumps(o), flush=True); return o
def cor(a, c):
    s = np.isfinite(a) & np.isfinite(c); return round(float(np.corrcoef(a[s], c[s])[0, 1]), 3)
for nm, v in dict(preonly=lg(F[:, pre[0], 0, st.index('hex6')]), all8=mean_of(3, ['hex6'])[:, 0]).items(): print('K3 hex6', nm, 'r log HEX', cor(v, ly[:, 2]), 'r final residual', cor(v, r[:, 2]), '| ten-detector whole-frame hex6', cor(f6, ly[:, 2]), flush=True)
rep = {}
for nm, X in SETS.items():
    rep[nm + ' HEX'] = report(nm + ' -> HEX', fit(fill(X), [2]))
best = min((k for k in rep), key=lambda k: rep[k]['delta']).replace(' HEX', ''); print('best set', best, flush=True); X = fill(SETS[best]); rep['best CV'] = report(best + ' -> CV', fit(X, [1])); rep['best CD'] = report(best + ' -> CD', fit(X, [0])); rep['best all3'] = report(best + ' -> all', fit(X, [0, 1, 2]))
np.savez_compressed(O / 'window_multi_best_pred.npz', pred=fit(X, [2]), ref=REF)
json.dump(rep, open(O / 'window_multi_eval.json', 'w'), indent=1); print('done', flush=True)
