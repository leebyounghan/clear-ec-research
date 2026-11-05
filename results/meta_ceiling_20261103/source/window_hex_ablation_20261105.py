"""Which features carry the HEX gain of the residual fit (2026-11-05)? -0.023 (CI -0.038, -0.009) on top of B + hex6 + stretch, five folds negative. Same LOFO HGB on HEX with feature groups: hex6 alone (nonlinear calibration of the shipped feature, the control), B's own predictions as features, window groups, emulator, and combinations; also with three seeds of the booster to see fit noise. A window effect must beat the hex6-alone control."""
import json, sys, warnings
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import stretch_nested_20261103 as N
E = C.E; K = list(C.K); b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']; g = E.donors(b['ids']); O = C.X / 'meta_ceiling_20261103/sweep'
z = np.load(O / 'legible_window.npz'); nw = [str(n) for n in z['names']]; W = z['F'][idx].astype(float); ze = np.load(O / 'technician_emulator.npz'); ne = [str(n) for n in ze['names']]; Em = ze['F'][idx].astype(float)
gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); a_, b_, w_ = -0.2084, 0.5989, 0.28; have = gf['f_active10_valid'] & ~gf['seen_leaked'] & np.isfinite(gf['f_active10']); good = ~gf['seen_leaked']; f6 = gf['f_active10'].astype(float); H = B.copy(); H[have, 2] = np.exp((1 - w_) * np.log(B[have, 2]) + w_ * (a_ + b_ * f6[have]))
def pipeline(base):
    o = np.full_like(base, np.nan)
    for f in range(5): o[fold == f] = N.apply_stretch(base[fold == f], N.fit_stretch(base, y, (fold != f) & good))
    return o
REF = pipeline(H)
def lg(X):
    X = np.where(X > 0, np.log(np.where(X > 0, X, 1)), np.nan); return X
def cols(prefix, stats=None):
    return lg(np.stack([W[:, nw.index(n)] for n in nw if n.startswith(prefix) and not n.endswith(('_x', '_y')) and (stats is None or n.rsplit('_', 1)[1] in stats or n[len(prefix) + 1:] in stats)], 1))
GROUPS = dict(hex6=f6[:, None], preds=np.log(H), top_hull=cols('top_hull'), top3_hull=cols('top3_hull'), top_box=cols('top_box'), top3_box=cols('top3_box'), whole=cols('whole'), centre_hull=cols('centre_hull'), emu=lg(Em), window_xy=np.stack([W[:, nw.index(n)] for n in ('top_hull_x', 'top_hull_y', 'top_box_x', 'top_box_y')], 1))
def fill(X):
    X = X.copy(); med = np.nanmedian(X, 0); bad = ~np.isfinite(X); X[bad] = np.take(med, np.nonzero(bad)[1]); return X
def fit(X, seed=0, j=2):
    P = np.full_like(H, np.nan)
    for f in range(5):
        tr = (fold != f) & good; C_ = H.copy(); q = tr & (y[:, j] > 0); t = np.log(y[q, j]) - np.log(H[q, j]); m = HGB(loss='absolute_error', max_depth=3, max_iter=200, learning_rate=0.03, min_samples_leaf=80, l2_regularization=5.0, random_state=seed).fit(X[q], t); C_[:, j] = H[:, j] * np.exp(m.predict(X))
        P[fold == f] = N.apply_stretch(C_[fold == f], N.fit_stretch(C_, y, tr))
    return P
def report(nm, p):
    pb = E.paired_bootstrap(E.ape(p, y), E.ape(REF, y), g); s, s0 = E.official(E.ape(p, y)), E.official(E.ape(REF, y)); per = [round(E.official(E.ape(p[fold == f], y[fold == f]))['mean'] - E.official(E.ape(REF[fold == f], y[fold == f]))['mean'], 3) for f in range(5)]; o = dict(score=round(s['mean'], 3), HEX=round(s['HEX'] - s0['HEX'], 3), delta=round(pb['delta'], 4), ci=[round(v, 4) for v in pb['ci95']], folds=per); print(nm, json.dumps(o), flush=True); return o
rep = {}
SETS = {'hex6 only': ['hex6'], 'preds only': ['preds'], 'hex6 + preds': ['hex6', 'preds'], 'top_hull': ['top_hull'], 'top3_hull': ['top3_hull'], 'top_box': ['top_box'], 'whole (control)': ['whole'], 'centre_hull (control)': ['centre_hull'], 'emulator': ['emu'], 'window position only': ['window_xy'], 'hex6 + top_hull': ['hex6', 'top_hull'], 'hex6 + top3_hull': ['hex6', 'top3_hull'], 'hex6 + whole (control)': ['hex6', 'whole'], 'hex6 + centre_hull (control)': ['hex6', 'centre_hull'], 'hex6 + emulator': ['hex6', 'emu'], 'hex6 + top_hull + top3_hull + top_box': ['hex6', 'top_hull', 'top3_hull', 'top_box'], 'all windows + emulator (as read)': ['hex6', 'top_hull', 'top3_hull', 'top_box', 'emu'], 'all + preds': ['hex6', 'preds', 'top_hull', 'top3_hull', 'top_box', 'emu']}
for nm, gs in SETS.items():
    rep[nm] = report(nm, fit(fill(np.concatenate([GROUPS[q] for q in gs], 1))))
json.dump(rep, open(O / 'window_hex_ablation.json', 'w'), indent=1); print('done', flush=True)
