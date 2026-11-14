"""Audit A1, part 4: can a correction reach the -22 excess from the image alone? Oracle ceiling first, then image-only, all nested by fold, saved in best_pipeline format (lane GPU 0, 2026-11-06)."""
import re, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HC, HistGradientBoostingRegressor as HR
from sklearn.metrics import roc_auc_score
sys.path.insert(0, 'experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
E = BP.E
OUT = 'experiments/lane_gpu0_20261106/a1'
z = np.load('experiments/lane_gpu0_20261106/frame_stats.npz', allow_pickle=True)
S = z['stats'].astype(np.float64)[BP.idx]; LP = z['log_power'].astype(np.float64)[BP.idx]
X = np.c_[S, LP]
ids, y, P, fold, H = BP.ids, BP.y, BP.REF_BEST, BP.fold, BP.H
res = np.log(P) - np.log(np.where(y > 0, y, np.nan))
sa = np.array([bool(re.fullmatch(r'\d+-\d\d', i)) for i in ids])
yy = np.array([int(i[-2:]) if a else -1 for i, a in zip(ids, sa)]); ser = np.array([int(i.split('-')[0]) if a else -1 for i, a in zip(ids, sa)])
m22 = sa & (yy == 22); win = m22 & (ser >= 173) & (ser <= 550)
print('rows -22', m22.sum(), 'window 173-550', win.sum(), 'window share', round(win.mean(), 4))


def report_cor(name, C):
    Q = P * np.exp(C)
    return BP.report(name, Q, P)


# Oracle: bias of -22 rows, of the window, from other folds' rows, no shrink. IDs are not available at test.
for label, mask in (('oracle: all -22 rows get their own bias', m22), ('oracle: window 173-550 gets its own bias', win)):
    C = np.zeros_like(P)
    for f in range(5):
        b = np.nanmean(res[mask & (fold != f)], 0)
        C[mask & (fold == f)] = -b
    report_cor(label, C)
# Upper bound: the rows' own bias from all folds, leaking, on the window only.
C = np.zeros_like(P); C[win] = -np.nanmean(res[win], 0); report_cor('leaking upper bound: window, own-row bias', C)

# Image only: does the window separate from other A rows with the same grey-level count?
same = sa & (S[:, list(z['names'].astype(str)).index('levels')] <= 125)
print('rows with levels <= 125 in A', same.sum(), 'window rows among them', (win & same).sum())
scores = np.zeros(len(y)); scores22 = np.zeros(len(y))
for f in range(5):
    tr = fold != f
    for lab, mask, out in (('window', win, scores), ('-22', m22, scores22)):
        m = HC(max_iter=150, learning_rate=0.05, max_depth=3, l2_regularization=5.0, random_state=0).fit(X[tr & sa], mask[tr & sa].astype(int))
        out[fold == f] = m.predict_proba(X[fold == f])[:, 1]
print('AUC within A rows: window', round(roc_auc_score(win[sa], scores[sa]), 3), '-22', round(roc_auc_score(m22[sa], scores22[sa]), 3), 'window within levels<=125', round(roc_auc_score(win[same], scores[same]), 3))
print('score share of positives in the top 5 % of A rows: window', round(float(win[sa][np.argsort(-scores[sa])[:int(0.05 * sa.sum())]].mean()), 3), '-22', round(float(m22[sa][np.argsort(-scores22[sa])[:int(0.05 * sa.sum())]].mean()), 3))
# Correction from the scores: residual regressed on the score, slope fit in the training folds.
Cw, C22 = {}, {}
for name, sc, store in (('window', scores, Cw), ('-22', scores22, C22)):
    C = np.zeros_like(P)
    for f in range(5):
        tr = fold != f
        for j in range(3):
            ok = tr & ~np.isnan(res[:, j])
            A = np.c_[np.ones(ok.sum()), sc[ok]]
            beta = np.linalg.lstsq(A, res[ok, j], rcond=None)[0]
            C[fold == f, j] = -(beta[1] * (sc[fold == f] - sc[ok].mean()))
    report_cor(f'image only: residual on the {name} score, slope from training folds', C)
    for f in range(5):
        store[f] = C
    BP.save(f'{OUT}/cor_{name}_score.npz', store)
# General image-statistic residual model, same features, strong regularisation.
C = np.zeros_like(P)
for f in range(5):
    tr = fold != f
    for j in range(3):
        ok = tr & ~np.isnan(res[:, j]) & (np.abs(res[:, j]) < 0.5)
        m = HR(max_iter=120, learning_rate=0.03, max_depth=3, l2_regularization=10.0, min_samples_leaf=60, loss='absolute_error', random_state=0).fit(X[ok], res[ok, j])
        C[fold == f, j] = -m.predict(X[fold == f])
    print('fold', f, 'done', flush=True)
report_cor('image only: gradient boosting on 90 label-free frame statistics', C)
BP.save(f'{OUT}/cor_frame_stats_gbdt.npz', {f: C for f in range(5)})
