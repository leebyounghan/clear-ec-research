"""Hyphen-form rows of B 2019: what differs, is it visible in the image, is there anything to correct? (lane GPU 0, 2026-11-06)"""
import re, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HC
from sklearn.metrics import roc_auc_score
sys.path.insert(0, 'experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
E = BP.E
z = np.load('experiments/lane_gpu0_20261106/frame_stats.npz', allow_pickle=True)
names = list(z['names'].astype(str)); S = z['stats'].astype(np.float64)[BP.idx]; LP = z['log_power'].astype(np.float64)[BP.idx]; X = np.c_[S, LP]
ids, y, P, fold, donors = BP.ids, BP.y, BP.REF_BEST, BP.fold, BP.donors
err = E.ape(P, y); res = np.log(P) - np.log(np.where(y > 0, y, np.nan))
hy = np.array([bool(re.match(r'\d{4}-\d+-', i)) for i in ids]); sp = np.array([bool(re.match(r'2019-\d+ ', i)) for i in ids])
print('rows hyphen', hy.sum(), 'space 2019', sp.sum(), 'donors with both forms', len(set(donors[hy]) & set(donors[sp])), 'donors hyphen', len(set(donors[hy])))
for lab, m in (('hyphen', hy), ('space 2019', sp), ('space 2020-22', np.array([bool(re.match(r'20(20|21|22)-\d+ ', i)) for i in ids]))):
    pc = np.nanpercentile(res[m], [1, 5, 25, 50, 75, 95, 99], axis=0).round(3)
    print(lab.ljust(14), 'CV res pct', pc[:, 1].tolist(), 'label CV pct', np.percentile(y[m, 1], [1, 5, 25, 50, 75, 95, 99]).round(3).tolist(), 'pred CV pct', np.percentile(P[m, 1], [1, 5, 25, 50, 75, 95, 99]).round(3).tolist(), 'r log CV', round(float(np.corrcoef(np.log(y[m & (y[:, 1] > 0), 1]), np.log(P[m & (y[:, 1] > 0), 1]))[0, 1]), 3))
    print(' ' * 14, 'trimmed mean err, top 1 % of rows removed', np.round([np.sort(err[m, j][~np.isnan(err[m, j])])[:int(0.99 * m.sum())].mean() for j in range(3)], 2).tolist(), 'r CD', round(float(np.corrcoef(np.log(y[m, 0]), np.log(P[m, 0]))[0, 1]), 3), 'r HEX', round(float(np.corrcoef(np.log(y[m & (y[:, 2] > 0), 2]), np.log(P[m & (y[:, 2] > 0), 2]))[0, 1]), 3))
print('\n## frame statistics, Cohen d hyphen against space 2019, top 14')
D = []
for j, n in enumerate(names):
    a, b = S[hy, j], S[sp, j]; s = np.sqrt((a.var() + b.var()) / 2) + 1e-9
    D.append((abs(a.mean() - b.mean()) / s, n, (a.mean() - b.mean()) / s, np.median(a), np.median(b)))
for _, n, d, ma, mb in sorted(D, reverse=True)[:14]:
    print(n.ljust(20), f'd {d:+.2f} median {ma:.3f} against {mb:.3f}')
print('\n## by serial, 2019, the two forms over serial bins of 100')
ser = np.array([int(re.match(r'\d{4}-(\d+)', i).group(1)) if re.match(r'\d{4}-\d+', i) else -1 for i in ids])
for lo in range(0, 1000, 100):
    h = hy & (ser >= lo) & (ser < lo + 100); s_ = sp & (ser >= lo) & (ser < lo + 100)
    print(lo, 'hyphen', int(h.sum()), np.round(np.nanmean(err[h], 0), 1).tolist() if h.sum() else '', 'space', int(s_.sum()), np.round(np.nanmean(err[s_], 0), 1).tolist() if s_.sum() else '')
# Image detection, nested.
sc = np.zeros(len(y)); B2019 = hy | sp
Bany = np.array([not re.fullmatch(r'\d+-\d\d', i) for i in ids])
for f in range(5):
    tr = (fold != f) & Bany
    m = HC(max_iter=150, learning_rate=0.05, max_depth=3, l2_regularization=5.0, random_state=0).fit(X[tr], hy[tr].astype(int))
    sc[fold == f] = m.predict_proba(X[fold == f])[:, 1]
print('\nAUC hyphen within source B', round(roc_auc_score(hy[Bany], sc[Bany]), 3), 'within B 2019', round(roc_auc_score(hy[B2019], sc[B2019]), 3))
# Oracle: hyphen rows get a nested log-affine fit of label on prediction, by metric.
Q = np.log(P).copy()
for f in range(5):
    for j in range(3):
        k = hy & (fold != f) & (y[:, j] > 0)
        A = np.c_[np.ones(k.sum()), np.log(P[k, j])]
        from sklearn.linear_model import QuantileRegressor as QR
        q = QR(quantile=0.5, alpha=0.0, solver='highs').fit(np.log(P[k, j])[:, None], np.log(y[k, j]))
        t = hy & (fold == f)
        Q[t, j] = q.predict(np.log(P[t, j])[:, None])
BP.report('oracle IDs: hyphen rows get their own median log-affine map', np.exp(Q), P)
# Image-only: score-gated version.
for thr in (0.5, 0.8):
    Q = np.log(P).copy()
    for f in range(5):
        for j in range(3):
            k = (sc > thr) & (fold != f) & (y[:, j] > 0) & Bany
            if k.sum() < 50:
                continue
            q = QR(quantile=0.5, alpha=0.0, solver='highs').fit(np.log(P[k, j])[:, None], np.log(y[k, j]))
            t = (sc > thr) & (fold == f) & Bany
            Q[t, j] = q.predict(np.log(P[t, j])[:, None])
    BP.report(f'image only: rows with hyphen score above {thr} get a median log-affine map', np.exp(Q), P)
