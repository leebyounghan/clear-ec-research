"""Audit A1, part 2: scan all acquisition order, not only -22. Bins in serial order within source and year; oracle ceiling of a per-bin bias correction, nested by fold; which bins stand out (lane GPU 0, 2026-11-06)."""
import re, sys
import numpy as np
sys.path.insert(0, 'experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
E = BP.E
z = np.load('experiments/lane_gpu0_20261106/frame_stats.npz', allow_pickle=True)
names = list(z['names'].astype(str)); S = z['stats'].astype(np.float64)[BP.idx]
ids, y, P, fold, donors = BP.ids, BP.y, BP.REF_BEST, BP.fold, BP.donors
err = E.ape(P, y); res = np.log(P) - np.log(np.where(y > 0, y, np.nan))
sa = np.array([bool(re.fullmatch(r'\d+-\d\d', i)) for i in ids])
yr = np.array([2000 + int(i[-2:]) if a else int(i[:4]) for i, a in zip(ids, sa)])
ser = np.array([int(i.split('-')[0]) if a else int(re.match(r'\d{4}-(\d+)', i).group(1)) for i, a in zip(ids, sa)])
BIN = 60
groups = np.full(len(y), -1); table = []; g = 0
for s_, name in ((True, 'A'), (False, 'B')):
    for v in sorted(set(yr[sa == s_])):
        rows = np.flatnonzero((sa == s_) & (yr == v)); rows = rows[np.argsort(ser[rows], kind='stable')]
        for c in np.array_split(rows, max(1, len(rows) // BIN)):
            groups[c] = g
            e = np.nanmean(err[c], 0); b = np.nanmean(res[c], 0); sd = np.nanstd(res[c], 0)
            table.append((f'{name}{v}', ser[c].min(), ser[c].max(), len(c), *e, e.mean(), *b, *sd, np.median(S[c, names.index('levels')]), np.median(S[c, names.index('hp_std')]), np.median(S[c, names.index('mean')]), *np.nanmean(np.where(y[c] > 0, y[c], np.nan), 0)))
            g += 1
print('group serial n | err CD CV HEX mean | bias CD CV HEX | resSD CD CV HEX | levels hp_std mean | label CD CV HEX')
for t in table:
    print(f'{t[0]} {t[1]:5d}-{t[2]:<5d} {t[3]:3d} | {t[4]:6.2f} {t[5]:6.2f} {t[6]:6.2f} {t[7]:6.2f} | {t[8]:+.3f} {t[9]:+.3f} {t[10]:+.3f} | {t[11]:.3f} {t[12]:.3f} {t[13]:.3f} | {t[14]:4.0f} {t[15]:5.1f} {t[16]:5.1f} | {t[17]:5.0f} {t[18]:.3f} {t[19]:.3f}')
# Oracle ceiling: per-bin log bias from the other folds' rows in the bin, shrunk k/(k+n), best k by nested choice over the other folds.
ok = y > 0
ref = P


def fit_apply(fold_out, k):
    out = np.zeros_like(P)
    tr = fold != fold_out
    for j in range(3):
        for gg in np.unique(groups):
            m = (groups == gg) & tr & ok[:, j]
            if m.sum():
                out[groups == gg, j] = -np.nanmean(res[m, j]) * m.sum() / (m.sum() + k)
    return out


for k in (5, 20, 60):
    Q = P.copy()
    for f in range(5):
        Q[fold == f] = (P * np.exp(fit_apply(f, k)))[fold == f]
    BP.report(f'oracle batch bias, 60-row bins in serial order, shrink k={k}', Q, ref)
# How concentrated is the gain in anomalous bins?
tt = np.array([t[7] for t in table]); print('bin mean-error quantiles', np.percentile(tt, [5, 25, 50, 75, 95]).round(2).tolist(), 'top bins', sorted([(round(t[7], 2), t[0], int(t[1]), int(t[2])) for t in table], reverse=True)[:12])
# B 2019 CV: what sets it apart?
m = (~sa) & (yr == 2019)
for lab, c in (('B 2019', m), ('B 2020', (~sa) & (yr == 2020)), ('B 2021', (~sa) & (yr == 2021)), ('B 2022', (~sa) & (yr == 2022))):
    r = np.corrcoef(np.log(y[c & ok[:, 1], 1]), np.log(P[c & ok[:, 1], 1]))[0, 1]
    q = np.percentile(y[c & ok[:, 1], 1], [1, 5, 25, 50, 75, 95, 99]).round(3).tolist()
    pq = np.percentile(P[c, 1], [1, 5, 25, 50, 75, 95, 99]).round(3).tolist()
    print(lab, 'CV r', round(r, 3), 'label pct', q, 'pred pct', pq, 'unique CV labels', len(np.unique(y[c, 1])))
