"""Audit A1, part 1: what differs in rows with ID suffix -22, using the best pipeline's OOF and label-free frame statistics (lane GPU 0, 2026-11-06)."""
import re, sys
import numpy as np
sys.path.insert(0, 'experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
E = BP.E
z = np.load('experiments/lane_gpu0_20261106/frame_stats.npz', allow_pickle=True)
names = list(z['names'].astype(str)); S = z['stats'].astype(np.float64)[BP.idx]; LP = z['log_power'].astype(np.float64)[BP.idx]; freq = z['freq']
ids = BP.ids; y = BP.y; P = BP.REF_BEST
K = ['CD', 'CV', 'HEX']
m_a = np.array([bool(re.fullmatch(r'\d+-\d\d', i)) for i in ids])
yy = np.array([int(i[-2:]) if a else -1 for i, a in zip(ids, m_a)])
serial = np.array([int(i.split('-')[0]) if a else -1 for i, a in zip(ids, m_a)])
yb = np.array([int(i[:4]) if not a else -1 for i, a in zip(ids, m_a)])
err = E.ape(P, y); err0 = E.ape(BP.REF_SUB1, y)
res = np.log(P) - np.log(np.where(y > 0, y, np.nan))


def row(mask, label):
    e = np.nanmean(err[mask], 0); e0 = np.nanmean(err0[mask], 0); b = np.nanmean(res[mask], 0); sd = np.nanstd(res[mask], 0); ml = np.nanmean(np.where(y[mask] > 0, y[mask], np.nan), 0); sdl = np.nanstd(np.log(np.where(y[mask] > 0, y[mask], np.nan)), 0)
    print(label.ljust(12), f'n={mask.sum():5d} best {e[0]:6.3f} {e[1]:6.3f} {e[2]:6.3f} mean {e.mean():6.3f} | sub1 {e0.mean():6.3f} | bias {b[0]:+.3f} {b[1]:+.3f} {b[2]:+.3f} | res SD {sd[0]:.3f} {sd[1]:.3f} {sd[2]:.3f} | label {ml[0]:6.0f} {ml[1]:.3f} {ml[2]:.3f} log SD {sdl[0]:.3f} {sdl[1]:.3f} {sdl[2]:.3f}')


print('## by source and year, best pipeline'); row(np.ones(len(y), bool), 'all')
for v in sorted(set(yy[m_a])):
    row(m_a & (yy == v), f'A -{v}')
for v in sorted(set(yb[~m_a])):
    row(~m_a & (yb == v), f'B {v}')
print('\n## share of the score: excess error sum of -22 over all rows')
m22 = m_a & (yy == 22)
print('rows', m22.sum(), 'share', round(m22.mean(), 3), 'if -22 had the mean error of other A rows, mean moves by', np.round(np.nanmean((np.where(m22[:, None], np.nanmean(err[m_a & ~m22], 0)[None], err)), 0).mean() - np.nanmean(err, 0).mean(), 4))
print('\n## -22 by serial, 10 equal bins in serial order')
o = np.argsort(serial[m22]); rows22 = np.flatnonzero(m22)[o]
for c in np.array_split(rows22, 10):
    mk = np.zeros(len(y), bool); mk[c] = True
    extras = {k: round(float(np.median(S[c, names.index(k)])), 2) for k in ('levels', 'mean', 'frac0', 'p99', 'hp_std', 'center_over_corner', 'blk8h', 'lap_var')}
    row(mk, f'{serial[c].min()}-{serial[c].max()}'); print(' ' * 13, extras)
print('\n## frame statistics: Cohen d of -22 against other A rows and against A -21, -23; sorted by |d| against other A')
oth = m_a & ~m22
D = []
for j, n in enumerate(names):
    a, b, s = S[m22, j], S[oth, j], np.sqrt((S[m22, j].var() + S[oth, j].var()) / 2) + 1e-9
    D.append((abs(a.mean() - b.mean()) / s, n, (a.mean() - b.mean()) / s, (a.mean() - S[m_a & (yy == 21), j].mean()) / s, (a.mean() - S[m_a & (yy == 23), j].mean()) / s, np.median(a), np.median(b)))
for _, n, d, d21, d23, med, medo in sorted(D, reverse=True)[:18]:
    print(n.ljust(20), f'd {d:+.2f}  d vs 21 {d21:+.2f}  d vs 23 {d23:+.2f}  median {med:.3f} others {medo:.3f}')
print('\n## levels histogram for A by year: most common values')
for v in sorted(set(yy[m_a])):
    lv = S[m_a & (yy == v), names.index('levels')].astype(int); u, c = np.unique(lv, return_counts=True); ord_ = np.argsort(-c)[:6]
    print(f'-{v}', [(int(u[i]), int(c[i])) for i in ord_])
print('\n## error by the number of grey levels, A rows, level values with at least 100 rows')
lv = S[:, names.index('levels')].astype(int)
for u, c in zip(*np.unique(lv[m_a], return_counts=True)):
    if c >= 100:
        row(m_a & (lv == u), f'levels {u}'); print(' ' * 13, 'years', dict(zip(*[x.tolist() for x in np.unique(yy[m_a & (lv == u)], return_counts=True)])))
