import sys; sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source')
import hexlib as L, numpy as np
BP = L.BP; b = BP._b; y, fold, good = L.y, L.fold, L.good; A, P1, B = b['A'], b['P1'], b['B']
N_ = BP.N
def st(P):
    Q = np.full_like(P, np.nan)
    for f in range(5): Q[fold == f] = N_.apply_stretch(P[fold == f], N_.fit_stretch(P, y, (fold != f) & good))
    return Q
allr = np.ones(L.N, bool)
for nm, P in (('A', A), ('P1', P1), ('B', B)):
    print(nm, 'raw HEX %.3f' % L.hexape(P, allr), 'stretched %.3f' % L.hexape(st(P), allr), 'CD %.3f CV %.3f' % tuple(np.nanmean(np.abs(st(P)[:, j] - y[:, j]) / y[:, j]) * 100 for j in (0, 1)))
ok = y[:, 2] > 0
print('r(logA,logP1) HEX', np.corrcoef(np.log(A[ok, 2]), np.log(P1[ok, 2]))[0, 1], 'err r', np.corrcoef(np.log(A[ok, 2] / y[ok, 2]), np.log(P1[ok, 2] / y[ok, 2]))[0, 1])
# pipeline with HEX base from A only: replace H HEX by A-based hex6 blend
for nm, base in (('A', A), ('P1', P1)):
    H2 = BP.H.copy(); h = BP._have; H2[:, 2] = base[:, 2]; H2[h, 2] = np.exp((1 - 0.28) * np.log(base[h, 2]) + 0.28 * (-0.2084 + 0.5989 * BP.f6[h]))
    P = BP.pipe(BP.BEST, base=H2); print('best pipeline with HEX base', nm, '%.3f' % L.hexape(P, allr), 'f04 %.3f' % L.hexape(P, L.S04))
