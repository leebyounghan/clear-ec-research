import sys, re; sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source')
import hexlib as L, numpy as np
y, P, ids, N = L.y, L.REF, L.BP.ids, L.N; allr = np.ones(N, bool); ok = y[:, 2] > 0
for s in (0.97, 0.98, 0.99, 1.0, 1.01, 1.02):
    Q = P.copy(); Q[:, 2] *= s; print('scale', s, 'HEX %.4f' % L.hexape(Q, allr), 'sub1-B scaled %.4f' % L.hexape(np.c_[L.BP.REF_SUB1[:, :2], L.BP.REF_SUB1[:, 2] * s], allr))
src = np.array(['B' if re.match(r'^\d{4}-\d{4}', i) else 'A' for i in ids])
yr = np.array([int(i[:4]) if s_ == 'B' else 2000 + int(re.findall(r'-(\d\d)', i)[0]) if re.findall(r'-(\d\d)', i) else 0 for i, s_ in zip(ids, src)])
for nm, PP in (('sub1', L.BP.REF_SUB1), ('best', P)):
    e = np.abs(PP[:, 2] - y[:, 2]) / np.where(ok, y[:, 2], 1) * 100
    print(nm, 'by source', {s_: round(float(e[ok & (src == s_)].mean()), 3) for s_ in 'AB'}, 'by year', {int(v): (round(float(e[ok & (yr == v)].mean()), 2), int((ok & (yr == v)).sum())) for v in np.unique(yr)})
lr = np.log(P[ok, 2] / y[ok, 2]); print('residual log SD %.4f, median %.4f' % (lr.std(), np.median(lr)))
