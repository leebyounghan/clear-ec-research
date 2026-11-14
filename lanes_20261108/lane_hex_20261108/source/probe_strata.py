import sys; sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source')
import hexlib as L, numpy as np
y, P = L.y, L.REF; ok = y[:, 2] > 0; ape = np.abs(P[:, 2] - y[:, 2]) / np.where(ok, y[:, 2], 1) * 100
lr = np.log(P[:, 2] / np.where(ok, y[:, 2], 1))
for nm, v in (('pred CD', P[:, 0]), ('pred HEX', P[:, 2]), ('label HEX', y[:, 2]), ('hex6 feat', L.BP.f6)):
    q = np.nanpercentile(v[ok], [33.3, 66.7]); t = np.digitize(v, q)
    print(nm, [(round(float(np.nanmean(v[ok & (t == k)])), 3), round(float(ape[ok & (t == k)].mean()), 3), round(float(np.median(lr[ok & (t == k)])), 4), int((ok & (t == k)).sum())) for k in range(3)])
# label deciles: error and bias
q = np.percentile(y[ok, 2], np.arange(10, 100, 10)); t = np.digitize(y[:, 2], q)
print('label deciles APE', [round(float(ape[ok & (t == k)].mean()), 2) for k in range(10)])
print('label deciles median log ratio', [round(float(np.median(lr[ok & (t == k)])), 3) for k in range(10)])
q = np.percentile(P[ok, 2], np.arange(10, 100, 10)); t = np.digitize(P[:, 2], q)
print('pred deciles APE', [round(float(ape[ok & (t == k)].mean()), 2) for k in range(10)])
print('pred deciles median log ratio', [round(float(np.median(lr[ok & (t == k)])), 3) for k in range(10)])
print('r(log pred, log y)', np.corrcoef(np.log(P[ok, 2]), np.log(y[ok, 2]))[0, 1], 'sd ratio', np.log(P[ok, 2]).std() / np.log(y[ok, 2]).std())
print('HEX label values distinct', len(np.unique(y[ok, 2])), 'min max', y[ok, 2].min(), y[ok, 2].max(), 'zeros', int((~ok).sum()))
