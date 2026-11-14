"""Row weights from a teacher table, same formula as make_nested_teacher_20261109.py (2026-11-09): d = mean_k |log y - log t| / median_k,
w = 1 up to the 90th percentile of d, then (q90 / d)^2, floor 0.2. Usage: rowweight_from_teacher_20261109.py <teacher.npz> <out.npz>"""
import sys, json
import numpy as np
with np.load('/workspace/clear-ec-research/datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True) as z:
    Y = z['Y'].astype(np.float64)
with np.load(sys.argv[1]) as z:
    ix, T = z['indices'].astype(int), z['pred'].astype(np.float64)
assert np.isfinite(T).all() and (T > 0).all() and (Y[ix] > 0).all()
dk = np.abs(np.log(Y[ix]) - np.log(T)); d = (dk / np.median(dk, 0)).mean(1); q = np.quantile(d, 0.9)
w = np.where(d <= q, 1.0, np.maximum(0.2, (q / d) ** 2))
np.savez_compressed(sys.argv[2], indices=ix, Z=w.astype(np.float32))
print(json.dumps(dict(teacher=sys.argv[1], rows=len(ix), teacher_mape=(np.abs(T / Y[ix] - 1) * 100).mean(0).round(3).tolist(), weight_mean=round(float(w.mean()), 4), share_below_1=round(float((w < 1).mean()), 4), share_at_floor=round(float((w <= 0.2001).mean()), 4))))
