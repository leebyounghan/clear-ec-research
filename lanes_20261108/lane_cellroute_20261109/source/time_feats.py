import sys, time
t0 = time.time(); sys.argv = ['x']; sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/source')
import feats_all as FA
import numpy as np
print('load', round(time.time() - t0, 1), 's', flush=True)
for r in (0, 100, 2000, 5000, 8000):
    t = time.time(); _, v = FA.one(r); v = np.array(v); print(r, round(time.time() - t, 2), 's', dict(zip(FA.COLS[:5], np.round(v[:5], 3))), dict(zip(FA.COLS[30:35], np.round(v[30:35], 3))), flush=True)
