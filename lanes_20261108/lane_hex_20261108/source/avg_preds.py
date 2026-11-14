"""Average graph-net prediction files (log-residual outputs f0..f4) into one, e.g. two 5-seed runs -> a 10-seed run.
usage: avg_preds.py OUT NAME1 NAME2 [...]  (names in lane preds/, without .npz)"""
import sys, numpy as np
P = '/workspace/clear-ec-research/experiments/lane_hex_20261108/preds'
zs = [np.load(f'{P}/{n}.npz') for n in sys.argv[2:]]
np.savez_compressed(f'{P}/{sys.argv[1]}.npz', **{f'f{f}': np.mean([z[f'f{f}'] for z in zs], 0) for f in range(5)})
print('wrote', sys.argv[1], 'from', sys.argv[2:])
