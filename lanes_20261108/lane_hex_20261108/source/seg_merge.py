"""Merge seg_run.py chunks -> preds/<OUT>.npz (frames, F, names). usage: seg_merge.py CHUNKDIR(rel LANE) OUT"""
import sys, glob, numpy as np
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
fs = sorted(glob.glob(f'{LANE}/{sys.argv[1]}/c[0-9][0-9][0-9].npz')); fr, F = [], []
for p in fs:
    z = np.load(p); fr.append(z['frames']); F.append(z['F']); nm = z['names']
np.savez_compressed(f'{LANE}/preds/{sys.argv[2]}.npz', frames=np.concatenate(fr), F=np.concatenate(F), names=nm)
print('merged', len(fs), 'chunks', sum(len(q) for q in fr), 'frames')
