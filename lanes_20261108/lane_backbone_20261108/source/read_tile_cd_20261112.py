"""Tile CD arm read (D-CDSIMPLE-001 (b)): runs/tile_cd_f{f}_s46 (raw) vs the plain DINOv3-B s46 (lane_gpu3 for folds 0/4) on the CD
column only, same rows; member read = mean of the per-epoch APE over epochs 8-12; also epoch 12 alone. Usage: read_tile_cd_20261112.py F[,F]"""
import json, sys
from pathlib import Path
import numpy as np
X = Path('/workspace/clear-ec-research/experiments')
def load(run, e):
    with np.load(run / f'predictions_epoch_{e:02d}.npz', allow_pickle=True) as z:
        return z['indices'].astype(int), z['pred'][:, 0].astype(float), z['target'][:, 0].astype(float)
out = {}
for f in [int(v) for v in sys.argv[1].split(',')]:
    tile = X / f'lane_backbone_20261108/runs/tile_cd_f{f}_s46/raw/runs/fold{f}/dinov3_b16'
    plain = X / f'lane_gpu3_20261106/runs/dinov3_b16_f{f}_s46/runs/fold{f}/dinov3_b16'
    r = {}
    for nm, run in (('tile', tile), ('plain', plain)):
        A = []; ix0 = None
        for e in range(8, 13):
            ix, p, y = load(run, e); ix0 = ix if ix0 is None else ix0; assert (ix == ix0).all(); A.append(np.abs(p / y - 1) * 100)
        r[nm] = dict(cd_member_e8_12=round(float(np.mean(A)), 4), cd_e12=round(float(A[-1].mean()), 4), n=int(len(ix0)))
    r['delta_member'] = round(r['tile']['cd_member_e8_12'] - r['plain']['cd_member_e8_12'], 4)
    out[f] = r
print(json.dumps(out))
