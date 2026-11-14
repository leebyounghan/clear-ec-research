"""Segmentation TTA for the mask-density component: average log cd_mm2 of seg_loose and seg_rot (HEX lane, rotated 90 deg; read-only),
where both exist (>=50 interior cells in each), else seg_loose alone. Same blend protocol as cd_segfeat2 (calibrated, one weight, nested)."""
import os, glob, numpy as np
os.environ['A2SNAP'] = '1'
import cd_segfeat2_20261109 as S2
from cd_common import *
def load(d):
    F = np.full((len(yc), 17), np.nan); names = None
    for pth in sorted(glob.glob(str(X / f'lane_hex_20261108/{d}/c*.npz'))):
        z = np.load(pth); names = list(z['names'])
        for k, fr in enumerate(z['frames']):
            r = S2.pos.get(int(fr), -1)
            if r >= 0: F[r] = z['F'][k]
    return F, {n: i for i, n in enumerate(names)}
Fl, nm = load('seg_loose'); Fr, _ = load('seg_rot')
hl = Fl[:, nm['n_int']] >= 50; hr = Fr[:, nm['n_int']] >= 50
print('seg_rot frames', int(np.isfinite(Fr[:, 0]).sum()), 'both >=50', int((hl & hr).sum()), flush=True)
dl, dr = Fl[:, nm['cd_mm2']], Fr[:, nm['cd_mm2']]
okb = hl & hr & (yc > 0); ly = np.log(np.where(yc > 0, yc, np.nan))
print('corr log y: loose', round(float(np.corrcoef(np.log(dl[okb]), ly[okb])[0, 1]), 3), 'rot', round(float(np.corrcoef(np.log(dr[okb]), ly[okb])[0, 1]), 3), 'avg', round(float(np.corrcoef(0.5 * np.log(dl[okb]) + 0.5 * np.log(dr[okb]), ly[okb])[0, 1]), 3), flush=True)
avg = np.where(hl & hr, np.exp(0.5 * np.log(np.where(hl, dl, 1)) + 0.5 * np.log(np.where(hr, dr, 1))), dl)
S2.run('seg_loose + seg_rot averaged mask CD', avg, hl)
S2.run('seg_loose mask CD (reference)', dl, hl)
