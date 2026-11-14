"""Label-free auxiliary targets for the backbone lane (2026-11-08): detector-window cell statistics from
meta_ceiling_20261103/sweep/legible_window.npz (pre-only peaks, rows = cache indices). Writes data/aux_legible15_20261108.npz."""
import numpy as np
from pathlib import Path
X = Path('/workspace/clear-ec-research/experiments'); out = X / 'lane_backbone_20261108/data'; out.mkdir(exist_ok=True)
z = np.load(X / 'meta_ceiling_20261103/sweep/legible_window.npz', allow_pickle=True); F, names = z['F'].astype(np.float64), list(z['names'])
col = lambda n: F[:, names.index(n)]
cols, use = [], []
for r in ('whole', 'top_box', 'top3_box'):
    for s in ('hex6', 'cv_area', 'cv_dist', 'cd'):
        v = col(f'{r}_{s}'); cols.append(np.log(np.clip(v, 1, None)) if s == 'cd' else v); use.append(f'{r}_{s}' + ('_log' if s == 'cd' else ''))
cols += [np.log(np.clip(col('whole_n'), 1, None)), col('top_box_x') / 1296, col('top_box_y') / 972]; use += ['whole_n_log', 'top_box_x_frac', 'top_box_y_frac']
Z = np.stack(cols, 1).astype(np.float32)
np.savez_compressed(out / 'aux_legible15_20261108.npz', indices=np.arange(len(Z)), Z=Z, names=np.array(use))
print(Z.shape, use, np.isnan(Z).mean(0).round(3).tolist(), np.nanmean(Z, 0).round(3).tolist())
