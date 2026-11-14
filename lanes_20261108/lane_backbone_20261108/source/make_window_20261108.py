"""Label-free window centres for the region view (2026-11-08): centre of the 538 x 408 box with the most pre-only detector peaks >= 0.5
(meta_ceiling_20261103/sweep/legible_window.npz, top_box_x / top_box_y; NaN where fewer than 10 confident peaks -> frame centre in the runner)."""
import numpy as np
from pathlib import Path
X = Path('/workspace/clear-ec-research/experiments'); z = np.load(X / 'meta_ceiling_20261103/sweep/legible_window.npz', allow_pickle=True)
F, names = z['F'], list(z['names']); c = np.stack([F[:, names.index('top_box_x')], F[:, names.index('top_box_y')]], 1).astype(np.float32)
np.savez_compressed(X / 'lane_backbone_20261108/data/window_topbox_20261108.npz', indices=np.arange(len(c)), Z=c)
print(c.shape, 'missing', int(np.isnan(c).any(1).sum()), 'x range', np.nanmin(c[:, 0]), np.nanmax(c[:, 0]), 'y range', np.nanmin(c[:, 1]), np.nanmax(c[:, 1]))
