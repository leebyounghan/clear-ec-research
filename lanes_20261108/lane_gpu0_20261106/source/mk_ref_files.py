"""Scale-1 files for rope_grid from the stored terminal predictions, and a check that stored and recomputed agree where both exist."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'experiments/lane_gpu0_20261106/source')
G = Path('experiments/lane_gpu0_20261106/rope_grid'); X = 'experiments/'
RECIPE = {46: {0: 'phase2_pooling_replication_20261026/f0_s46_std', 1: 'oof_recipeA_20261027/fold1/regression_std_s46', 2: 'oof_recipeA_20261027/fold2/regression_std_s46_v2', 3: 'oof_recipeA_20261027/fold3/regression_std_s46', 4: 'phase2_pooling_20261020/arms/std'},
          47: {0: 'phase2_pooling_replication_20261026/f0_s47_std', 4: 'phase2_pooling_replication_20261026/f4_s47_std', **{f: f'oof_recipeA_20261027/fold{f}/regression_std_s47' for f in (1, 2, 3)}},
          48: {f: f'oof_recipeA_20261027/fold{f}/regression_std_s48' for f in range(5)}}
for s, d in RECIPE.items():
    for f, rt in d.items():
        if not Path(X + rt + f'/runs/fold{f}/cpdino_vitb/predictions_epoch_20.npz').exists():
            print('no member', f, s); continue
        z = np.load(X + rt + f'/runs/fold{f}/cpdino_vitb/predictions_epoch_20.npz', allow_pickle=True)
        out = G / f'f{f}_s{s}_1.0.npz'
        if out.exists():
            q = np.load(out); print(f, s, 'max abs log difference, recomputed against stored', round(float(np.abs(np.log(q['pred'].astype(np.float64)) - np.log(z['pred'].astype(np.float64))).max()), 5))
        else:
            np.savez_compressed(out, indices=z['indices'].astype(int), pred=z['pred'].astype(np.float32))
print('done', len(list(G.glob('*_1.0.npz'))))
