"""CV lane (2026-11-08): frozen token-statistic heads with a CV-weighted loss. Same features, head, schedule and read as lane GPU 2's frozen_read_20261106
(own encoder per fold), only the recipe loss is weighted per column by LOSS_W (CD,CV,HEX; normalised to mean 1). Features: folds 0 and 4 from lane GPU 2's
frozen_own_f{0,4} (read only), folds 1-3 from this lane. Writes reads/own/<prefix>_<arm>_f<f>.npz in this lane.
FEAT=dino reads this lane's frozen_dino_f<f> (DINOv3 encoders) instead.
usage: LOSS_W=0.25,2.5,0.25 frozen_cv_head_20261108.py PREFIX arm[,arm...] [--folds 0,1,2,3,4]"""
import importlib.util, os, sys
from pathlib import Path
import torch
L2 = Path('/workspace/clear-ec-research/experiments/lane_gpu2_20261106'); ME = Path('/workspace/clear-ec-research/experiments/lane_cv_20261108')
spec = importlib.util.spec_from_file_location('FR', L2 / 'source/frozen_read_20261106.py'); FR = importlib.util.module_from_spec(spec); spec.loader.exec_module(FR)
FR.LANE = ME; FR.READS = ME / 'reads/own'; FR.READS.mkdir(parents=True, exist_ok=True)
W = [float(v) for v in os.environ['LOSS_W'].split(',')]; W = [3 * v / sum(W) for v in W]; Wt = None
def recipe_loss_w(z, zt, sigma):
    global Wt
    if Wt is None: Wt = torch.tensor(W, device=z.device, dtype=z.dtype)
    return ((0.5 * (z - zt).abs() + 2.5 * torch.expm1((z - zt) * sigma).abs()) * Wt).mean()
FR.recipe_loss = recipe_loss_w
prefix, arms = sys.argv[1], sys.argv[2].split(','); folds = [int(v) for v in (sys.argv[sys.argv.index('--folds') + 1] if '--folds' in sys.argv else '0,1,2,3,4').split(',')]
print('LOSS_W normalised', W, 'folds', folds, flush=True)
for arm in arms:
    for f in folds:
        FR.OUT = (ME / f'frozen_dino_f{f}') if os.environ.get('FEAT') == 'dino' else ((L2 if f in (0, 4) else ME) / f'frozen_own_f{f}'); FR.read_arm(f'{prefix}_{arm}', FR.ARMS[arm], [f], [0, 1, 2])
print('done', flush=True)
