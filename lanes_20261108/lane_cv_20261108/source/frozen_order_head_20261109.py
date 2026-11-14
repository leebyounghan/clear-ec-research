"""CV lane (2026-11-09): frozen token-statistic heads with a noise-weighted pairwise order loss on CV (KG idea-cv-noise-weighted-order-loss, lee2026stochorder).
Recipe loss on all three columns plus LAMBDA x pairwise BCE on the CV column: for pairs (i, j) in a batch, target p_ij = Phi((log y_i - log y_j) / (sqrt2 * s)),
logit = (log yhat_i - log yhat_j) / (sqrt2 * s), with s = 0.065 (sampling SE of CV from ~150 cells, log units). Near-tied pairs carry little gradient, far pairs
enforce order. Same features (own encoders), head and schedule as lane GPU 2's frozen read. Writes reads/own/<prefix>_<arm>_f<f>.npz in this lane.
usage: ORDER_LAMBDA=1 frozen_order_head_20261109.py PREFIX arm[,arm...] [--folds 0,4]"""
import importlib.util, math, os, sys
from pathlib import Path
import torch
L2 = Path('/workspace/clear-ec-research/experiments/lane_gpu2_20261106'); ME = Path('/workspace/clear-ec-research/experiments/lane_cv_20261108')
spec = importlib.util.spec_from_file_location('FR', L2 / 'source/frozen_read_20261106.py'); FR = importlib.util.module_from_spec(spec); spec.loader.exec_module(FR)
FR.LANE = ME; FR.READS = ME / 'reads/own'; FR.READS.mkdir(parents=True, exist_ok=True)
LAM = float(os.environ.get('ORDER_LAMBDA', '1')); SN = 0.065; _base = FR.recipe_loss
def loss_order(z, zt, sigma):
    base = _base(z, zt, sigma); s = sigma[1]; dp = (z[:, 1][:, None] - z[:, 1][None, :]) * s; dt = (zt[:, 1][:, None] - zt[:, 1][None, :]) * s
    p = 0.5 * (1 + torch.erf(dt / (2 * SN))); lg = dp / (math.sqrt(2) * SN); m = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
    return base + LAM * torch.nn.functional.binary_cross_entropy_with_logits(lg[m], p[m])
FR.recipe_loss = loss_order
prefix, arms = sys.argv[1], sys.argv[2].split(','); folds = [int(v) for v in (sys.argv[sys.argv.index('--folds') + 1] if '--folds' in sys.argv else '0,1,2,3,4').split(',')]
print('order lambda', LAM, 'folds', folds, flush=True)
for arm in arms:
    for f in folds:
        FR.OUT = (L2 if f in (0, 4) else ME) / f'frozen_own_f{f}'; FR.read_arm(f'{prefix}_{arm}', FR.ARMS[arm], [f], [0, 1, 2])
print('done', flush=True)
