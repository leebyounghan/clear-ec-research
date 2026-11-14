"""HEX-weighted family recipe (HEX lane, 2026-11-08). Runs the backbone lane's family recipe (frozen copy in this folder,
source/family_recipe_copy.py) unchanged except the regression loss: the same per-target loss (0.5 |z - zt| + 2.5 |expm1|),
weighted 0.5 / 0.5 / 2 over CD / CV / HEX (mean weight 1) instead of the plain mean. Same arguments as the recipe; run from
/workspace/clear-ec-research. usage: hexcps.py all --family cpsam_c512 --fold 4 --root <lane>/runs/hexcps_f4 --ssl-from <ssl last.pt> --epochs 12"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import family_recipe_copy as FR
W = torch.tensor([float(q) for q in os.environ.get('HEX_LW', '0.5,0.5,2.0').split(',')])


def loss_w(z, targets, mu, sigma):
    zt = (targets.log() - mu) / sigma; d = z.float() - zt
    per = 0.5 * d.abs().mean(0) + 2.5 * torch.expm1(d * sigma).abs().mean(0)
    return (per * W.to(per.device)).mean()


FR.b.loss_fn = loss_w
# folds 1-3 (validation sets checked equal to the chain folds, 2026-11-08)
FR.SPLITS.update({1: 'experiments/ssl_screen_20260911/preflight/fold1.npz', 2: 'experiments/overnight_target_20260912/preflight/fold2.npz',
                  3: 'experiments/overnight_target_20260912/preflight/fold3.npz'})
if __name__ == '__main__':
    print('hexcps loss weights', W.tolist(), flush=True)
    FR.main()
