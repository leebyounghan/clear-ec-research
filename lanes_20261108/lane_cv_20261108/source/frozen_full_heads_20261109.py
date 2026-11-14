"""CV lane (2026-11-09): full-data frozen token-statistic heads for the freeze candidate. Features: frozen_own_f99 (full-data recipe A seed-46 encoder, all rows).
Same head, loss and schedule as lane GPU 2's frozen_read_20261106.fit, trained on all good rows; saves each seed's head at epochs 16-20 (the OOF read averages
those 5 epochs' log predictions) with mu/sigma, to frozen_heads_full/<arm>_s<seed>_e<ep>.pt. Arms: std, spectrum, cov_sqrt, autocorr."""
import importlib.util, math, sys
from pathlib import Path
import numpy as np, torch
L2 = Path('/workspace/clear-ec-research/experiments/lane_gpu2_20261106'); ME = Path('/workspace/clear-ec-research/experiments/lane_cv_20261108')
spec = importlib.util.spec_from_file_location('FR', L2 / 'source/frozen_read_20261106.py'); FR = importlib.util.module_from_spec(spec); spec.loader.exec_module(FR)
FR.LANE = ME; FR.READS = ME / 'reads/own'; FR.OUT = ME / 'frozen_own_f99'; OUT = ME / 'frozen_heads_full'; OUT.mkdir(exist_ok=True)
y, good = FR.y, FR.good; tr = np.where(good & (y > 0).all(1))[0]
for arm in ('std', 'spectrum', 'cov_sqrt', 'autocorr'):
    Xa = FR.features(FR.ARMS[arm], tr); shapes = [(n, a.shape[1]) for n, a in Xa]
    ly = torch.from_numpy(np.log(y[tr])).float().cuda(); mu, sigma = ly.mean(0), ly.std(0, unbiased=False); zt = (ly - mu) / sigma; t_ = torch.from_numpy(tr).cuda(); Xt = [a[t_] for _, a in Xa]
    for seed in (0, 1, 2):
        torch.manual_seed(seed); head = FR.BlockHead(shapes, 128, 3).cuda(); torch.nn.init.normal_(head.net[-1].weight, std=1e-3); torch.nn.init.zeros_(head.net[-1].bias)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4, fused=True); updates = math.ceil(len(tr) / 16); step = 0
        for ep in range(20):
            order = torch.from_numpy(np.random.default_rng(seed * 1000 + ep).permutation(len(tr))).cuda()
            for i in range(0, len(tr), 16):
                b_ = order[i:i + 16]; r = FR.ratio(step, updates)
                for g in opt.param_groups: g['lr'] = 1e-3 * r
                loss = FR.recipe_loss(head([a[b_] for a in Xt]), zt[b_], sigma); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); step += 1
            if ep >= 15: torch.save(dict(model=head.state_dict(), blocks=FR.ARMS[arm], shapes=shapes, mu=mu.cpu(), sigma=sigma.cpu(), epoch=ep + 1, seed=seed, rows=len(tr)), OUT / f'{arm}_s{seed}_e{ep + 1}.pt')
        print(arm, 'seed', seed, 'saved', flush=True)
    del Xa, Xt; torch.cuda.empty_cache()
print('done', flush=True)
