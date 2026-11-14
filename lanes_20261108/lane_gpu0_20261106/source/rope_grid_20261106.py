"""Stored members under a fixed RoPE coordinate scale at inference (lane GPU 0, 2026-11-06). No flips, terminal epoch. One npz per member and scale; existing files are skipped, so it can be stopped and rerun.
Usage: rope_grid_20261106.py <folds, comma> <seeds, comma> <variants, comma; r or hH_wW>"""
import json, math, sys, time, types
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path[:0] = ['experiments/lane_gpu0_20261106/source', 'experiments/meta_ceiling_20261103/source', 'experiments/phase2_pooling_20261020/source']
import phase2_pooling_regression_20261019 as R
b = R.b
OUT = Path('experiments/lane_gpu0_20261106/rope_grid'); OUT.mkdir(exist_ok=True)
X = 'experiments/'
RECIPE = {46: {0: 'phase2_pooling_replication_20261026/f0_s46_std', 1: 'oof_recipeA_20261027/fold1/regression_std_s46', 2: 'oof_recipeA_20261027/fold2/regression_std_s46_v2', 3: 'oof_recipeA_20261027/fold3/regression_std_s46', 4: 'phase2_pooling_20261020/arms/std'},
          47: {0: 'phase2_pooling_replication_20261026/f0_s47_std', 4: 'phase2_pooling_replication_20261026/f4_s47_std', **{f: f'oof_recipeA_20261027/fold{f}/regression_std_s47' for f in (1, 2, 3)}},
          48: {f: f'oof_recipeA_20261027/fold{f}/regression_std_s48' for f in range(5)}}
R.ARGS = types.SimpleNamespace(pooling='std', seed=46)
with np.load(b.META, allow_pickle=True) as z:
    ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)


def parse(v):
    if v.startswith('h'):
        h, w = v[1:].split('_w')
        return float(h), float(w)
    return float(v), float(v)


def patch(rope, rh, rw):
    def forward(*, H, W):
        dd = {'device': rope.periods.device, 'dtype': rope.dtype}
        ch = torch.arange(0.5, H, **dd) / H
        cw = torch.arange(0.5, W, **dd) / W
        coords = torch.stack(torch.meshgrid(ch, cw, indexing='ij'), dim=-1).flatten(0, 1)
        coords = (2.0 * coords - 1.0) * torch.tensor([rh, rw], **dd)
        angles = 2 * math.pi * coords[:, :, None] / rope.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return torch.sin(angles), torch.cos(angles)
    rope.forward = forward


folds = [int(v) for v in sys.argv[1].split(',')]; seeds = [int(v) for v in sys.argv[2].split(',')]; variants = sys.argv[3].split(',')
for seed in seeds:
    for f in folds:
        todo = [v for v in variants if not (OUT / f'f{f}_s{seed}_{v}.npz').exists()]
        if not todo:
            continue
        rt = X + RECIPE[seed][f]
        m = json.load(open(rt + '/manifest.json'))
        net, _ = R.build_cpu(seed)
        net.load_state_dict(torch.load(rt + f'/runs/fold{f}/cpdino_vitb/epoch_20.pt', map_location='cpu', weights_only=True)['model'])
        net = net.cuda().eval()
        rope = net[0].original.rope_embed
        mu = torch.tensor(m['log_mean']).cuda(); sg = torch.tensor(m['log_std']).cuda()
        va = np.array(m['validation_indices'])
        dl = DataLoader(R.Frames(va, y, len(ids), 0, False, seed), batch_size=16, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2)
        for v in todo:
            t = time.time(); patch(rope, *parse(v)); out = []
            with torch.inference_mode():
                for x, _, ix in dl:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        z = net(b.inputs(x, R.ARCH)).float()
                    out.append(torch.exp(z * sg + mu).cpu().numpy())
            np.savez_compressed(OUT / f'f{f}_s{seed}_{v}.npz', indices=va, pred=np.concatenate(out).astype(np.float32))
            print(f'fold {f} seed {seed} variant {v} done {time.time() - t:.0f}s', flush=True)
        del net
        torch.cuda.empty_cache()
print('grid finished', flush=True)
