"""RoPE coordinate scale at inference (lane GPU 0, 2026-11-06). The encoder trains with a random coordinate rescale in [1/2, 2] at every forward and validates at 1. Does the prediction depend on the scale, and does averaging over scales help?
Reference members, seed 46, folds 0 and 4, terminal epoch. All validation rows, no flips. Multiplying coordinates by r is the same as dividing the RoPE periods by r. Inference only; stores predictions."""
import json, sys, types
from pathlib import Path
import numpy as np, torch
sys.path[:0] = ['experiments/lane_gpu0_20261106/source', 'experiments/meta_ceiling_20261103/source', 'experiments/phase2_pooling_20261020/source', 'experiments/eval_harness_20261031/source']
import phase2_pooling_regression_20261019 as R
import eval_oof_20261031 as E
from torch.utils.data import DataLoader
b = R.b
ROOTS = {0: 'experiments/phase2_pooling_replication_20261026/f0_s46_std', 4: 'experiments/phase2_pooling_20261020/arms/std'}
SCALES = [1.0, 2 ** -0.5, 2 ** 0.5]
OUT = Path('experiments/lane_gpu0_20261106/rope_tta'); OUT.mkdir(exist_ok=True)
R.ARGS = types.SimpleNamespace(pooling='std', seed=46)
base = E.load_base(); gp = E.donors(base['ids']); pos = {int(i): n for n, i in enumerate(base['idx'])}
with np.load(b.META, allow_pickle=True) as z:
    ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
ALL = []
for f, rt in ROOTS.items():
    m = json.load(open(rt + '/manifest.json'))
    net, _ = R.build_cpu(46)
    net.load_state_dict(torch.load(rt + f'/runs/fold{f}/cpdino_vitb/epoch_20.pt', map_location='cpu', weights_only=True)['model'])
    net = net.cuda().eval()
    rope = net[0].original.rope_embed
    per0 = rope.periods.data.clone()
    mu = torch.tensor(m['log_mean']).cuda(); sg = torch.tensor(m['log_std']).cuda()
    va = np.array(m['validation_indices'])
    dl = DataLoader(R.Frames(va, y, len(ids), 0, False, 46), batch_size=8, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2)
    stored = np.load(rt + f'/runs/fold{f}/cpdino_vitb/predictions_epoch_20.npz', allow_pickle=True)
    P = {}
    for r in SCALES:
        rope.periods.data = per0 / r
        out = []
        with torch.inference_mode():
            for x, t, ix in dl:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z = net(b.inputs(x, R.ARCH)).float()
                out.append(torch.exp(z * sg + mu).cpu().numpy())
        P[r] = np.concatenate(out).astype(np.float64)
        e = E.official(E.ape(P[r], stored['target'].astype(np.float64)))
        print(f'fold {f} scale {r:.3f}', {k: round(e[k], 3) for k in ('CD', 'CV', 'HEX', 'mean')}, 'median log change vs scale 1 %', np.round(np.median(np.log(P[r]) - np.log(P[1.0]), 0) * 100, 2).tolist(), flush=True)
    assert np.abs(np.log(P[1.0]) - np.log(stored['pred'].astype(np.float64))).max() < 0.02, 'scale 1 must reproduce the stored epoch 20'
    np.savez_compressed(OUT / f'f{f}_s46.npz', indices=va, target=stored['target'], **{f'scale_{r:.3f}': P[r].astype(np.float32) for r in SCALES})
    ALL.append((va, stored['target'].astype(np.float64), P))
    del net
    torch.cuda.empty_cache()
yy = np.concatenate([a[1] for a in ALL]); g = np.concatenate([gp[[pos[int(i)] for i in a[0]]] for a in ALL])
p1 = np.concatenate([a[2][1.0] for a in ALL])
e1 = E.ape(p1, yy)
for name, rs in (('average of 0.707, 1, 1.414', SCALES), ('scale 0.707 alone', [SCALES[1]]), ('scale 1.414 alone', [SCALES[2]])):
    pp = np.exp(np.mean([np.log(np.concatenate([a[2][r] for a in ALL])) for r in rs], 0))
    pb = E.paired_bootstrap(E.ape(pp, yy), e1, g); s, s0 = E.official(E.ape(pp, yy)), E.official(e1)
    print(name, 'mean', round(s['mean'], 4), 'against', round(s0['mean'], 4), 'delta', round(pb['delta'], 4), 'CI', np.round(pb['ci95'], 4).tolist(), {k: round(s[k] - s0[k], 4) for k in ('CD', 'CV', 'HEX')}, flush=True)
