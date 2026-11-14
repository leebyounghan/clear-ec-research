"""Verify the full-data attention-pool DINOv3-B (fold 99, seed 46): sha256 of the deploy files, strict load of each checkpoint into the
runner's own network (recipe A net with the AttnPool head, built exactly as bb_runner_attnpool_20261111.py does), and recompute the
stored predictions on the held rows (the two HEX-zero rows) through the runner's own evaluate(): epoch_08..12.pt single view vs
predictions_epoch_08..11.npz and final.pt single view vs predictions_epoch_12.npz (raw track), final.pt with four-flip TTA vs predictions_final_tta.npz for every track.
Mirrors lane_cd_20261108/source/cd_verify_f99_vitl_20261111.py. Run from /workspace/clear-ec-research with one visible GPU."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
sys.path.insert(0, 'experiments/lane_backbone_20261108/source')
import bb_runner_attnpool_20261111 as B  # noqa: E402
F, R = B.F, B.R
B.FAMILY = 'dinov3_b16'
ROOT = Path('experiments/lane_backbone_20261108/runs/attnpool_f99_dinov3b16_s46')
SSL = 'experiments/lane_cd_20261108/arms/dinov3_b16_f99_s46/ssl/last.pt'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def build(seed=46):
    net = F.build_net(argparse.Namespace(family='dinov3_b16'), seed, SSL)
    dim = net[1].normalized_shape[0]; torch.manual_seed(seed + 2)
    net[0] = B.AttnPool(net[0].tokens, dim).cuda(); net[1] = nn.LayerNorm(4 * dim).cuda(); net[2] = nn.Linear(4 * dim, 128).cuda()
    return net


ids, y, tr, va = B.arrays(99); n = len(ids)
R.ARGS = argparse.Namespace(seed=46)
net = build()
dl = DataLoader(R.Frames(va, y, n, 0, False, 46), batch_size=8, shuffle=False, num_workers=1)
out = dict(held_rows=va.tolist(), held_ids=ids[va].tolist(), files={})
checks = [('raw', f'epoch_{e:02d}.pt', f'predictions_epoch_{e:02d}.npz', False) for e in (8, 9, 10, 11)]
checks += [('raw', 'final.pt', 'predictions_epoch_12.npz', False)]  # the runner saves epoch 12 as final.pt (no epoch_12.pt)
checks += [(t, 'final.pt', 'predictions_final_tta.npz', True) for t in ('raw', 'ema998', 'ema999', 'swa6')]
for track, name, stored, tta in checks:
    d = ROOT / track / 'runs' / 'fold99' / 'dinov3_b16'
    ck = torch.load(d / name, map_location='cuda', weights_only=False)
    net.load_state_dict(ck['model'], strict=True)
    mu = torch.tensor(ck['log_mean'], device='cuda', dtype=torch.float32); sigma = torch.tensor(ck['log_std'], device='cuda', dtype=torch.float32)
    p = B.evaluate(net, dl, mu, sigma, va, tta=tta)
    with np.load(d / stored) as z:
        assert (z['indices'] == va).all()
        q = z['pred'].astype(np.float64)
    rec = dict(sha256=sha(d / name), bytes=(d / name).stat().st_size, epoch=int(ck['epoch']), strict_load=True,
               max_rel_diff_vs_stored=float(np.abs(p / q - 1).max()), recomputed=np.round(p, 4).tolist(), stored=np.round(q, 4).tolist())
    key = f'{track}/{name}' + ('/tta' if tta else '/single')
    out['files'][key] = rec
    print(json.dumps({key: {k: rec[k] for k in ('sha256', 'epoch', 'max_rel_diff_vs_stored')}}), flush=True)
# sanity read on training frames: f99 raw log mean of epochs 8-12 (single view, the HEX lane's read) vs the attention-pool OOF members
# (each row from the fold run that held it out, same read) on 256 fixed training rows; in-sample for f99, out-of-fold for the members
rows = np.sort(np.random.default_rng(1111).choice(tr, 256, replace=False))
dls = DataLoader(R.Frames(rows, y, n, 0, False, 46), batch_size=8, shuffle=False, num_workers=2)
d99 = ROOT / 'raw' / 'runs' / 'fold99' / 'dinov3_b16'
logs = []
for name in [f'epoch_{e:02d}.pt' for e in (8, 9, 10, 11)] + ['final.pt']:
    ck = torch.load(d99 / name, map_location='cuda', weights_only=False); net.load_state_dict(ck['model'], strict=True)
    mu = torch.tensor(ck['log_mean'], device='cuda', dtype=torch.float32); sigma = torch.tensor(ck['log_std'], device='cuda', dtype=torch.float32)
    logs.append(np.log(B.evaluate(net, dls, mu, sigma, rows, tta=False)))
f99 = np.mean(logs, 0)
oof = np.full((len(rows), 3), np.nan)
for f in range(5):
    dd = Path(f'experiments/lane_backbone_20261108/runs/attnpool_f{f}_dinov3b16_s46/raw/runs/fold{f}/dinov3_b16')
    P = []
    for e in range(8, 13):
        with np.load(dd / f'predictions_epoch_{e:02d}.npz') as z:
            P.append((z['indices'].astype(int), np.log(z['pred'].astype(np.float64))))
    ix = P[0][0]; lp = np.mean([p[1] for p in P], 0); pos = {v: j for j, v in enumerate(ix)}
    for j, r_ in enumerate(rows):
        if r_ in pos:
            oof[j] = lp[pos[r_]]
ok = np.isfinite(oof).all(1); ly = np.log(y[rows])
san = {}
for k, nm in enumerate(('CD', 'CV', 'HEX')):
    a, b_, t = f99[ok, k], oof[ok, k], ly[ok, k]
    san[nm] = dict(r_f99_vs_oof=round(float(np.corrcoef(a, b_)[0, 1]), 4), mean_log_ratio_f99_over_oof=round(float((a - b_).mean()), 4),
                   r_f99_vs_label=round(float(np.corrcoef(a, t)[0, 1]), 4), r_oof_vs_label=round(float(np.corrcoef(b_, t)[0, 1]), 4))
out['sanity_training_rows'] = dict(n=int(ok.sum()), rows_seed=1111, read='raw, mean log over epochs 8-12, single view', **san)
print(json.dumps({'sanity': out['sanity_training_rows']}), flush=True)
Path('experiments/lane_backbone_20261108/reads/verify_attnpool_f99_20261111.json').write_text(json.dumps(out, indent=1))
print('saved reads/verify_attnpool_f99_20261111.json')
