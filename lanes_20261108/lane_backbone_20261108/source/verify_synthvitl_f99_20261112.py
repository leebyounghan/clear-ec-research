"""Verify the full-data synthetic-pretrained ViT-L (synthft_vitl, fold 99, seed 46; D-SYN-005). Copy of verify_synthft3_f99_20261111.py
with the ViT-L family and paths. Checks, as for the plain ViT-L f99 builds (lane_cd_20261108/source/cd_verify_f99_vitl_20261111.py):
sha256 of every deploy file, strict load of each checkpoint into the runner's own network (recipe A network, family dinov3_l16t,
built as bb_runner_attnpool_20261111.py does without --attn-pool), and recompute of the stored predictions on the held rows (the two
HEX-zero rows) through the runner's own evaluate(): epoch_08..11.pt single view vs predictions_epoch_08..11.npz, final.pt single view
vs predictions_epoch_12.npz (raw), final.pt four-flip TTA vs predictions_final_tta.npz for every track.
Sanity read on 256 fixed training rows (raw, log mean of epochs 8-12, single view): f99 vs (a) the synthetic ViT-L OOF members (each row
from the fold run that held it out) and (b) the plain ViT-L f99 s46 build (lane_cd_20261108/arms/vitl16_f99_s46), both in-sample for f99.
Run from /workspace/clear-ec-research with one visible GPU."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, 'experiments/lane_backbone_20261108/source')
import bb_runner_attnpool_20261111 as B  # noqa: E402
F, R = B.F, B.R
FAM = 'dinov3_l16t'
B.FAMILY = FAM
ROOT = Path('experiments/lane_backbone_20261108/runs/synthft_vitl_f99_s46')
SSL = 'experiments/lane_backbone_20261108/runs/synthpre_vitl_f99_s46/pretrain.pt'
PLAIN = Path('experiments/lane_cd_20261108/arms/vitl16_f99_s46/raw/runs/fold99') / FAM


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def load(net, path):
    ck = torch.load(path, map_location='cuda', weights_only=False)
    net.load_state_dict(ck['model'], strict=True)
    mu = torch.tensor(ck['log_mean'], device='cuda', dtype=torch.float32); sigma = torch.tensor(ck['log_std'], device='cuda', dtype=torch.float32)
    return ck, mu, sigma


ids, y, tr, va = B.arrays(99); n = len(ids)
R.ARGS = argparse.Namespace(seed=46)
net = F.build_net(argparse.Namespace(family=FAM), 46, SSL)
dl = DataLoader(R.Frames(va, y, n, 0, False, 46), batch_size=8, shuffle=False, num_workers=1)
out = dict(held_rows=va.tolist(), held_ids=ids[va].tolist(), files={})
checks = [('raw', f'epoch_{e:02d}.pt', f'predictions_epoch_{e:02d}.npz', False) for e in (8, 9, 10, 11)]
checks += [('raw', 'final.pt', 'predictions_epoch_12.npz', False)]  # epoch 12 is saved as final.pt (no epoch_12.pt)
checks += [(t, 'final.pt', 'predictions_final_tta.npz', True) for t in ('raw', 'ema998', 'ema999', 'swa6')]
for track, name, stored, tta in checks:
    d = ROOT / track / 'runs' / 'fold99' / FAM
    ck, mu, sigma = load(net, d / name)
    p = B.evaluate(net, dl, mu, sigma, va, tta=tta)
    with np.load(d / stored) as z:
        assert (z['indices'] == va).all()
        q = z['pred'].astype(np.float64)
    rec = dict(sha256=sha(d / name), bytes=(d / name).stat().st_size, epoch=int(ck['epoch']), strict_load=True,
               max_rel_diff_vs_stored=float(np.abs(p / q - 1).max()), recomputed=np.round(p, 4).tolist(), stored=np.round(q, 4).tolist())
    key = f'{track}/{name}' + ('/tta' if tta else '/single')
    out['files'][key] = rec
    print(json.dumps({key: {k: rec[k] for k in ('sha256', 'epoch', 'max_rel_diff_vs_stored')}}), flush=True)
rows = np.sort(np.random.default_rng(1111).choice(tr, 256, replace=False))
dls = DataLoader(R.Frames(rows, y, n, 0, False, 46), batch_size=8, shuffle=False, num_workers=2)
names = [f'epoch_{e:02d}.pt' for e in (8, 9, 10, 11)] + ['final.pt']


def logmean(d):
    L = []
    for name in names:
        _, mu, sigma = load(net, d / name)
        L.append(np.log(B.evaluate(net, dls, mu, sigma, rows, tta=False)))
    return np.mean(L, 0)


f99 = logmean(ROOT / 'raw' / 'runs' / 'fold99' / FAM)
plain = logmean(PLAIN)
oof = np.full((len(rows), 3), np.nan)
for f in [f for f in range(5) if Path(f'experiments/lane_backbone_20261108/runs/synthft_vitl_f{f}_s46/exit.json').exists()]:
    dd = Path(f'experiments/lane_backbone_20261108/runs/synthft_vitl_f{f}_s46/raw/runs/fold{f}') / FAM
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
    a, b_, c, t = f99[ok, k], oof[ok, k], plain[ok, k], ly[ok, k]
    san[nm] = dict(r_f99_vs_oof=round(float(np.corrcoef(a, b_)[0, 1]), 4), mean_log_ratio_f99_over_oof=round(float((a - b_).mean()), 4),
                   r_f99_vs_plain_f99=round(float(np.corrcoef(a, c)[0, 1]), 4), mean_log_ratio_f99_over_plain_f99=round(float((a - c).mean()), 4),
                   r_f99_vs_label=round(float(np.corrcoef(a, t)[0, 1]), 4), r_oof_vs_label=round(float(np.corrcoef(b_, t)[0, 1]), 4),
                   r_plain_f99_vs_label=round(float(np.corrcoef(c, t)[0, 1]), 4))
out['sanity_training_rows'] = dict(n=int(ok.sum()), rows_seed=1111, read='raw, mean log over epochs 8-12, single view', **san)
print(json.dumps({'sanity': out['sanity_training_rows']}), flush=True)
Path('experiments/lane_backbone_20261108/reads/verify_synthvitl_f99_20261112.json').write_text(json.dumps(out, indent=1))
print('saved reads/verify_synthvitl_f99_20261112.json')
