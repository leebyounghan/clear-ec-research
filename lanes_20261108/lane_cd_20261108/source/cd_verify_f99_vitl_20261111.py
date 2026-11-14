"""Verify the full-data ViT-L members (fold 99, seeds 46/47): sha256 of the deploy files, load each checkpoint into the runner's own
network, and recompute the stored predictions on the held rows (the two HEX-zero rows) through the runner's own evaluate():
final.pt with four-flip TTA vs predictions_final_tta.npz, epoch_08..11.pt single view vs predictions_epoch_08..11.npz.
Run from /workspace/clear-ec-research with one visible GPU (inference only, ~5 GB)."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, 'experiments/lane_cd_20261108/source')
import bb_runner_cd_20261108 as B  # noqa: E402
F, R = B.F, B.R
B.FAMILY = 'dinov3_l16t'
ROOT = Path('experiments/lane_cd_20261108/arms')
SSL = ROOT / 'vitl16_f99_s46/ssl/last.pt'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


ids, y, tr, va = B.arrays(99); n = len(ids)
out = dict(held_rows=va.tolist(), held_ids=ids[va].tolist())
for s in (46, 47):
    d = ROOT / f'vitl16_f99_s{s}/raw/runs/fold99/dinov3_l16t'
    R.ARGS = argparse.Namespace(seed=s)
    net = F.build_net(argparse.Namespace(family='dinov3_l16t'), s, str(SSL))
    dl = DataLoader(R.Frames(va, y, n, 0, False, s), batch_size=8, shuffle=False, num_workers=1)
    rec = {}
    for name, stored, tta in [(f'epoch_{e:02d}.pt', f'predictions_epoch_{e:02d}.npz', False) for e in (8, 9, 10, 11)] + [('final.pt', 'predictions_final_tta.npz', True)]:
        ck = torch.load(d / name, map_location='cuda', weights_only=False)
        missing, unexpected = net.load_state_dict(ck['model'], strict=True), None
        mu = torch.tensor(ck['log_mean'], device='cuda', dtype=torch.float32); sigma = torch.tensor(ck['log_std'], device='cuda', dtype=torch.float32)
        p = B.evaluate(net, dl, mu, sigma, va, tta=tta)
        with np.load(d / stored) as z:
            assert (z['indices'] == va).all()
            q = z['pred'].astype(np.float64)
        rec[name] = dict(sha256=sha(d / name), epoch=int(ck['epoch']), max_rel_diff_vs_stored=float(np.abs(p / q - 1).max()),
                         cd_recomputed=[round(float(v), 2) for v in p[:, 0]], cd_stored=[round(float(v), 2) for v in q[:, 0]])
        print(json.dumps({f's{s}': {name: rec[name]}}), flush=True)
    out[f's{s}'] = rec
    del net; torch.cuda.empty_cache()
Path('experiments/lane_cd_20261108/reads/verify_f99_vitl_20261111.json').write_text(json.dumps(out, indent=1))
print('saved reads/verify_f99_vitl_20261111.json')
