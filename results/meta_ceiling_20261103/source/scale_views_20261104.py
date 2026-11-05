"""Zoomed views of every validation row of one trained recipe A member (2026-11-04), for the per-image zoom-response feature. Views s = 0.8 and 1.25 of the raw frame before CLAHE, one view each, no flips;
s = 1 comes from the stored predictions. Writes indices, log predictions at both zooms and at s = 1. usage: scale_views_20261104.py ROOT OUT.npz. One GPU, shared with a training."""
import json, sys
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import phase2_foundation_train_20261002 as b
import phase2_foundation_preflight_20261002 as preflight
import phase2_pooling_20261019 as pooling
SCALES = (0.8, 1.25)
H, W = 972, 1296


def zoom(a, s):
    M = np.array([[s, 0, (1 - s) * (W - 1) / 2], [0, s, (1 - s) * (H - 1) / 2]], np.float32)
    return cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


class Views(torch.utils.data.Dataset):
    def __init__(self, idx):
        self.idx, self.raw = idx, None
    def __len__(self):
        return len(self.idx) * len(SCALES)
    def __getitem__(self, k):
        cv2.setNumThreads(1)
        if self.raw is None:
            self.raw = np.memmap(b.RAW, mode='r', dtype=np.uint8, shape=(8954, H, W))
        i, j = divmod(k, len(SCALES)); a = zoom(np.asarray(self.raw[int(self.idx[i])]), SCALES[j])
        a = cv2.createCLAHE(2.0, (W // 32, H // 32)).apply(a); lo, hi = np.percentile(a, (1, 99)); a = np.clip((a.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
        return torch.from_numpy(np.ascontiguousarray(a[None])), i, j


def main():
    root, out = Path(sys.argv[1]), sys.argv[2]
    torch.set_num_threads(2); m = json.loads((root / 'manifest.json').read_text()); run = root / 'runs' / f'fold{m["fold"]}' / 'cpdino_vitb'
    ck = torch.load(run / f'epoch_{m["recipe"]["epochs"]:02d}.pt', map_location='cpu', weights_only=True)
    net, _, _ = preflight.build('cpdino_vitb', False, native_convnext_norm=True); net[0] = pooling.SpatialPoolEncoder(net[0].original, m['recipe']['pooling']); dim = 768 * len(pooling.MODES[m['recipe']['pooling']])
    if dim != 768:
        net[1], net[2] = torch.nn.LayerNorm(dim), torch.nn.Linear(dim, 128)
    net.load_state_dict(ck['model'], strict=True); net = net.cuda().eval().requires_grad_(False)
    mu, sigma = np.array(ck['log_mean'], float), np.array(ck['log_std'], float)
    with np.load(run / 'predictions.npz') as z:
        idx, pred = z['indices'].copy(), z['pred'].astype(np.float64)
    L = np.zeros((len(idx), len(SCALES), 3)); dl = torch.utils.data.DataLoader(Views(idx), batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    for k, (x, i, j) in enumerate(dl):
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]), torch.autocast('cuda', dtype=torch.bfloat16):
            z = net(b.inputs(x, 'cpdino_vitb')).float().cpu().numpy()
        L[i.numpy(), j.numpy()] = z * sigma + mu
        if k % 100 == 0:
            print(k * 4, 'of', len(idx) * len(SCALES), flush=True)
    np.savez(out, indices=idx, L=L, s1=np.log(pred), scales=np.array(SCALES), fold=m['fold']); print('done', flush=True)


if __name__ == '__main__':
    main()
