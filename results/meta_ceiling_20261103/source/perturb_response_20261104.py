"""Interrogating a trained member with perturbations that should not change the labels (2026-11-04). The zoom response carried signal about the error (r 0.19-0.24 with the CD residual). Do other label-preserving
perturbations? Same 500 fold-0 validation rows, seed 46 std. Views: blur sigma 1 and 2.5 px, gamma 0.8 and 1.25 on the raw frame, added noise, rotation 7 deg, shift 40 px. For each: mean change in log prediction, change spread,
correlation of the per-image change with the residual and its size, and a half-split fit of the residual on the change. usage: perturb_response_20261104.py ROOT OUT. One GPU, shared."""
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
H, W = 972, 1296
VIEWS = ('id', 'blur1', 'blur2p5', 'gamma0p8', 'gamma1p25', 'noise', 'rot7', 'shift40')


def view(a, name, seed):
    if name == 'id': return a
    if name == 'blur1': return cv2.GaussianBlur(a, (0, 0), 1.0)
    if name == 'blur2p5': return cv2.GaussianBlur(a, (0, 0), 2.5)
    if name == 'gamma0p8': return np.clip(255 * (a / 255.0) ** 0.8, 0, 255).astype(np.uint8)
    if name == 'gamma1p25': return np.clip(255 * (a / 255.0) ** 1.25, 0, 255).astype(np.uint8)
    if name == 'noise': return np.clip(a + np.random.default_rng(seed).normal(0, 6, a.shape), 0, 255).astype(np.uint8)
    if name == 'rot7': return cv2.warpAffine(a, cv2.getRotationMatrix2D(((W - 1) / 2, (H - 1) / 2), 7, 1.0), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if name == 'shift40': return cv2.warpAffine(a, np.array([[1, 0, 40], [0, 1, 40]], np.float32), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    raise ValueError(name)


class Views(torch.utils.data.Dataset):
    def __init__(self, idx):
        self.idx, self.raw = idx, None
    def __len__(self):
        return len(self.idx) * len(VIEWS)
    def __getitem__(self, k):
        cv2.setNumThreads(1)
        if self.raw is None:
            self.raw = np.memmap(b.RAW, mode='r', dtype=np.uint8, shape=(8954, H, W))
        i, j = divmod(k, len(VIEWS)); a = view(np.asarray(self.raw[int(self.idx[i])]), VIEWS[j], int(self.idx[i]))
        a = cv2.createCLAHE(2.0, (W // 32, H // 32)).apply(a); lo, hi = np.percentile(a, (1, 99)); a = np.clip((a.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
        return torch.from_numpy(np.ascontiguousarray(a[None])), i, j


def main():
    root, out = Path(sys.argv[1]), sys.argv[2]
    torch.set_num_threads(2); m = json.loads((root / 'manifest.json').read_text()); run = root / 'runs' / f'fold{m["fold"]}' / 'cpdino_vitb'
    ck = torch.load(run / f'epoch_{m["recipe"]["epochs"]:02d}.pt', map_location='cpu', weights_only=True)
    net, _, _ = preflight.build('cpdino_vitb', False, native_convnext_norm=True); net[0] = pooling.SpatialPoolEncoder(net[0].original, m['recipe']['pooling']); dim = 768 * len(pooling.MODES[m['recipe']['pooling']])
    if dim != 768:
        net[1], net[2] = torch.nn.LayerNorm(dim), torch.nn.Linear(dim, 128)
    net.load_state_dict(ck['model'], strict=True); net = net.cuda().eval().requires_grad_(False); mu, sigma = np.array(ck['log_mean'], float), np.array(ck['log_std'], float)
    z0 = np.load(sys.argv[3]); idx, y = z0['idx'], z0['y']; L = np.zeros((len(idx), len(VIEWS), 3))
    dl = torch.utils.data.DataLoader(Views(idx), batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    for k, (x, i, j) in enumerate(dl):
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]), torch.autocast('cuda', dtype=torch.bfloat16):
            z = net(b.inputs(x, 'cpdino_vitb')).float().cpu().numpy()
        L[i.numpy(), j.numpy()] = z * sigma + mu
        if k % 100 == 0: print(k * 4, 'of', len(idx) * len(VIEWS), flush=True)
    np.savez(out + '.npz', idx=idx, L=L, y=y); ly = np.log(np.where(y > 0, y, np.nan)); P = L[:, 0]; r = ly - P; rep = dict(n=len(idx))
    for v, nm in enumerate(VIEWS[1:], 1):
        D = L[:, v] - P; e = {}
        for j, k in enumerate(('CD', 'CV', 'HEX')):
            q = np.isfinite(r[:, j]); dl_ = []; rng = np.random.default_rng(0)
            for _ in range(40):
                pm = rng.permutation(np.where(q)[0]); a, c = pm[: len(pm) // 2], pm[len(pm) // 2:]; f = np.c_[np.ones(len(y)), D[:, j]]; w = np.linalg.lstsq(f[a], r[a, j], rcond=None)[0]; e1 = np.abs(np.exp(P[c, j] + f[c] @ w) - y[c, j]) / y[c, j]; e0 = np.abs(np.exp(P[c, j] + r[a, j].mean()) - y[c, j]) / y[c, j]; dl_.append((e1.mean() - e0.mean()) * 100)
            e[k] = dict(mean_change=round(float(D[q, j].mean()), 4), sd_change=round(float(D[q, j].std()), 4), r_resid=round(float(np.corrcoef(D[q, j], r[q, j])[0, 1]), 3), r_abs_resid=round(float(np.corrcoef(np.abs(D[q, j]), np.abs(r[q, j]))[0, 1]), 3), half_split=round(float(np.mean(dl_)), 3))
        rep[nm] = e; print(nm, json.dumps(e), flush=True)
    json.dump(rep, open(out, 'w'), indent=1); print('done', flush=True)


if __name__ == '__main__':
    main()
