"""Does recipe A read cell size? Zoom response of a trained member (2026-11-04). Zooming a frame by s makes each cell s times wider, so the frame's true CD becomes
CD / s^2 while CV and HEX stay put. This needs no labels and the model has never seen a zoom (training uses flips only).
Fold 0, seed 46 std, validation rows, one view per scale. Reported per metric: slope of log prediction on log s (ideal -2, 0, 0); the slope's spread across images; error at each scale after
undoing the zoom; zoom TTA with the physical correction and with a slope learned on half the rows; error correlation between scales; the same rows' four-flip TTA as reference.
usage: scale_response_20261104.py ROOT N OUT.json. One GPU, shared with a training."""
import json, math, sys
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import phase2_foundation_train_20261002 as b
import phase2_foundation_preflight_20261002 as preflight
import phase2_pooling_20261019 as pooling
SCALES = (0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4)
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


def official(p, y):
    s = y > 0; e = np.where(s, np.abs(p - y) / np.where(s, y, 1.0), np.nan) * 100; m = np.nanmean(e, 0)
    return dict(CD=round(float(m[0]), 3), CV=round(float(m[1]), 3), HEX=round(float(m[2]), 3), mean=round(float(m.mean()), 3))


def main():
    root, n, out = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    torch.set_num_threads(2); m = json.loads((root / 'manifest.json').read_text()); run = root / 'runs' / f'fold{m["fold"]}' / 'cpdino_vitb'
    ck = torch.load(run / f'epoch_{m["recipe"]["epochs"]:02d}.pt', map_location='cpu', weights_only=True)
    net, _, _ = preflight.build('cpdino_vitb', False, native_convnext_norm=True); net[0] = pooling.SpatialPoolEncoder(net[0].original, m['recipe']['pooling']); dim = 768 * len(pooling.MODES[m['recipe']['pooling']])
    if dim != 768:
        net[1], net[2] = torch.nn.LayerNorm(dim), torch.nn.Linear(dim, 128)
    net.load_state_dict(ck['model'], strict=True); net = net.cuda().eval().requires_grad_(False)
    mu, sigma = np.array(ck['log_mean'], float), np.array(ck['log_std'], float)
    with np.load(b.META, allow_pickle=True) as z:
        Y = z['Y'].astype(np.float64)
    with np.load(run / 'predictions.npz') as z:
        va = z['indices'].copy()
    idx = np.sort(np.random.default_rng(0).choice(va, min(n, len(va)), replace=False)); y = Y[idx]
    L = np.zeros((len(idx), len(SCALES), 3)); dl = torch.utils.data.DataLoader(Views(idx), batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    for k, (x, i, j) in enumerate(dl):
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]), torch.autocast('cuda', dtype=torch.bfloat16):
            z = net(b.inputs(x, 'cpdino_vitb')).float().cpu().numpy()
        L[i.numpy(), j.numpy()] = z * sigma + mu
        if k % 100 == 0:
            print(k * 4, 'of', len(idx) * len(SCALES), flush=True)
    ls = np.log(np.array(SCALES)); x = ls - ls.mean(); slopes = ((L - L.mean(1, keepdims=True)) * x[None, :, None]).sum(1) / (x * x).sum()
    ideal = np.array([-2.0, 0.0, 0.0]); rep = dict(n=len(idx), scales=SCALES, slope_median={k: round(float(np.median(slopes[:, j])), 3) for j, k in enumerate(('CD', 'CV', 'HEX'))}, slope_iqr={k: [round(float(q), 3) for q in np.percentile(slopes[:, j], (25, 75))] for j, k in enumerate(('CD', 'CV', 'HEX'))})
    rep['log_pred_mean_by_scale'] = {str(s): [round(float(v), 3) for v in L[:, j].mean(0)] for j, s in enumerate(SCALES)}
    def undo(j, be=ideal): return np.exp(L[:, j] - be * np.log(SCALES[j]))
    rep['error_by_scale_physical'] = {str(s): official(undo(j), y) for j, s in enumerate(SCALES)}
    half = np.random.default_rng(1).permutation(len(idx)); A, C = half[: len(idx) // 2], half[len(idx) // 2:]; be = np.array([np.median(slopes[A, j]) for j in range(3)])
    rep['learned_slope'] = [round(float(v), 3) for v in be]
    for nm, sel in dict(z90_110=(2, 3, 4), z80_125=(1, 3, 5), z80_90_110_125=(1, 2, 3, 4, 5), all7=tuple(range(7))).items():
        ph = np.exp(np.mean([np.log(undo(j)) for j in sel], 0)); le = np.exp(np.mean([L[:, j] - be * np.log(SCALES[j]) for j in sel], 0))
        rep[nm] = dict(physical=official(ph, y), learned_on_half_read_on_other=official(le[C], y[C]), s1_same_half=official(np.exp(L[C, 3]), y[C]))
    e = np.log(np.maximum(y, 1e-6)) - L[:, 3]
    rep['resid_corr_s1_vs'] = {str(s): [round(float(np.corrcoef((np.log(np.maximum(y[:, q], 1e-6)) - np.log(undo(j)[:, q]))[y[:, q] > 0], e[y[:, q] > 0, q])[0, 1]), 3) for q in range(3)] for j, s in enumerate(SCALES) if s != 1.0}
    tta = run / 'predictions_tta4.npz'
    if tta.exists():
        with np.load(tta, allow_pickle=True) as z:
            keys = list(z.files); pk = [k for k in keys if k in ('pred', 'pred_tta', 'tta_pred', 'prediction')]; rep['tta_keys'] = keys
            if pk:
                pos = {int(i): q for q, i in enumerate(z['indices'])}; rep['flip_tta4'] = official(z[pk[0]][[pos[int(i)] for i in idx]], y)
    rep['s1'] = official(np.exp(L[:, 3]), y); json.dump(rep, open(out, 'w'), indent=1); np.savez(out + '.npz', idx=idx, L=L, y=y); print(json.dumps(rep), flush=True); print('done', flush=True)


if __name__ == '__main__':
    main()
