"""Frozen read, step 1 (lane GPU 1, 2026-11-06): mean and std pooled whole-frame tokens of one encoder on all 8954 frames. No labels.
Same input as recipe A inference: CLAHE, percentile 1-99, replicate pad to 992 x 1312, Cellpose 0-1 range, no TTA.
Usage: frozen_features_20261106.py <name> <checkpoint or external>; writes experiments/lane_gpu1_20261106/features/<name>.npz."""
import sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, 'experiments/meta_ceiling_20261103/source')
import phase2_foundation_preflight_20261002 as foundation

RAW = 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat'
N = 8954
OUT = Path('experiments/lane_gpu1_20261106/features')


class Frames(Dataset):
    def __init__(self):
        self.raw = None

    def __len__(self):
        return N

    def __getitem__(self, i):
        cv2.setNumThreads(1)
        if self.raw is None:
            self.raw = np.memmap(RAW, mode='r', dtype=np.uint8, shape=(N, 972, 1296))
        a = cv2.createCLAHE(2.0, (1296 // 32, 972 // 32)).apply(np.asarray(self.raw[i]))
        lo, hi = np.percentile(a, (1, 99))
        return torch.from_numpy(np.clip((a.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)[None]), i


def load_encoder(ckpt):
    torch.manual_seed(0)
    model, _, _ = foundation.build('cpdino_vitb', False)
    enc = model[0].original
    if ckpt != 'external':
        c = torch.load(ckpt, map_location='cpu', weights_only=False)
        if 'encoder' in c:
            enc.load_state_dict(c['encoder'], strict=True)
        else:
            pre = '0.original.'
            enc.load_state_dict({k[len(pre):]: v for k, v in c['model'].items() if k.startswith(pre)}, strict=True)
        del c
    return enc.cuda().eval()


def main():
    name, ckpt = sys.argv[1], sys.argv[2]
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f'{name}.npz'
    if out.exists():
        print('exists', out, flush=True)
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    enc = load_encoder(ckpt)
    dl = DataLoader(Frames(), batch_size=8, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)
    M, S, I = [], [], []
    t0 = time.time()
    with torch.inference_mode():
        for n, (x, ix) in enumerate(dl):
            x = F.pad(x.cuda(non_blocking=True).expand(-1, 3, -1, -1), (0, 16, 0, 20), mode='replicate').contiguous()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                t = enc.forward_features(x)['x_norm_patchtokens']
            t = t.float()
            m = t.mean(1)
            M.append(m.cpu().numpy()); S.append((t.var(1, unbiased=False) + 1e-12).sqrt().cpu().numpy()); I.append(ix.numpy())
            if n % 100 == 0:
                print(name, n * 8, round(time.time() - t0), flush=True)
    I = np.concatenate(I)
    assert np.array_equal(I, np.arange(N))
    tmp = out.with_suffix('.tmp.npz')
    np.savez(tmp, mean=np.concatenate(M).astype(np.float32), std=np.concatenate(S).astype(np.float32), ckpt=str(ckpt))
    tmp.replace(out)
    print('done', name, round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
