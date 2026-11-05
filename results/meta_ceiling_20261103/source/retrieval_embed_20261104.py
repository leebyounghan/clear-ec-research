"""Label-free token statistics for every frame (2026-11-04): the all-data SSL5 encoder (no labels ever), one view, mean and std of the
patch tokens plus a coarse 8 x 6 grid of token means. Used to test whether an image can find its fellow-eye or adjacent-ID partner,
and for nearest-neighbour residual correction. One GPU, two loader workers, shares the GPU with a running training."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import phase2_foundation_train_20261002 as b
import phase2_foundation_preflight_20261002 as preflight
OUT = Path('experiments/meta_ceiling_20261103/retrieval/ssl_tokens.npz')
assert torch.cuda.device_count() == 1
net, _, _ = preflight.build('cpdino_vitb', False, native_convnext_norm=True)
enc = net[0].original
ck = torch.load('experiments/fulldata_recipeA_20261028/ssl5_s1003/run/last.pt', map_location='cpu', weights_only=True)
enc.load_state_dict(ck['encoder'], strict=True); enc = enc.cuda().eval().requires_grad_(False)
with np.load(b.META, allow_pickle=True) as z:
    y = z['Y'].astype(np.float32); n = len(z['ids'])
idx = np.arange(n)
dl = torch.utils.data.DataLoader(b.Frames(idx, y, n), batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
M, S, C, seen = [], [], [], []
for k, (x, _, ix) in enumerate(dl):
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]), torch.autocast('cuda', dtype=torch.bfloat16):
        t = enc.forward_features(b.inputs(x, 'cpdino_vitb'))['x_norm_patchtokens']
    t = t.float(); B = t.shape[0]
    M.append(t.mean(1).cpu().numpy()); S.append(t.std(1).cpu().numpy())
    C.append(F.adaptive_avg_pool2d(t.transpose(1, 2).reshape(B, 768, 124, 164), (6, 8)).flatten(1).cpu().numpy().astype(np.float16)); seen.append(ix.numpy())
    if k % 200 == 0:
        print(k * 4, flush=True)
np.savez(OUT, idx=np.concatenate(seen), mean=np.concatenate(M), std=np.concatenate(S), coarse=np.concatenate(C))
print('done', flush=True)
