"""Image values at cell-graph nodes from a label-free encoder (lane GPU 1, 2026-11-07), for belief 14 of the audit: SSL5 tokens at nodes memorised (32 dims +0.165, +0.144; 8 dims +0.041), with one token source and no regulariser.
For each node of a graph file with xy: the whole-frame patch token at the node, PCA fitted on nodes of 600 random frames, top 64 components, z-scored. Same input as inference. Output is aligned with the graph file's nodes, in the format cell_graph_sweep3 --tokens reads.
Usage: node_tokens_20261107.py <encoder checkpoint> <graphs_xy.npz> <out.npz>"""
import sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, 'experiments/lane_gpu1_20261106/source')
import frozen_features_20261106 as FF

N = 8954
ckpt, gpath, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
with np.load(gpath) as z:
    xy, off = z['xy'].astype(np.float32), z['offs'].astype(np.int64)
assert len(off) == N + 1 and off[-1] == len(xy)
col = np.clip(np.rint((xy[:, 0] - 4) / 8).astype(np.int64), 0, 163); row = np.clip(np.rint((xy[:, 1] - 4) / 8).astype(np.int64), 0, 123)
torch.backends.cuda.matmul.allow_tf32 = True
enc = FF.load_encoder(ckpt)
DIM = 64
fit = set(np.random.default_rng(0).choice(N, 600, replace=False).tolist())
Z = np.empty((len(xy), DIM), np.float16)
dl = torch.utils.data.DataLoader(FF.Frames(), batch_size=8, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)


def tokens(x):
    x = torch.nn.functional.pad(x.cuda(non_blocking=True).expand(-1, 3, -1, -1), (0, 16, 0, 20), mode='replicate').contiguous()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        return enc.forward_features(x)['x_norm_patchtokens'].float()


# First pass over the frames chosen for the fit, so every node then uses the same projection.
S = []
t0 = time.time()
sub = torch.utils.data.DataLoader(torch.utils.data.Subset(FF.Frames(), sorted(fit)), batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
for x, ix in sub:
    t = tokens(x)
    for k, i in enumerate(ix.tolist()):
        a, c = off[i], off[i + 1]
        if c > a:
            S.append(t[k, row[a:c] * 164 + col[a:c]].cpu())
S = torch.cat(S).double(); mu = S.mean(0); U, s, V = torch.pca_lowrank(S - mu, q=DIM, center=False, niter=6)
print('pca nodes', len(S), 'explained', round(float((s ** 2).sum() / ((S - mu) ** 2).sum()), 4), round(time.time() - t0), flush=True)
mu_c, V_c = mu.float().cuda(), V.float().cuda(); sd = torch.from_numpy(np.ones(DIM)).float().cuda()
sd = ((S - mu) @ V).std(0).float().cuda()
del S
for n, (x, ix) in enumerate(dl):
    t = tokens(x)
    for k, i in enumerate(ix.tolist()):
        a, c = off[i], off[i + 1]
        if c > a:
            Z[a:c] = (((t[k, torch.from_numpy(row[a:c] * 164 + col[a:c]).cuda()] - mu_c) @ V_c) / sd).cpu().numpy().astype(np.float16)
    if n % 100 == 0:
        print(n * 8, round(time.time() - t0), flush=True)
tmp = out.with_suffix('.tmp.npz')
np.savez(tmp, tokens=Z, ckpt=str(ckpt), graphs=str(gpath))
tmp.replace(out)
print('done', out, round(time.time() - t0), flush=True)
