"""Image features for cell-graph nodes (2026-11-05). So far the graph sees geometry and detector confidence only. The all-data label-free SSL5 encoder runs on every frame; its patch tokens are sampled bilinearly at each node, reduced by a PCA fit on nodes of the first 800 frames in random order, whitened, 32 dimensions. One set per graph file. usage: node_tokens_20261105.py GRAPH_A.npz OUT_A.npz GRAPH_B.npz OUT_B.npz. One GPU, shared."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import phase2_foundation_train_20261002 as b
import phase2_foundation_preflight_20261002 as preflight
O = Path('experiments/meta_ceiling_20261103/sweep'); DIMS = 32; args = sys.argv[1:]; pairs = list(zip(args[::2], args[1::2])); assert torch.cuda.device_count() == 1; torch.set_num_threads(2)
net, _, _ = preflight.build('cpdino_vitb', False, native_convnext_norm=True); enc = net[0].original
ck = torch.load('experiments/fulldata_recipeA_20261028/ssl5_s1003/run/last.pt', map_location='cpu', weights_only=True); enc.load_state_dict(ck['encoder'], strict=True); enc = enc.cuda().eval().requires_grad_(False)
G = []
for g_, o_ in pairs:
    z = np.load(O / g_); G.append(dict(xy=z['xy'].astype(np.float32), off=z['offs'], out=O / o_, tok=None, mu=None, V=None, s=None, sample=[]))
    assert len(G[-1]['off']) == 8955
with np.load(b.META, allow_pickle=True) as z:
    y = z['Y'].astype(np.float32)
order = np.random.default_rng(0).permutation(8954); FIT = 800
dl = torch.utils.data.DataLoader(b.Frames(order, y, 8954), batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
def sample(tok, xy):
    p = torch.from_numpy(xy).cuda(); gx = ((p[:, 0] - 4) / 8 + 0.5) / 164 * 2 - 1; gy = ((p[:, 1] - 4) / 8 + 0.5) / 124 * 2 - 1
    return F.grid_sample(tok[None], torch.stack([gx, gy], 1)[None, None], mode='bilinear', align_corners=False, padding_mode='border')[0, :, 0].T
seen = 0; buffer = []
for k, (x, _, ix) in enumerate(dl):
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]), torch.autocast('cuda', dtype=torch.bfloat16):
        t = enc.forward_features(b.inputs(x, 'cpdino_vitb'))['x_norm_patchtokens']
    t = t.float().transpose(1, 2).reshape(len(ix), 768, 124, 164)
    for n, i in enumerate(ix.tolist()):
        for gr in G:
            a, c = gr['off'][i], gr['off'][i + 1]
            if c == a: continue
            v = sample(t[n], gr['xy'][a:c])
            if gr['V'] is None:
                gr['sample'].append((a, c, v.cpu().half()))
            else:
                gr['tok'][a:c] = (((v - gr['mu']) @ gr['V']) / gr['s']).clamp(-8, 8).cpu().numpy().astype(np.float16)
    seen += len(ix)
    if seen >= FIT and G[0]['V'] is None:
        for gr in G:
            S = torch.cat([q[2] for q in gr['sample']]).float().cuda(); idx = torch.randperm(len(S), device=S.device)[:200000]; S0 = S[idx]; gr['mu'] = S0.mean(0); U, s_, V = torch.pca_lowrank(S0 - gr['mu'], q=DIMS, center=False, niter=4); gr['V'] = V; gr['s'] = s_ / (len(S0) - 1) ** 0.5; gr['tok'] = np.zeros((len(gr['xy']), DIMS), np.float16)
            for a, c, v in gr['sample']: gr['tok'][a:c] = (((v.float().cuda() - gr['mu']) @ gr['V']) / gr['s']).clamp(-8, 8).cpu().numpy().astype(np.float16)
            gr['sample'] = []; print('pca fitted', gr['out'].name, 'explained top', [round(float(q), 3) for q in (s_[:4] ** 2 / ((S0 - gr['mu']) ** 2).sum() * (len(S0) - 1) / (len(S0) - 1))], flush=True)
    if k % 200 == 0: print(seen, flush=True)
for gr in G:
    np.savez(gr['out'], tokens=gr['tok']); print('saved', gr['out'].name, gr['tok'].shape, flush=True)
print('done', flush=True)
