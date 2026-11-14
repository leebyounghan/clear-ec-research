"""CV lane (2026-11-08). Token statistics from the DINOv3 B/16 regression encoders (epoch 20, seed 46), each fold's own encoder on ALL rows (the 'own'
read of lane GPU 2, but for DINOv3). Blocks in lane GPU 2's format so its frozen head code reads them: std (768), cov (PCA-64 upper triangle), spectrum
(PCA-64 radial power, 24 log bins), autocorr (768 x 8 lags). Grid 62 x 82 (stride 16 on the 992 x 1312 padded frame). PCA basis fit once, label-free,
on 400 fold-0 validation frames through the fold-0 encoder. Writes frozen_dino_f<f>/<block>.npy in this lane.
usage (from /workspace/clear-ec-research, one visible GPU): dino_token_stats_20261108.py FOLDS"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, 'experiments/lane_gpu3_20261106/source')
import family_recipe_20261106 as FRm
sys.path.insert(0, 'experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
b = FRm.b; X = Path('experiments'); ME = X / 'lane_cv_20261108'
ROOTS = {0: 'lane_gpu3_20261106/runs/dinov3_b16_f0_s46', 4: 'lane_gpu3_20261106/runs/dinov3_b16_f4_s46', 1: 'lane_cd_20261108/arms/dinov3_b16_f1_s46', 2: 'lane_cd_20261108/arms/dinov3_b16_f2_s46', 3: 'lane_cd_20261108/arms/dinov3_b16_f3_s46'}
HT, WT = 62, 82; N = HT * WT; K = 64; NSPEC = 24; LAGS = (1, 2, 3, 4, 5, 6, 8, 10); PAD = 10; BS = 8
BLOCKS = dict(std=768, cov=K * (K + 1) // 2, spectrum=K * NSPEC, autocorr=768 * len(LAGS))
def load(f):
    tok = FRm.build_tokens('dinov3_b16'); net = torch.nn.Sequential(FRm.Pooled(tok, 'std'), torch.nn.LayerNorm(tok.dim), torch.nn.Linear(tok.dim, 128), torch.nn.GELU(), torch.nn.Linear(128, 3))
    ck = torch.load(X / ROOTS[f] / f'runs/fold{f}/dinov3_b16/epoch_20.pt', map_location='cpu', weights_only=False); net.load_state_dict(ck['model'], strict=True); return net.cuda().eval()
@torch.inference_mode()
def toks(net, x):
    with torch.autocast('cuda', dtype=torch.bfloat16): t = net[0].tokens(FRm.inputs(x, 'dinov3_b16'))
    assert t.shape[1] == N, t.shape; return t.float()
def geometry():
    fy = torch.fft.fftfreq(HT).cuda(); fx = torch.fft.rfftfreq(WT).cuda(); r = torch.sqrt(fy[:, None] ** 2 + fx[None] ** 2)
    edges = torch.logspace(np.log10(1 / 31), np.log10(0.5), NSPEC + 1).cuda(); sb = torch.bucketize(r.flatten(), edges) - 1; ok = (sb >= 0) & (sb < NSPEC)
    hp, wp = HT + PAD, WT + PAD; dy = torch.arange(hp).cuda(); dy = torch.minimum(dy, hp - dy); dx = torch.arange(wp).cuda(); dx = torch.minimum(dx, wp - dx)
    rr = torch.sqrt(dy[:, None].float() ** 2 + dx[None].float() ** 2); ov = ((HT - dy[:, None]).clamp(min=1) * (WT - dx[None]).clamp(min=1)).float()
    keep = (dy[:, None] <= PAD) & (dx[None] <= PAD); masks = [((rr.round() == l) & keep) for l in LAGS]; return sb, ok, masks, ov
@torch.inference_mode()
def stats(t, PM, PP, G):
    B = len(t); sb, ok, masks, ov = G; m = t.mean(1); d = t - m[:, None]; out = dict(std=(d.pow(2).mean(1) + 1e-12).sqrt())
    Z = (t - PM) @ PP; Zc = Z - Z.mean(1, keepdim=True); C = Zc.transpose(1, 2) @ Zc / N; iu = torch.triu_indices(K, K).cuda(); out['cov'] = C[:, iu[0], iu[1]]
    P = torch.fft.rfft2(Zc.reshape(B, HT, WT, K).permute(0, 3, 1, 2)).abs().pow(2).flatten(2); S = torch.zeros(B, K, NSPEC, device=t.device); S.index_add_(2, sb[ok], P[:, :, ok]); out['spectrum'] = (S / S.sum(2, keepdim=True).clamp(min=1e-9)).flatten(1)
    g = d.reshape(B, HT, WT, 768).permute(0, 3, 1, 2); A = torch.fft.irfft2(torch.fft.rfft2(g, s=(HT + PAD, WT + PAD)).abs().pow(2), s=(HT + PAD, WT + PAD)) / ov; A0 = A[:, :, 0, 0].clamp(min=1e-9)
    out['autocorr'] = torch.stack([A[:, :, q].mean(-1) / A0 for q in masks], 2).flatten(1); return out
bb = E.load_base(); meta = np.load(b.META, allow_pickle=True); total = len(meta['ids']); yy = meta['Y'].astype(np.float32); nrow = len(bb['idx']); pos = np.full(total, -1); pos[bb['idx']] = np.arange(nrow)
pca = ME / 'frozen_dino_pca.npz'
if not pca.exists():
    net = load(0); ids = bb['idx'][bb['fold'] == 0]; pick = np.sort(np.random.default_rng(0).choice(ids, 400, replace=False)); keep = []; gen = torch.Generator().manual_seed(0)
    for x, _, _ in DataLoader(b.Frames(pick, yy, total), batch_size=BS, num_workers=2, shuffle=False):
        t = toks(net, x); keep.append(t[:, torch.randperm(N, generator=gen)[:2000].cuda()].reshape(-1, 768).double())
    T_ = torch.cat(keep); PM = T_.mean(0, keepdim=True); w, V = torch.linalg.eigh((T_ - PM).T @ (T_ - PM) / len(T_)); PP = V[:, -K:].flip(1)
    np.savez(pca, mean=PM.float().cpu().numpy(), basis=PP.float().cpu().numpy()); del net; torch.cuda.empty_cache(); print('pca fit', flush=True)
z = np.load(pca); PM, PP = torch.from_numpy(z['mean']).cuda(), torch.from_numpy(z['basis']).cuda(); G = geometry()
for f in [int(c) for c in sys.argv[1]]:
    D = ME / f'frozen_dino_f{f}'
    if (D / 'meta.json').exists(): print('skip', f); continue
    D.mkdir(parents=True, exist_ok=True); net = load(f); mm = {k: np.lib.format.open_memmap(D / f'{k}.npy', mode='w+', dtype=np.float32, shape=(nrow, d)) for k, d in BLOCKS.items()}; t0 = time.time(); done = 0
    for x, _, ix in DataLoader(b.Frames(bb['idx'], yy, total), batch_size=BS, num_workers=2, pin_memory=True, shuffle=False):
        r = pos[ix.numpy()]; o = stats(toks(net, x), PM, PP, G)
        for k, v in o.items(): mm[k][r] = v.cpu().numpy()
        done += len(r)
        if done % 1600 == 0: print(json.dumps(dict(fold=f, done=done, of=nrow, fps=round(done / (time.time() - t0), 2))), flush=True)
    for k in mm: mm[k].flush()
    (D / 'meta.json').write_text(json.dumps(dict(root=ROOTS[f], rows=int(nrow), seconds=time.time() - t0))); print('done', f, flush=True); del net; torch.cuda.empty_cache()
