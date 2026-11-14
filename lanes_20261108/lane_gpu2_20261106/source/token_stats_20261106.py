"""Token statistics from the recipe A encoders, frozen read for G3 (lane GPU 2, 2026-11-06).
Each row goes through the seed-46 epoch-20 encoder of the fold that held it out, one unflipped view, bf16 forward as in training, statistics in float32.
Writes one float32 memmap per block under frozen_s46/, plus meta.json with the block sizes and the check against the stored predictions.
Blocks: see BLOCKS below. PCA basis is fit once, without labels, on fold-0 validation tokens through the fold-0 encoder, and used for all encoders so the features stay comparable across folds.
Usage: token_stats_20261106.py extract | align | own <fold>"""
import json, os, sys, time
from pathlib import Path
os.chdir('/workspace/clear-ec-research')
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
X = Path('experiments')
sys.path.insert(0, str(X / 'phase2_pooling_20261020/source'))
sys.path.insert(0, str(X / 'phase2_pooling_20261020/final_readback_20261021'))
sys.path.insert(0, str(X / 'eval_harness_20261031/source'))
import phase2_foundation_train_20261002 as b
import phase2_pooling_inference_20261020 as I
import eval_oof_20261031 as E

LANE = X / 'lane_gpu2_20261106'
OUT = LANE / 'frozen_s46'
ENC = {0: 'phase2_pooling_replication_20261026/f0_s46_std', 1: 'oof_recipeA_20261027/fold1/regression_std_s46', 2: 'oof_recipeA_20261027/fold2/regression_std_s46_v2',
       3: 'oof_recipeA_20261027/fold3/regression_std_s46', 4: 'phase2_pooling_20261020/arms/std'}
HT, WT = 124, 164; N = HT * WT; K = 64; BS = 8
Q = (.01, .05, .25, .5, .75, .95, .99); QI = torch.tensor([round(q * (N - 1)) for q in Q])
LAGS = (1, 2, 3, 4, 5, 6, 8, 10); PAD = 10; NSPEC = 24; SCALES = (2, 4, 8, 16); LOCAL = (16, 32)
ROI_T = (51, 67); ROI_STRIDE = 8; HULL_T = (38, 48)
BLOCKS = dict(mean=768, std=768, skew=768, kurt=768, quant=768 * len(Q), cov=K * (K + 1) // 2, autocorr=768 * len(LAGS), spectrum=K * NSPEC, multiscale=768 * len(SCALES),
              localhet=768 * 2 * len(LOCAL), winstat=3 * 7, legible_pred=6, legible_std=768 * 2, cls=768, direct=3)


def load_enc(f):
    root = X / ENC[f]; ck = torch.load(root / f'runs/fold{f}/cpdino_vitb/epoch_20.pt', map_location='cpu', weights_only=True)
    net = I.build_model('std'); net.load_state_dict(ck['model'], strict=True); net = net.cuda().eval()
    return net, torch.tensor(ck['log_mean'], dtype=torch.float32).cuda(), torch.tensor(ck['log_std'], dtype=torch.float32).cuda(), root


def fw(net, x):
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        o = net[0].original.forward_features(b.inputs(x, 'cpdino_vitb'))
    return o['x_norm_patchtokens'].float(), o['x_norm_clstoken'].float()


@torch.inference_mode()
def head(net, s, mu, sigma):
    """s: ..., 768 std values. Returns log target."""
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z = net[1:](s.to(torch.bfloat16)).float()
    return z * sigma + mu


def geometry():
    fy = torch.fft.fftfreq(HT).cuda(); fx = torch.fft.rfftfreq(WT).cuda(); r = torch.sqrt(fy[:, None] ** 2 + fx[None] ** 2)
    edges = torch.logspace(np.log10(1 / 62), np.log10(0.5), NSPEC + 1).cuda(); sb = torch.bucketize(r.flatten(), edges) - 1; ok = (sb >= 0) & (sb < NSPEC)
    hp, wp = HT + PAD, WT + PAD; dy = torch.arange(hp).cuda(); dy = torch.minimum(dy, hp - dy); dx = torch.arange(wp).cuda(); dx = torch.minimum(dx, wp - dx)
    rr = torch.sqrt(dy[:, None].float() ** 2 + dx[None].float() ** 2); ov = ((HT - dy[:, None]).clamp(min=1) * (WT - dx[None]).clamp(min=1)).float()
    keep = (dy[:, None] <= PAD) & (dx[None] <= PAD); masks = [((rr.round() == l) & keep) for l in LAGS]
    return sb, ok, masks, ov


@torch.inference_mode()
def stats(net, x, cen, mu, sigma, PM, PP, G):
    t, cls = fw(net, x); B = len(t); out = {}; sb, ok, masks, ov = G
    m = t.mean(1); d = t - m[:, None]; v = d.pow(2).mean(1); sd = (v + 1e-12).sqrt()
    out.update(mean=m, std=sd, skew=d.pow(3).mean(1) / sd ** 3, kurt=d.pow(4).mean(1) / (v + 1e-12) ** 2 - 3, cls=cls)
    out['quant'] = t.sort(1).values[:, QI.cuda()].flatten(1)
    Z = (t - PM) @ PP; Zc = Z - Z.mean(1, keepdim=True); C = Zc.transpose(1, 2) @ Zc / N; iu = torch.triu_indices(K, K).cuda(); out['cov'] = C[:, iu[0], iu[1]]
    P = torch.fft.rfft2(Zc.reshape(B, HT, WT, K).permute(0, 3, 1, 2)).abs().pow(2).flatten(2)
    S = torch.zeros(B, K, NSPEC, device=t.device); S.index_add_(2, sb[ok], P[:, :, ok]); out['spectrum'] = (S / S.sum(2, keepdim=True).clamp(min=1e-9)).flatten(1)
    g = d.reshape(B, HT, WT, 768).permute(0, 3, 1, 2)
    A = torch.fft.irfft2(torch.fft.rfft2(g, s=(HT + PAD, WT + PAD)).abs().pow(2), s=(HT + PAD, WT + PAD)) / ov; A0 = A[:, :, 0, 0].clamp(min=1e-9)
    out['autocorr'] = torch.stack([A[:, :, q].mean(-1) / A0 for q in masks], 2).flatten(1)
    tg = t.reshape(B, HT, WT, 768).permute(0, 3, 1, 2); out['multiscale'] = torch.cat([F.avg_pool2d(tg, s, s).flatten(2).std(2, unbiased=False) for s in SCALES], 1)
    t2 = tg.pow(2); L = []
    for s in LOCAL:
        var = (F.avg_pool2d(t2, s, s) - F.avg_pool2d(tg, s, s).pow(2)).clamp(min=0); ls = (var + 1e-12).sqrt().flatten(2); L += [ls.mean(2), ls.std(2, unbiased=False)]
    out['localhet'] = torch.cat(L, 1)
    wv = (F.avg_pool2d(t2, ROI_T, ROI_STRIDE) - F.avg_pool2d(tg, ROI_T, ROI_STRIDE).pow(2)).clamp(min=0); ws = (wv + 1e-12).sqrt().flatten(2).transpose(1, 2)
    lp = head(net, ws, mu, sigma); so = lp.sort(1).values; n = lp.shape[1]
    out['winstat'] = torch.cat([lp.mean(1), lp.std(1, unbiased=False), so[:, 0], so[:, -1], so[:, n // 10], so[:, n // 2], so[:, 9 * n // 10]], 1)
    LS, LP = [], []
    for (h, w), c in zip((ROI_T, HULL_T), (cen[:, :2], cen[:, 2:])):
        ys, xs = [], []
        for i in range(B):
            y0 = int(np.clip(round((float(c[i, 1]) - h * 4) / 8), 0, HT - h)); x0 = int(np.clip(round((float(c[i, 0]) - w * 4) / 8), 0, WT - w)); ys.append(y0); xs.append(x0)
        s_ = torch.stack([(tg[i, :, ys[i]:ys[i] + h, xs[i]:xs[i] + w].flatten(1).var(1, unbiased=False) + 1e-12).sqrt() for i in range(B)]); LS.append(s_); LP.append(head(net, s_, mu, sigma))
    out['legible_std'] = torch.cat(LS, 1); out['legible_pred'] = torch.cat(LP, 1)
    out['direct'] = head(net, sd, mu, sigma)
    return out


def rows_of(base, f): return base['idx'][base['fold'] == f]


def centres(base):
    z = np.load(X / 'meta_ceiling_20261103/sweep/legible_window.npz', allow_pickle=True); nm = list(z['names'].astype(str)); Fm = z['F']
    c = np.stack([Fm[:, nm.index(q)] for q in ('top_box_x', 'top_box_y', 'top_hull_x', 'top_hull_y')], 1).astype(np.float32)
    bad = ~np.isfinite(c); c[:, [0, 2]] = np.where(bad[:, [0, 2]], 648., c[:, [0, 2]]); c[:, [1, 3]] = np.where(bad[:, [1, 3]], 486., c[:, [1, 3]])
    return c, int(bad[:, 0].sum())


def fit_pca(base):
    net, mu, sigma, _ = load_enc(0); ids = rows_of(base, 0); rng = np.random.default_rng(0); pick = np.sort(rng.choice(ids, 400, replace=False)); yy = np.zeros((base['total'], 3), np.float32)
    dl = DataLoader(b.Frames(pick, yy, base['total']), batch_size=BS, num_workers=3, pin_memory=True, shuffle=False); keep = []; gen = torch.Generator().manual_seed(0)
    for x, _, _ in dl:
        t, _c = fw(net, x); keep.append(t[:, torch.randperm(N, generator=gen)[:2000].cuda()].reshape(-1, 768).double())
    T_ = torch.cat(keep); PM = T_.mean(0, keepdim=True); w, V = torch.linalg.eigh((T_ - PM).T @ (T_ - PM) / len(T_))
    PP = V[:, -K:].flip(1); del net
    return PM.float(), PP.float(), [float(v) for v in (w.flip(0)[:K].sum() / w.sum(), w.flip(0)[0] / w.sum())]


def extract():
    OUT.mkdir(parents=True, exist_ok=True); bb = E.load_base(); meta_y = np.load(b.META, allow_pickle=True); total = len(meta_y['ids']); bb['total'] = total; yy = meta_y['Y'].astype(np.float32)
    nrow = len(bb['idx']); pos = np.full(total, -1); pos[bb['idx']] = np.arange(nrow); cen, nbad = centres(bb)
    mm = {k: np.lib.format.open_memmap(OUT / f'{k}.npy', mode='w+', dtype=np.float32, shape=(nrow, d)) for k, d in BLOCKS.items()}
    PM, PP, ev = fit_pca(bb); np.savez(OUT / 'pca.npz', mean=PM.cpu().numpy(), basis=PP.cpu().numpy()); G = geometry(); info = dict(pca_variance_top64=ev[0], pca_variance_top1=ev[1], blocks=BLOCKS, encoders=ENC, legible_missing=nbad, folds={}); t0 = time.time(); done = 0
    for f in range(5):
        net, mu, sigma, root = load_enc(f); ids = rows_of(bb, f); dl = DataLoader(b.Frames(ids, yy, total), batch_size=BS, num_workers=3, pin_memory=True, prefetch_factor=2, shuffle=False)
        for x, _, ix in dl:
            r = pos[ix.numpy()]; o = stats(net, x, torch.from_numpy(cen[ix.numpy()]), mu, sigma, PM, PP, G)  # legible_window is keyed by cache row
            for k, v in o.items(): mm[k][r] = v.cpu().numpy()
            done += len(r)
            if done % 800 == 0: print(json.dumps(dict(done=done, of=nrow, fps=round(done / (time.time() - t0), 2), gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))), flush=True)
        for k in mm: mm[k].flush()
        z = np.load(root / f'runs/fold{f}/cpdino_vitb/predictions_epoch_20.npz'); rr = pos[z['indices'].astype(int)]; p = np.exp(np.asarray(mm['direct'][rr]).astype(np.float64)); diff = np.abs(p / z['pred'] - 1)
        info['folds'][f] = dict(rows=int(len(ids)), checked=int(len(rr)), median_rel_diff=float(np.median(diff)), p99_rel_diff=float(np.percentile(diff, 99)), stored_score=E.official(E.ape(z['pred'].astype(float), z['target'].astype(float))), direct_score=E.official(E.ape(p, z['target'].astype(float))))
        print(json.dumps(dict(fold=f, **info['folds'][f])), flush=True); del net; torch.cuda.empty_cache()
    info['seconds'] = time.time() - t0; (OUT / 'meta.json').write_text(json.dumps(info, indent=1)); print('extract done', flush=True)


def extract_own(f):
    """Fallback in the gate: encoder f on all rows. Rows of fold f are copied from frozen_s46, the others computed. Same PCA basis."""
    D = LANE / f'frozen_own_f{f}'; D.mkdir(parents=True, exist_ok=True); bb = E.load_base(); meta_y = np.load(b.META, allow_pickle=True); total = len(meta_y['ids']); yy = meta_y['Y'].astype(np.float32); nrow = len(bb['idx']); pos = np.full(total, -1); pos[bb['idx']] = np.arange(nrow); cen, _ = centres(bb)
    z = np.load(OUT / 'pca.npz'); PM, PP = torch.from_numpy(z['mean']).cuda(), torch.from_numpy(z['basis']).cuda(); G = geometry(); mine = bb['fold'] == f
    mm = {k: np.lib.format.open_memmap(D / f'{k}.npy', mode='w+', dtype=np.float32, shape=(nrow, d)) for k, d in BLOCKS.items()}
    for k in mm: mm[k][mine] = np.load(OUT / f'{k}.npy', mmap_mode='r')[mine]
    net, mu, sigma, root = load_enc(f); ids = bb['idx'][~mine]; dl = DataLoader(b.Frames(ids, yy, total), batch_size=BS, num_workers=3, pin_memory=True, prefetch_factor=2, shuffle=False); t0 = time.time(); done = 0
    for x, _, ix in dl:
        r = pos[ix.numpy()]; o = stats(net, x, torch.from_numpy(cen[ix.numpy()]), mu, sigma, PM, PP, G)
        for k, v in o.items(): mm[k][r] = v.cpu().numpy()
        done += len(r)
        if done % 800 == 0: print(json.dumps(dict(fold=f, done=done, of=len(ids), fps=round(done / (time.time() - t0), 2))), flush=True)
    for k in mm: mm[k].flush()
    (D / 'meta.json').write_text(json.dumps(dict(encoder=ENC[f], rows=int(nrow), computed=int(len(ids)), seconds=time.time() - t0))); print('own done', f, flush=True)


def align():
    """How far apart are the five encoders on the same frames? 64 rows, each encoder."""
    bb = E.load_base(); meta_y = np.load(b.META, allow_pickle=True); total = len(meta_y['ids']); yy = meta_y['Y'].astype(np.float32); ids = np.sort(np.random.default_rng(1).choice(bb['idx'], 64, replace=False)); S = []; D = []
    for f in range(5):
        net, mu, sigma, _ = load_enc(f); out = []; dd = []
        for x, _, _ in DataLoader(b.Frames(ids, yy, total), batch_size=BS, num_workers=2, shuffle=False):
            t, _c = fw(net, x); s = (t.var(1, unbiased=False) + 1e-12).sqrt(); out.append(s.cpu()); dd.append(head(net, s, mu, sigma).cpu())
        S.append(torch.cat(out).numpy()); D.append(torch.cat(dd).numpy()); del net; torch.cuda.empty_cache()
    S, D = np.stack(S), np.stack(D); res = dict(rows=64, per_channel_r=[], mean_ratio=[], direct_r=[])
    for a in range(5):
        for c in range(a + 1, 5):
            r = [np.corrcoef(S[a, :, j], S[c, :, j])[0, 1] for j in range(768)]; res['per_channel_r'].append(float(np.median(r))); res['mean_ratio'].append(float(np.median(S[a].mean(0) / S[c].mean(0)))); res['direct_r'].append([float(np.corrcoef(D[a, :, j], D[c, :, j])[0, 1]) for j in range(3)])
    res['summary'] = dict(median_channel_r=float(np.median(res['per_channel_r'])), direct_r_mean=np.mean(res['direct_r'], 0).tolist()); print(json.dumps(res['summary']), flush=True); OUT.mkdir(parents=True, exist_ok=True); (OUT / 'align.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'own': extract_own(int(sys.argv[2]))
    else: {'extract': extract, 'align': align}[sys.argv[1]]()
