"""SSL with strong, domain-specific views (lane GPU 1, plan row G2, 2026-11-06). No labels, fold train rows only, external CPDINO start.
Same LeJEPA math as SSL5: invariance + SIGReg from the frozen pilot, exact feature-gradient caching, same optimiser, LR, warmup and clip.
Set per arm:
  --crop N     square crop side, 0 = whole frame padded as in recipe A. Crops are cut after CLAHE at native scale, so pixel pitch and CLAHE tile geometry match inference. RoPE keeps the whole-frame grid scale.
  --optics     none; measured = a perturbation of the raw frame before CLAHE: exposure, gamma, illumination field, blur, noise, broad ranges (noise and uncompensated blur go past real frames, see lane GPU 0's report); transfer = lane GPU 0's calibrated donor transfer, inside the real range.
  --pool       statistic the SSL sees: mean (as SSL5), std (as the regression), both.
  --views V    views per image.
  --fold 99    all frames, no validation, for the final fit.
Writes experiments/lane_gpu1_20261106/ssl/<name>/: config.json, history.jsonl, run/epochNN.pt, run/last.pt, run/complete.json."""
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, 'experiments/meta_ceiling_20261103/source')
sys.path.insert(0, 'experiments/phase2_lejepa_total5_20261003/source')
import phase2_foundation_preflight_20261002 as foundation
import phase2_lejepa_20261003 as pilot

RAW = 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat'
N, H, W = 8954, 972, 1296
LANE = Path('experiments/lane_gpu1_20261106')
SPLITS = {0: 'experiments/ssl_screen_20260911/preflight/fold0.npz', 1: 'experiments/ssl_screen_20260911/preflight/fold1.npz', 2: 'experiments/overnight_target_20260912/preflight/fold2.npz',
          3: 'experiments/overnight_target_20260912/preflight/fold3.npz', 4: 'experiments/target_fold4_20260912/preflight/fold4.npz'}
OPTICS = dict(
    # Ranges come from lane GPU 1 reads of lane GPU 0's frame_stats.npz (see protocol.json).
    # All labelled frames, p5-p95: mean level 51-145 (log SD about 0.35), centre over corner 1.5-16, left-right gradient -0.05 to 0.68, noise SD 2.2-8.6 gray levels (bright parts about twice dark parts), Tenengrad 205-1322.
    measured=dict(gain=0.25, gamma=0.25, vign=(-0.2, 0.8), tilt=0.25, blur_p=0.5, blur_max=2.0, noise_p=0.8, noise_max=8.0),
)
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
_yy = (_yy - H / 2) / (H / 2); _xx = (_xx - W / 2) / (W / 2)


def perturb(raw, rng, P):
    """Raw frame to raw frame. All in the 0-255 domain, before CLAHE, as a different session on the same device would look."""
    f = raw.astype(np.float32) / 255
    f = f ** float(np.exp(rng.uniform(-P['gamma'], P['gamma'])))
    cx, cy = rng.uniform(-.3, .3, 2)
    field = np.exp(-rng.uniform(*P['vign']) * ((_xx - cx) ** 2 + (_yy - cy) ** 2) + rng.uniform(-P['tilt'], P['tilt']) * _xx + rng.uniform(-P['tilt'], P['tilt']) * _yy)
    f = f * field * float(np.exp(rng.normal(0, P['gain']))) * 255
    q = float(np.percentile(f[::8, ::8], 99))
    if q > 250:  # real frames clip at most a small hot spot; keep the view inside that
        f = f * (250 / q)
    if rng.random() < P['blur_p']:
        f = cv2.GaussianBlur(f, (0, 0), float(rng.uniform(.4, P['blur_max'])))
    if rng.random() < P['noise_p']:
        n = np.empty((H, W), np.float32)
        cv2.setRNGSeed(int(rng.integers(2 ** 31 - 1)))
        cv2.randn(n, 0, float(rng.uniform(.5, P['noise_max'])))
        f = f + n * np.sqrt(np.clip(f, 0, 255) / 128)
    return np.clip(f, 0, 255).astype(np.uint8)


# Calibrated donor transfer, copied from lane GPU 0's recipeA_measured_aug2_20261106.py (noise floor, focus and exposure of real frames, same source, no sharpening, grain kept in the real range).
_STATS = None
KAPPA, GAIN, FOCUS_MAP = (0.4, 0.95), (0.6, 1.6), ([0.25, 0.64, 1.88, 3.63], [0.75, 1.0, 1.5, 2.0])


def stats():
    global _STATS
    if _STATS is None:
        with np.load('experiments/lane_gpu0_20261106/frame_aug_stats.npz', allow_pickle=True) as z:
            _STATS = {k: z[k] for k in ('src', 'sigma', 'focus', 'p50', 'p99')}
    return _STATS


def transfer(a, own, donor, src, rng):
    s = stats(); x = a.astype(np.float32); sig = float(s['sigma'][own])
    d = float(s['focus'][donor] - s['focus'][own])
    if d > FOCUS_MAP[0][0]:
        b = float(np.interp(d, *FOCUS_MAP)); x = cv2.GaussianBlur(x, (0, 0), b)
        k = cv2.getGaussianKernel(int(2 * np.ceil(4 * b) + 1), b); sig *= float((k ** 2).sum())
    g = float(np.clip((s['p99'][donor] - s['p50'][donor]) / max(float(s['p99'][own] - s['p50'][own]), 1.0), *GAIN))
    g = min(g, max(float(s['sigma'][donor]), float(s['sigma'][own])) / max(sig, 1e-3))
    x = g * x + (float(s['p50'][donor]) - g * float(s['p50'][own])); sig *= g
    v = float(s['sigma'][donor]) ** 2 - sig ** 2
    if v > 0:
        amp = (np.maximum(x, 1.0) / max(float(x.mean()), 1.0)) ** KAPPA[src]
        amp /= float(np.sqrt((amp[H // 4:3 * H // 4, W // 4:3 * W // 4] ** 2).mean()))
        x = x + rng.standard_normal(x.shape, dtype=np.float32) * amp * float(np.sqrt(v))
    x[a == 0] = 0
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def clahe01(a):
    a = cv2.createCLAHE(2.0, (W // 32, H // 32)).apply(a)
    lo, hi = np.percentile(a, (1, 99))
    return np.clip((a.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)


def spatial(a, rng, crop):
    if crop:
        y, x = int(rng.integers(0, H - crop + 1)), int(rng.integers(0, W - crop + 1))
        a = np.rot90(a[y:y + crop, x:x + crop], int(rng.integers(4)))
    if rng.random() < .5:
        a = a[:, ::-1]
    if rng.random() < .5:
        a = a[::-1, :]
    return np.ascontiguousarray(np.clip(a * rng.uniform(.9, 1.1) + rng.uniform(-.03, .03), 0, 1))


class Views(Dataset):
    def __init__(self, order, epoch, a):
        self.order, self.epoch, self.a, self.raw = order, epoch, a, None
        self.P = OPTICS.get(a.optics)
        self.xfer = a.optics == 'transfer'; self.by_source = None

    def __len__(self):
        return len(self.order)

    def __getitem__(self, row):
        cv2.setNumThreads(1)
        if self.raw is None:
            self.raw = np.memmap(RAW, mode='r', dtype=np.uint8, shape=(N, H, W))
        ix = int(self.order[row]); r = np.asarray(self.raw[ix]); a = self.a
        base = None if self.P or self.xfer else clahe01(r)
        out = []
        if self.xfer and self.by_source is None:
            st = stats(); so = np.sort(self.order); self.by_source = {q: so[st['src'][so] == q] for q in (0, 1)}
        for v in range(a.views):
            rng = np.random.default_rng(np.random.SeedSequence([a.seed, self.epoch, ix, v]))
            if self.xfer:
                src = int(stats()['src'][ix])
                im = clahe01(transfer(r, ix, int(rng.choice(self.by_source[src])), src, rng))
            else:
                im = clahe01(perturb(r, rng, self.P)) if self.P else base
            out.append(spatial(im, rng, a.crop))
        return torch.from_numpy(np.stack(out))[:, None], ix


def to_gpu(x, crop):
    x = x.cuda(non_blocking=True).expand(-1, 3, -1, -1)
    return (x if crop else F.pad(x, (0, 16, 0, 20), mode='replicate')).contiguous()


def pool_tokens(t, pool):
    if pool == 'mean':
        return t.mean(1)
    s = (t.float().var(1, unbiased=False) + 1e-12).sqrt().to(t.dtype)
    return s if pool == 'std' else torch.cat([t.mean(1), s], -1)


def objective(flat, projector, sigreg, V, lamb):
    z = projector(flat.float()).reshape(-1, V, 128).transpose(0, 1)
    inv = (z - z.mean(0, keepdim=True)).square().mean()
    reg = sigreg(z)
    return (1 - lamb) * inv + lamb * reg, z, inv, reg


def cache_backward(enc, projector, sigreg, x, micro, a, check):
    """pilot.cache_backward with V views and a chosen pooling. Features without graph, loss on all of them, then replay the encoder with the exact RNG and the feature gradients."""
    def forward(part):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return pool_tokens(enc.forward_features(to_gpu(part, a.crop))['x_norm_patchtokens'], a.pool).float()
    feats, states = [], []
    with torch.no_grad():
        for part in x.split(micro):
            states.append(pilot.rng_state())
            feats.append(forward(part))
    leaf = torch.cat(feats).detach().requires_grad_(True)
    loss, z, inv, reg = objective(leaf, projector, sigreg, a.views, a.lamb)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite SSL objective')
    loss.backward()
    g = leaf.grad.detach(); after = pilot.rng_state(); err = 0.
    for i, part in enumerate(x.split(micro)):
        pilot.restore_rng(states[i])
        emb = forward(part)
        if check:
            err = max(err, (emb.detach() - feats[i]).abs().max().item())
            assert torch.allclose(emb.detach(), feats[i], atol=2e-3, rtol=2e-3), f'replay drift {err}'
        torch.autograd.backward(emb, g[i * micro:i * micro + len(part)])
    pilot.restore_rng(after)
    with torch.no_grad():
        f = leaf.reshape(-1, a.views, leaf.shape[-1]).mean(1)
        s = torch.linalg.svdvals(f - f.mean(0)); q = s.square() / s.square().sum().clamp_min(1e-12)
        rank = (-torch.where(q > 0, q * q.clamp_min(1e-12).log(), torch.zeros_like(q)).sum()).exp().item()
    return dict(loss=loss.item(), invariance=inv.item(), sigreg=reg.item(), feature_std=f.std(0).mean().item(), projector_std=z.std(dim=1).mean().item(), effective_rank=rank, replay_err=err)


def build(a):
    torch.set_num_threads(2)
    np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model, _, details = foundation.build('cpdino_vitb', False)
    enc = model[0].original.cuda().train()
    del model
    if a.init:
        c = torch.load(a.init, map_location='cpu', weights_only=True)
        enc.load_state_dict(c['encoder'], strict=True)
        del c
    if a.crop:
        orig = enc.rope_embed.forward; cache = {}
        def rope(*, H, W):
            s, c = orig(H=124, W=164)
            if (H, W) not in cache:
                cache[(H, W)] = (torch.arange(H, device=s.device)[:, None] * 164 + torch.arange(W, device=s.device)).flatten()
            i = cache[(H, W)]
            return s[i], c[i]
        enc.rope_embed.forward = rope
    d = 1536 if a.pool == 'both' else 768
    proj = nn.Sequential(nn.Linear(d, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Linear(2048, 128)).cuda().train()
    reg = pilot.SIGReg().cuda()
    opt = torch.optim.AdamW([{'params': enc.parameters(), 'lr': a.encoder_lr}, {'params': proj.parameters(), 'lr': 1e-3}], weight_decay=1e-4, fused=True)
    return enc, proj, reg, opt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('name'); p.add_argument('--fold', type=int, required=True); p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--crop', type=int, default=0); p.add_argument('--optics', default='none'); p.add_argument('--pool', default='mean'); p.add_argument('--views', type=int, default=2)
    p.add_argument('--micro', type=int, default=4); p.add_argument('--seed', type=int, default=1003); p.add_argument('--encoder-lr', type=float, default=1e-5); p.add_argument('--lamb', type=float, default=0.02)
    p.add_argument('--init', default=''); p.add_argument('--workers', type=int, default=4); p.add_argument('--smoke', type=int, default=0)
    a = p.parse_args()
    assert a.crop % 16 == 0 and a.pool in ('mean', 'std', 'both') and (a.optics in ('none', 'transfer') or a.optics in OPTICS)
    if a.fold == 99:  # all data, for the final fit: every frame, no validation
        tr, va = np.arange(N, dtype=np.int64), np.zeros(0, dtype=np.int64)
    else:
        with np.load(SPLITS[a.fold]) as z:
            tr, va = z['train'].copy().astype(np.int64), z['validation'].copy().astype(np.int64)
    assert not np.intersect1d(tr, va).size and len(tr) % 32 != 1
    root = LANE / 'ssl' / (a.name + ('_smoke' if a.smoke else '')); run = root / 'run'
    if (run / 'complete.json').exists():
        print('already complete', a.name, flush=True); return
    assert a.smoke or not (run / 'last.pt').exists(), 'no implicit resume'
    run.mkdir(parents=True, exist_ok=True)
    (root / 'config.json').write_text(json.dumps(dict(vars(a), train_n=len(tr), val_n=len(va), optics_params=OPTICS.get(a.optics), started_utc=time.strftime('%FT%TZ', time.gmtime())), indent=1))
    enc, proj, reg, opt = build(a)
    step, ref, t0 = 0, None, time.time(); log = (root / 'history.jsonl').open('a')
    for epoch in range(a.epochs):
        order = np.random.default_rng(a.seed + epoch).permutation(tr)
        dl = DataLoader(Views(order, epoch, a), batch_size=32, shuffle=False, num_workers=a.workers, pin_memory=True, prefetch_factor=2, drop_last=False)
        seen = 0; tick = time.time()
        for x, ix in dl:
            assert not np.intersect1d(ix.numpy(), va).size
            step += 1; r = min(1., step / 20)
            opt.param_groups[0]['lr'] = a.encoder_lr * r; opt.param_groups[1]['lr'] = 1e-3 * r
            opt.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
            v = cache_backward(enc, proj, reg, x.flatten(0, 1), a.micro, a, step == 1)
            norm = nn.utils.clip_grad_norm_(list(enc.parameters()) + list(proj.parameters()), 1., error_if_nonfinite=True)
            opt.step(); seen += len(ix)
            ref = ref or v['feature_std']
            if v['feature_std'] < max(1e-6, ref * 1e-3) or v['projector_std'] < 1e-6:
                raise ValueError('Representation collapse')
            if step == 1 or step % 10 == 0:
                rec = dict(epoch=epoch + 1, step=step, seen=seen, sec=round(time.time() - t0), peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1), grad=round(norm.item(), 3), **{k: round(q, 5) for k, q in v.items()})
                log.write(json.dumps(rec) + '\n'); log.flush(); print(json.dumps(rec), flush=True)
            if a.smoke and step >= a.smoke:
                print('smoke ok', a.name, round((time.time() - t0) / step, 2), 's/step', flush=True); return
        assert seen == len(tr)
        tmp = run / 'last.pt.tmp'
        torch.save(dict(encoder=enc.state_dict(), epoch=epoch + 1, step=step), tmp); tmp.replace(run / 'last.pt')
        torch.save(dict(encoder=enc.state_dict(), epoch=epoch + 1, step=step), run / f'epoch{epoch + 1:02d}.pt')
        print(json.dumps(dict(epoch_done=epoch + 1, sec=round(time.time() - tick))), flush=True)
    (run / 'complete.json').write_text(json.dumps(dict(status='completed', epochs=a.epochs, steps=step, seconds=round(time.time() - t0), finished_utc=time.strftime('%FT%TZ', time.gmtime()))))
    print('ssl done', a.name, round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
