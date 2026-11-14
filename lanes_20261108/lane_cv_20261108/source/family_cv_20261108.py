#!/usr/bin/env python3
"""CV lane copy (2026-11-08) of the CD lane runner, adds CD_LOSS=cvw3 (CV column x3); TARGET_SD=1 (11-10) trains column 1 on the cell-area SD = CV x 1e6 / CD instead of CV; CD_LOSS=lds (11-11) weights each sample by 1 / smoothed CV label density. CD lane copy (2026-11-08) of lane GPU 3's family runner: all five folds wired, fold 2's two HEX-zero validation rows scored on CD/CV only,
optional loss variant via env CD_LOSS (recipe | cdw3: CD column weighted x3 | logl1: log L1 only). Otherwise unchanged.

Recipe A budget on another encoder family, one GPU (lane GPU 3, 2026-11-06).

Only the encoder changes. Kept from the winner: fold-train-only LeJEPA SSL (5 epochs, seed 1003, mean pool, flip/gain/offset views, batch 32, encoder LR 1e-5),
then regression with std pool of the final normed patch tokens, LayerNorm -> Linear128 -> GELU -> Linear3, 20 epochs, schedule10, head LR 1e-3, encoder LR 1e-5,
AdamW wd 1e-4, global clip 1.0, effective batch 16, bf16, H/V flips, same split files, same order and flip seeds. Frozen modules supply the loss, metric, SIGReg,
gradient caching, flips and schedule. Epochs 16-20 keep weights; every epoch keeps its validation predictions in the reference schema.
Run from /workspace/clear-ec-research with one visible GPU."""
import argparse, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

sys.path[:0] = ['experiments/oof_recipeA_20261027/staging_regression_zero', 'experiments/phase2_crossfold_20261015/source']
import phase2_foundation_preflight_20261002 as pre
import phase2_foundation_train_20261002 as b
import phase2_lejepa_20261003 as pilot
import phase2_crossfold_ssl_20261015 as S
import phase2_pooling_20261019 as pooling
import phase2_backbone_ssl_regression_20261020 as R

# CV lane: CROP_LW=1 trains on a 544 x 412 crop centred on the legible window of box size (label-free, from pre-only peaks), flips after the crop.
if os.environ.get('CROP_LW'):
    _z = np.load('experiments/meta_ceiling_20261103/sweep/legible_window.npz', allow_pickle=True); _nm = list(_z['names'].astype(str))
    _CX = np.nan_to_num(_z['F'][:, _nm.index('top_box_x')].astype(float), nan=648.0); _CY = np.nan_to_num(_z['F'][:, _nm.index('top_box_y')].astype(float), nan=486.0)
    _CW, _CH = 544, 412
    class LWFrames(R.Frames):
        def __getitem__(self, row):
            x, yv, index = b.Frames.__getitem__(self, row); i = int(index)
            x0 = int(np.clip(round(_CX[i] - _CW / 2), 0, 1296 - _CW)); y0 = int(np.clip(round(_CY[i] - _CH / 2), 0, 972 - _CH)); x = x[..., y0:y0 + _CH, x0:x0 + _CW]
            if self.use_augmentation:
                h, v = R.flip_decisions(self.seed, self.epoch, index)
                if h: x = torch.flip(x, (-1,))
                if v: x = torch.flip(x, (-2,))
            return x.contiguous(), yv, index
    R.Frames = LWFrames

# CV lane: AUG_BUNDLE=1 adds the label-preserving perturbations of recipeA_invariance_20261104 during training (raw frame, before CLAHE):
# rotation +-10 deg (reflected border), gamma log-uniform [0.8, 1.25], Gaussian blur sigma [0.3, 2.0] with p 0.5, Gaussian noise sigma [1, 7] with p 0.5; flips as before.
if os.environ.get('AUG_BUNDLE'):
    import cv2 as _cv2
    def _perturb(a, rng, W=1296, H=972):
        ang = rng.uniform(-10.0, 10.0); a = _cv2.warpAffine(a, _cv2.getRotationMatrix2D(((W - 1) / 2, (H - 1) / 2), ang, 1.0), (W, H), flags=_cv2.INTER_LINEAR, borderMode=_cv2.BORDER_REFLECT_101)
        g = np.exp(rng.uniform(np.log(0.8), np.log(1.25))); a = _cv2.LUT(a, np.clip(255 * (np.arange(256) / 255.0) ** g, 0, 255).astype(np.uint8))
        if rng.random() < 0.5: a = _cv2.GaussianBlur(a, (0, 0), rng.uniform(0.3, 2.0))
        if rng.random() < 0.5: a = np.clip(a.astype(np.float32) + rng.normal(0, rng.uniform(1.0, 7.0), a.shape).astype(np.float32), 0, 255).astype(np.uint8)
        return a
    class AugFrames(R.Frames):
        def __getitem__(self, row):
            if not self.use_augmentation: return super().__getitem__(row)
            _cv2.setNumThreads(1)
            if self.raw is None: self.raw = np.memmap(b.RAW, mode='r', dtype=np.uint8, shape=(self.total, 972, 1296))
            index = int(self.indices[row]); a = _perturb(np.asarray(self.raw[index]), np.random.default_rng(np.random.SeedSequence([self.seed, 88, self.epoch, index])))
            a = _cv2.createCLAHE(2.0, (1296 // 32, 972 // 32)).apply(a); lo, hi = np.percentile(a, (1, 99)); a = np.clip((a.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
            x = torch.from_numpy(np.ascontiguousarray(a[None])); h, v = R.flip_decisions(self.seed, self.epoch, index)
            if h: x = torch.flip(x, (-1,))
            if v: x = torch.flip(x, (-2,))
            return x, torch.from_numpy(self.targets[index]), index
    R.Frames = AugFrames

LANE = Path('experiments/lane_cv_20261108')
SPLITS = {0: 'experiments/ssl_screen_20260911/preflight/fold0.npz', 1: 'experiments/ssl_screen_20260911/preflight/fold1.npz',
          2: 'experiments/overnight_target_20260912/preflight/fold2.npz', 3: 'experiments/overnight_target_20260912/preflight/fold3.npz',
          4: 'experiments/target_fold4_20260912/preflight/fold4.npz',
          99: 'experiments/fulldata_recipeA_20261028/fulldata_regression_split.npz'}  # full data as in the CD lane runner (train 8952, the two HEX-zero rows held)
CD_LOSS = os.environ.get('CD_LOSS', 'recipe')


def safe_metrics(pred, y):
    v = {}
    for j, k in enumerate(('CD', 'CV', 'HEX')):
        ok = y[:, j] > 0; v[k] = float((np.abs(pred[ok, j] - y[ok, j]) / y[ok, j]).mean() * 100) if ok.any() else -1.0   # -1 = no scorable rows (full-data split), as in the CD lane runner
    s = [v[k] for k in ('CD', 'CV', 'HEX') if v[k] >= 0]; v['mean'] = float(np.mean(s)) if s else -1.0; v['n'] = len(y); return v


LDS = {}   # CV lane 11-11: label-distribution smoothing (KG idea-imbalanced-regression-cv-tails)


def lds_init(cv_train):
    """Per-sample weight = 1 / Gaussian-KDE density of the training rows' log CV (Silverman bandwidth), mean 1 over the training rows,
    capped at 5 (renormalise and cap alternately until both hold). Fixed a priori; nothing chosen on reported rows."""
    v = np.log(np.asarray(cv_train, np.float64)); h = 1.06 * v.std() * len(v) ** (-0.2); g = np.linspace(v.min() - 3 * h, v.max() + 3 * h, 1024)
    dens = np.zeros_like(g)
    for c in np.array_split(v, max(1, len(v) // 512)): dens += np.exp(-0.5 * ((g[:, None] - c[None, :]) / h) ** 2).sum(1)
    w = 1.0 / np.maximum(dens, 1e-12)
    for _ in range(50):   # mean 1 over the training rows and max 5: alternate renormalise / cap until both hold
        w = w / np.interp(v, g, w).mean()
        if np.interp(v, g, w).max() <= 5.0 + 1e-6: break
        w = np.minimum(w, 5.0)
    LDS.update(g0=float(g[0]), dg=float(g[1] - g[0]), w=torch.tensor(w, dtype=torch.float32, device='cuda'), h=float(h), wmax=float(np.interp(v, g, w).max()))
    print(json.dumps(dict(lds_bandwidth=round(h, 4), weight_max=round(LDS['wmax'], 3), weight_p50=round(float(np.median(np.interp(v, g, w))), 3))), flush=True)


def lane_loss(z, targets, mu, sigma):
    if CD_LOSS == 'recipe':
        return b.loss_fn(z, targets, mu, sigma)
    zt = (targets.log() - mu) / sigma; d = z.float() - zt
    if CD_LOSS == 'cdw3':
        w = d.new_tensor([3., 1., 1.]) / 5. * 3.
        return (0.5 * d.abs() * w).mean() + (2.5 * torch.expm1(d * sigma).abs() * w).mean()
    if CD_LOSS == 'cvw3':
        w = d.new_tensor([1., 3., 1.]) / 5. * 3.
        return (0.5 * d.abs() * w).mean() + (2.5 * torch.expm1(d * sigma).abs() * w).mean()
    if CD_LOSS == 'lds':   # recipe loss with per-sample weights from the CV label density (LDS)
        i = ((targets[:, 1].float().log() - LDS['g0']) / LDS['dg']).round().long().clamp(0, len(LDS['w']) - 1); w = LDS['w'][i]
        return ((0.5 * d.abs() + 2.5 * torch.expm1(d * sigma).abs()) * w[:, None]).mean()
    if CD_LOSS == 'logl1':
        return (d * sigma).abs().mean() * 5.
    if CD_LOSS == 'order':   # CV lane 11-09: recipe loss + 1.0 x noise-weighted pairwise order loss on CV (KG lee2026stochorder; s = 0.065 log units)
        base = b.loss_fn(z, targets, mu, sigma)
        if len(z) < 2: return base
        s1 = sigma[1].float(); dp = (z[:, 1].float()[:, None] - z[:, 1].float()[None, :]) * s1; dt = (zt[:, 1][:, None] - zt[:, 1][None, :]) * s1
        p = 0.5 * (1 + torch.erf(dt / (2 * 0.065))); m = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
        return base + torch.nn.functional.binary_cross_entropy_with_logits(dp[m] / (math.sqrt(2) * 0.065), p[m])
    raise ValueError(CD_LOSS)
DV3B = 'experiments/phase2_foundation_20261002/weights/dinov3-vitb16'
DV3L = 'experiments/lane_gpu3_20261106/weights/dinov3-vitl16'
HUB = 'experiments/phase2_backbone_screen_20261001/hf_cache/hub'
CONVNEXT = HUB + '/models--timm--convnext_base.fb_in22k_ft_in1k/snapshots/67758fcbcf9f007ec7630bf8636b74a8f24e52a0/model.safetensors'
DINOV2 = HUB + '/models--timm--vit_base_patch14_dinov2.lvd142m/snapshots/4685c99dabffe5affac90bd99dbffd25801ae58d/model.safetensors'
MEAN, STD = (.485, .456, .406), (.229, .224, .225)
FAMILIES = {
    'cpdino_b8': dict(kind='cpdino', micro=4, norm=False, pad=(16, 20)),
    'dinov3_b16': dict(kind='hf', path=DV3B, stride=16, micro=16, norm=True, pad=(16, 20)),
    'dinov3_b16_s8': dict(kind='hf', path=DV3B, stride=8, micro=4, norm=True, pad=(16, 20)),
    'dinov3_l16': dict(kind='hf', path=DV3L, stride=16, micro=8, norm=True, pad=(16, 20)),
    'convnext_b32': dict(kind='convnext', micro=16, norm=True, pad=(16, 20)),
    'dinov2_b14': dict(kind='dinov2', micro=16, norm=True, pad=(6, 8)),
    'cpsam_c512': dict(kind='cpsam', micro=4, norm=False, pad=(0, 0), crop=512),
}


class CPDINOTokens(nn.Module):
    def __init__(self):
        super().__init__()
        net, _, _ = pre.build('cpdino_vitb', False)
        self.original, self.dim = net[0].original, 768

    def forward(self, x):
        return self.original.forward_features(x)['x_norm_patchtokens']


class HFTokens(nn.Module):
    def __init__(self, path, stride):
        super().__init__()
        from transformers import AutoModel
        enc = AutoModel.from_pretrained(path, local_files_only=True, dtype=torch.float32)
        if stride != enc.config.patch_size:
            pe = enc.embeddings.patch_embeddings
            assert pe.kernel_size == (16, 16), pe.kernel_size
            pe.stride, pe.padding = (stride, stride), ((16 - stride) // 2,) * 2
            enc.config.patch_size = stride
            enc.rope_embeddings.config.patch_size = stride
        self.original, self.prefix, self.dim, self.stride = enc, 1 + enc.config.num_register_tokens, enc.config.hidden_size, stride

    def forward(self, x):
        t = self.original(pixel_values=x).last_hidden_state[:, self.prefix:]
        assert t.shape[1] == (x.shape[-2] // self.stride) * (x.shape[-1] // self.stride), t.shape
        return t


class ConvNeXtTokens(nn.Module):
    """Native stride 32, grid 31x41; the pretrained head norm stays, so these are the final normed tokens as on the ViTs."""
    def __init__(self):
        super().__init__()
        import timm
        from safetensors.torch import load_file
        m = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=False)
        m.load_state_dict(load_file(CONVNEXT), strict=True); m.reset_classifier(0)
        self.original, self.dim = m, 1024

    def forward(self, x):
        t = self.original.head.norm(self.original.forward_features(x)).flatten(2).transpose(1, 2)
        assert t.shape[1] == (x.shape[-2] // 32) * (x.shape[-1] // 32), t.shape
        return t


class DINOv2Tokens(nn.Module):
    """Patch 14, grid 70x93 after padding to 980x1302; the position embedding is resampled at each forward and stays trainable."""
    def __init__(self):
        super().__init__()
        import timm
        from safetensors.torch import load_file
        m = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, dynamic_img_size=True)
        m.load_state_dict(load_file(DINOV2), strict=True)
        self.original, self.prefix, self.dim = m, m.num_prefix_tokens, 768

    def forward(self, x):
        t = self.original.forward_features(x)[:, self.prefix:]
        assert t.shape[1] == (x.shape[-2] // 14) * (x.shape[-1] // 14), t.shape
        return t


class CPSAMTokens(nn.Module):
    """CPSAM v2 ViT-L, kernel 8, global attention, loaded and run as the frozen build does, with checkpointing in training; tokens are the neck output, 256-d, grid 64x64 on a 512 crop."""
    def __init__(self):
        super().__init__()
        from cellpose.vit import CPSAM
        enc = CPSAM().encoder
        state = torch.load('cpsam_models/cpsam_v2', map_location='cpu', weights_only=True)
        enc.load_state_dict({k[len('encoder.'):]: v for k, v in state.items() if k.startswith('encoder.')}, strict=True)
        self.original, self.dim = enc, 256

    def forward(self, x):
        e = self.original; t = e.patch_embed(x); pos = e.pos_embed
        if pos.shape[1:3] != t.shape[1:3]:
            pos = nn.functional.interpolate(pos.permute(0, 3, 1, 2), size=t.shape[1:3], mode='bicubic', align_corners=False).permute(0, 2, 3, 1)
        t = t + pos
        for block in e.blocks:
            t = checkpoint(block, t, use_reentrant=False) if self.training else block(t)
        t = e.neck(t.permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)
        assert t.shape[1] == (x.shape[-2] // 8) * (x.shape[-1] // 8), t.shape
        return t


def build_tokens(name):
    f = FAMILIES[name]
    if f['kind'] == 'cpsam':
        return CPSAMTokens()
    if f['kind'] == 'convnext':
        return ConvNeXtTokens()
    if f['kind'] == 'dinov2':
        return DINOv2Tokens()
    return CPDINOTokens() if f['kind'] == 'cpdino' else HFTokens(f['path'], f['stride'])


class Pooled(nn.Module):
    def __init__(self, tokens, mode):
        super().__init__()
        self.tokens, self.mode = tokens, mode

    def forward(self, x):
        return pooling.pool_tokens(self.tokens(x), self.mode)


def inputs(x, name):
    f = FAMILIES[name]
    x = x.cuda(non_blocking=True).expand(-1, 3, -1, -1)
    if f.get('crop'):
        c = f['crop']; top, left = (x.shape[-2] - c) // 2, (x.shape[-1] - c) // 2
        x = x[..., top:top + c, left:left + c]
    x = torch.nn.functional.pad(x, (0, f['pad'][0], 0, f['pad'][1]), mode='replicate')
    if f['norm']:
        x = (x - x.new_tensor(MEAN)[None, :, None, None]) / x.new_tensor(STD)[None, :, None, None]
    return x.contiguous()


def arrays(fold):
    with np.load(b.META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    if os.environ.get('TARGET_SD'):   # CV lane 11-10: column 1 becomes the cell-area SD (um^2) = CV x 1e6 / CD (CV = SD/mean area, mean area = 1e6/CD); CD and HEX unchanged
        ok = (y[:, 0] > 0) & (y[:, 1] > 0); y = y.copy(); y[ok, 1] = (y[ok, 1].astype(np.float64) * 1e6 / y[ok, 0].astype(np.float64)).astype(np.float32)
    with np.load(SPLITS[fold]) as z:
        tr, va = z['train'].astype(np.int64), z['validation'].astype(np.int64)
    assert not np.intersect1d(tr, va).size and (y[tr] > 0).all() and (y[va, 0] > 0).all() and (y[va, 1] > 0).all()
    return ids, y, tr, va


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True


def log(obj):
    print(json.dumps(obj), flush=True)


def run_ssl(a, out, max_steps=None):
    """LeJEPA exactly as phase2_crossfold_ssl_20261015, encoder swapped; micro only groups the gradient replay."""
    if (out / 'complete.json').exists():
        return out / 'last.pt'
    micro = a.micro or FAMILIES[a.family]['micro']
    ids, _, tr, va = arrays(a.fold)
    seed = 1003; S.seed_all(seed); torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = True
    tok = build_tokens(a.family); enc = Pooled(tok, 'mean').cuda().train()
    proj = nn.Sequential(nn.Linear(tok.dim, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Linear(2048, 128)).cuda().train()
    reg = pilot.SIGReg().cuda(); rc = pilot.RECIPE
    opt = torch.optim.AdamW([{'params': enc.parameters(), 'lr': rc['encoder_lr']}, {'params': proj.parameters(), 'lr': rc['projector_lr']}], weight_decay=rc['weight_decay'], fused=True)
    def fwd(e, part):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return e(inputs(part, a.family))
    out.mkdir(parents=True, exist_ok=True); step = 0; t0 = time.monotonic(); ref = None; hist = []
    for epoch in range(1, a.ssl_epochs + 1):
        order = np.random.default_rng(seed + epoch - 1).permutation(tr)
        dl = DataLoader(S.Views(order, len(ids), epoch - 1, seed), batch_size=32, shuffle=False, num_workers=a.workers, pin_memory=True, prefetch_factor=2, drop_last=False)
        for i, (x, ix) in enumerate(dl, 1):
            assert not np.intersect1d(ix.numpy(), va).size
            ratio = min(1., (step + 1) / rc['warmup_steps'])
            opt.param_groups[0]['lr'] = rc['encoder_lr'] * ratio; opt.param_groups[1]['lr'] = rc['projector_lr'] * ratio
            opt.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
            v = pilot.cache_backward(enc, proj, reg, x.flatten(0, 1), micro, fwd, i == 1)
            norm = nn.utils.clip_grad_norm_(list(enc.parameters()) + list(proj.parameters()), 1., error_if_nonfinite=True)
            if norm.item() <= 0:
                raise ValueError('Zero gradient')
            opt.step(); step += 1
            ref = v['feature_std'] if ref is None else ref
            if v['feature_std'] < max(1e-6, ref * 1e-3) or v['projector_std'] < 1e-6:
                raise ValueError('Representation collapse')
            if i == 1 or i % 25 == 0:
                item = dict(stage='ssl', epoch=epoch, step=step, loss=v['loss'], invariance=v['invariance'], sigreg=v['sigreg'], feature_std=v['feature_std'], rank=v['effective_rank'],
                            grad=norm.item(), seconds=time.monotonic() - t0, peak_gib=torch.cuda.max_memory_allocated() / 2**30, replay=v['replay_max_abs_error'])
                hist.append(item); log(item)
            if max_steps and step >= max_steps:
                return dict(seconds_per_step=(time.monotonic() - t0) / step, peak_gib=torch.cuda.max_memory_allocated() / 2**30, last=v)
    tmp = out / 'last.pt.tmp'
    torch.save(dict(tokens=tok.state_dict(), epoch=a.ssl_epochs, step=step, family=a.family, fold=a.fold), tmp); tmp.replace(out / 'last.pt')
    pre.save(out / 'complete.json', dict(status='completed', family=a.family, fold=a.fold, epochs=a.ssl_epochs, steps=step, seconds=time.monotonic() - t0, validation_images_used=0, target_arrays_loaded=False, history=hist))
    del enc, tok, proj, reg, opt; torch.cuda.empty_cache()
    return out / 'last.pt'


def evaluate(net, name, dl, mu, sigma, expected):
    net.eval(); preds, seen = [], []
    with torch.inference_mode():
        for x, _, ix in dl:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = net(inputs(x, name)).float()
            preds.append(torch.exp(z * sigma + mu).cpu().numpy()); seen.append(ix.numpy())
    np.testing.assert_array_equal(np.concatenate(seen), expected)
    return np.concatenate(preds)


def build_net(a, seed, ssl_path):
    seed_everything(seed)
    tok = build_tokens(a.family)
    net = nn.Sequential(Pooled(tok, 'std'), nn.LayerNorm(tok.dim), nn.Linear(tok.dim, 128), nn.GELU(), nn.Linear(128, 3))
    torch.manual_seed(seed)
    nn.init.normal_(net[-1].weight, std=1e-3); nn.init.zeros_(net[-1].bias)
    if ssl_path is not None:
        ck = torch.load(ssl_path, map_location='cpu', weights_only=True)
        tok.load_state_dict(ck['tokens'] if 'tokens' in ck else {'original.' + k: v for k, v in ck['encoder'].items()}, strict=True); del ck
    return net.cuda()


def run_regression(a, root, ssl_path, max_updates=None):
    micro = a.micro or FAMILIES[a.family]['micro']
    run = root / 'runs' / f'fold{a.fold}' / a.family
    if not max_updates:
        if (run / 'complete.json').exists():
            return
        if run.exists():
            raise FileExistsError(f'No overwrite or implicit resume: {run}')
        run.mkdir(parents=True)
    ids, y, tr, va = arrays(a.fold); n = len(ids)
    if CD_LOSS == 'lds': lds_init(y[tr, 1])
    net = build_net(a, a.seed, ssl_path)
    opt = torch.optim.AdamW([{'params': net[0].parameters(), 'lr': 1e-5}, {'params': list(net[1:].parameters()), 'lr': 1e-3}], weight_decay=1e-4, fused=True)
    logy = np.log(y[tr].astype(np.float64)); mu = torch.tensor(logy.mean(0), device='cuda', dtype=torch.float32); sigma = torch.tensor(logy.std(0), device='cuda', dtype=torch.float32)
    R.ARGS = argparse.Namespace(seed=a.seed)
    updates = math.ceil(len(tr) / 16); step = 0; history = []; wall = time.monotonic()
    def eval_loader(rows):
        return DataLoader(R.Frames(rows, y, n, 0, False, a.seed), batch_size=micro, shuffle=False, num_workers=a.workers, pin_memory=True, prefetch_factor=2, drop_last=False)
    val = eval_loader(va); probe = np.sort(np.random.default_rng(12012).choice(tr, min(256, len(tr)), replace=False)); pl = eval_loader(probe)
    for epoch in range(a.epochs):
        order = np.random.default_rng(a.seed + epoch).permutation(tr)
        dl = DataLoader(R.Frames(order, y, n, epoch, True, a.seed), batch_size=16, shuffle=False, num_workers=a.workers, pin_memory=True, prefetch_factor=2, drop_last=False)
        net.train(); tick = time.monotonic(); torch.cuda.reset_peak_memory_stats(); stats = torch.zeros(4, dtype=torch.float64, device='cuda'); nsum = nmax = clipped = 0.; seen = 0
        for x, t, ix in dl:
            ratio = R.lr_ratio(step, updates, a.schedule_epochs)
            opt.param_groups[0]['lr'] = 1e-5 * ratio; opt.param_groups[1]['lr'] = 1e-3 * ratio
            opt.zero_grad(set_to_none=True); m = len(t)
            for s in range(0, m, micro):
                xx, yy = x[s:s + micro], t[s:s + micro].cuda(non_blocking=True); c = len(yy)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z = net(inputs(xx, a.family))
                    loss = lane_loss(z, yy, mu, sigma)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                with torch.no_grad():
                    rel = torch.expm1((z.float() - (yy.log() - mu) / sigma) * sigma).abs()
                    stats[0] += loss.detach().double() * c; stats[1:4] += rel.double().sum(0) * 100
                (loss * (c / m)).backward()
            norm = nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True).item()
            nsum += norm; nmax = max(nmax, norm); clipped += norm > 1.
            opt.step(); step += 1; seen += m
            if step % 25 == 0 or step == 1:
                log(dict(status='training', epoch=epoch + 1, updates=step, seen=seen, gradient_norm=norm, encoder_lr=opt.param_groups[0]['lr'], head_lr=opt.param_groups[1]['lr'],
                         seconds_this_epoch=time.monotonic() - tick, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, loss=(stats[0] / seen).item()))
            if max_updates and step >= max_updates:
                return dict(seconds_per_update=(time.monotonic() - tick) / step, peak_gib=torch.cuda.max_memory_allocated() / 2**30, loss=(stats[0] / seen).item())
        assert seen == len(tr)
        train_seconds = time.monotonic() - tick
        pred = evaluate(net, a.family, val, mu, sigma, va); score = safe_metrics(pred, y[va])
        pp = evaluate(net, a.family, pl, mu, sigma, probe); ps = safe_metrics(pp, y[probe])
        vals = (stats / len(tr)).cpu().tolist()
        item = dict(epoch=epoch + 1, score=score, train_seconds=train_seconds, epoch_seconds=time.monotonic() - tick, online_train_loss=vals[0], online_train_mape=dict(zip(('CD', 'CV', 'HEX'), vals[1:4])),
                    train_probe_score=ps, gradient_norm_mean=nsum / updates, gradient_norm_max=nmax, gradient_clip_fraction=clipped / updates, updates=step, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        history.append(item); pre.save(run / 'history.json', history); log(dict(epoch_complete=epoch + 1, **item))
        np.savez_compressed(run / f'predictions_epoch_{epoch + 1:02d}.npz', indices=va, ids=ids[va], target=y[va], pred=pred)
        if epoch + 1 > a.epochs - 5:
            torch.save(dict(model=net.state_dict(), epoch=epoch + 1, family=a.family, fold=a.fold, seed=a.seed, log_mean=logy.mean(0).tolist(), log_std=logy.std(0).tolist()), run / f'epoch_{epoch + 1:02d}.pt')
    last5 = [h['score'] for h in history[-5:]]
    pre.save(run / 'result.json', dict(family=a.family, fold=a.fold, seed=a.seed, terminal=history[-1]['score'], last5_mean={k: float(np.mean([s[k] for s in last5])) for k in ('CD', 'CV', 'HEX', 'mean')},
                                       seconds=time.monotonic() - wall, ssl=str(ssl_path), recipe=dict(epochs=a.epochs, schedule_epochs=a.schedule_epochs, micro=micro, pool='std')))
    pre.save(run / 'complete.json', dict(status='completed'))


def parity(a):
    """Runner check: reference epoch 20 weights through this eval path must return the reference score; then its first updates must show the reference gradient norms."""
    ref = Path(a.parity_ref); f = a.fold; ids, y, tr, va = arrays(f); n = len(ids)
    ck = torch.load(ref / 'runs' / f'fold{f}' / 'cpdino_vitb' / 'epoch_20.pt', map_location='cpu', weights_only=True)
    net = build_net(a, a.seed, None); state = {(k.replace('0.original.', '0.tokens.original.', 1) if k.startswith('0.original.') else k): v for k, v in ck['model'].items()}
    net.load_state_dict(state, strict=True)
    mu = torch.tensor(ck['log_mean'], device='cuda', dtype=torch.float32); sigma = torch.tensor(ck['log_std'], device='cuda', dtype=torch.float32); R.ARGS = argparse.Namespace(seed=a.seed)
    logy = np.log(y[tr].astype(np.float64)); assert np.array_equal(logy.mean(0), ck['log_mean'])
    dl = DataLoader(R.Frames(va, y, n, 0, False, a.seed), batch_size=4, shuffle=False, num_workers=a.workers, pin_memory=True, prefetch_factor=2)
    s = b.metrics(evaluate(net, a.family, dl, mu, sigma, va), y[va]); h = json.loads((ref / 'audit.json').read_text())['score']
    log(dict(parity_eval=s, reference=h)); assert abs(s['mean'] - h['mean']) < 1e-3, (s, h)


def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['all', 'smoke', 'parity']); p.add_argument('--family', required=True, choices=list(FAMILIES)); p.add_argument('--fold', type=int, required=True, choices=list(SPLITS))
    p.add_argument('--seed', type=int, default=46); p.add_argument('--root', type=Path); p.add_argument('--ssl-epochs', type=int, default=5); p.add_argument('--ssl-from', type=Path)
    p.add_argument('--epochs', type=int, default=20); p.add_argument('--schedule-epochs', type=int, default=10); p.add_argument('--micro', type=int, default=0); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--parity-ref'); p.add_argument('--updates', type=int, default=8)
    a = p.parse_args()
    assert torch.cuda.device_count() == 1, 'one visible GPU'
    if a.command == 'parity':
        parity(a)
        return
    if a.command == 'smoke':
        s = run_ssl(a, a.root / 'smoke_ssl', max_steps=3) if a.ssl_epochs else None
        torch.cuda.empty_cache()
        r = run_regression(a, a.root, a.ssl_from, max_updates=a.updates)
        log(dict(smoke=a.family, ssl=s, regression=r))
        return
    a.root.mkdir(parents=True, exist_ok=True)
    if a.ssl_from is not None:
        ssl = a.ssl_from
    elif a.ssl_epochs:
        ssl = run_ssl(a, a.root / 'ssl')
    else:
        ssl = None
    pre.save(a.root / 'launch.json', dict(family=a.family, fold=a.fold, seed=a.seed, ssl=str(ssl), ssl_epochs=a.ssl_epochs, epochs=a.epochs, schedule_epochs=a.schedule_epochs, micro=a.micro or FAMILIES[a.family]['micro'], cd_loss=CD_LOSS, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), pid=os.getpid(), started_utc=time.strftime('%FT%TZ', time.gmtime())))
    run_regression(a, a.root, ssl)


if __name__ == '__main__':
    main()
