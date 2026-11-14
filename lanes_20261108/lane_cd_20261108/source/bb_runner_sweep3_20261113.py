#!/usr/bin/env python3
# 2026-11-13 CD lane sweep copy of lane_backbone_20261108/source/bb_runner_20261108.py: adds --batch (effective batch, default 16; schedule updates follow it) and --clip (max grad norm, default 1.0, 0 = off). Nothing else changed.
# sweep3 (11-13 23:35): adds split ids 301 / 302 (fold-0 source specialists). sweep2 (11-13 23:15): adds --target-w W1,W2,W3 (per-metric weights of the recipe loss, weighted mean over CD/CV/HEX; empty = recipe). Copy of bb_runner_sweep_20261113.py, which a running job uses.
"""Backbone lane runner (2026-11-08). Recipe A regression (CPDINO ViT-B kernel 16 stride 8, std pool, LN-128-GELU-3 head,
schedule10, H/V flips, effective batch 16, AdamW wd 1e-4, clip 1, bf16) started from the fold's SSL5 checkpoint, with
switchable techniques. Every switch off = recipe A through the lane GPU 3 family runner (parity-checked there).

  --drop-path P     per-sample drop of each block's whole residual, linear 0..P over depth, kept rows rescaled 1/(1-p)
  --llrd G          layer-wise LR decay: last block and final norm get --enc-lr, each block below x G, embeddings x G^L
  --enc-lr/--head-lr
  --ema D1,D2       EMA weight tracks (fp32, every update), started by copying the weights at --ema-start (default epoch 3)
  --swa-start E     uniform average of weights over every update from epoch E on
  --aux FILE        label-free auxiliary targets, npz {indices, Z (N,k)}; extra head outputs, loss weight --aux-w
  --teacher FILE    soft targets, npz {indices, pred (N,3)}; target = exp((1-a) log y + a log t), a = --teacher-a
  --rope-fixed      switch off DINOv3's train-mode RoPE coordinate rescale
  --row-weight FILE per-row loss weights, npz {indices, Z (N,)}; loss = mean(w * per-row recipe loss) / mean training weight
  --transpose-p P   per batch, transpose the frames (portrait input) with probability P; labels are rotation invariant
  --zoom Z          per batch, resize by s, log-uniform in [1/Z, Z]; CD label x 1/s^2 (CD = 1e6 / mean cell area), CV and HEX kept
  --appear P        stacked appearance augmentation on the raw frame before recipe A's CLAHE + percentile (zhang2020bigaug style,
                    no geometry): gamma, contrast, blur-or-sharpen, Gaussian noise, each with probability P; training rows only
  --shading         flat-field correction of the raw frame before recipe A's CLAHE + percentile, on every row (training and
                    evaluation): divide by a heavily blurred copy (Gaussian sigma 64 px, via 1/8 downscale), floor 8 grey levels,
                    rescaled to the frame's mean background; fixed constants, not tuned
  --window FILE     region view: crop --window-size (w,h) around a label-free per-image centre (npz {indices, Z=(cx, cy)}, frame
                    centre where missing), jittered +-(--window-jitter) px in training, flipped with the frame, upsampled x --window-scale

Every track (raw, ema<D>, swa) is written like a separate run, so read tools treat each as one model:
  <root>/<track>/runs/fold<f>/cpdino_b8/{history.json, predictions_epoch_XX.npz, final.pt, result.json}
Run from /workspace/clear-ec-research with one visible GPU."""
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, 'experiments/lane_backbone_20261108/source')
import family_bb_20261108 as F  # noqa: E402  lane copy of lane GPU 3's family runner (+ timm DINOv3 ViT-L); puts the frozen recipe modules on sys.path

b, R, pre = F.b, F.R, F.pre
SSL5 = {0: 'experiments/phase2_crossfold_20261015/fold0/ssl5/run/last.pt', 4: 'experiments/phase2_crossfold_20261015/fold4/ssl5/run/last.pt',
        **{f: f'experiments/oof_recipeA_20261027/fold{f}/ssl5/run/last.pt' for f in (1, 2, 3)}}
SPLITS = {0: 'experiments/ssl_screen_20260911/preflight/fold0.npz', 1: 'experiments/ssl_screen_20260911/preflight/fold1.npz',
          2: 'experiments/overnight_target_20260912/preflight/fold2.npz', 3: 'experiments/overnight_target_20260912/preflight/fold3.npz',
          4: 'experiments/target_fold4_20260912/preflight/fold4.npz'}  # the recipe A OOF split files
# leave-one-source-out shift splits (2026-11-10): 201 = train P1 ('NNNN-YY') minus hold-out, 202 = train P2 ('NNNN-NNNN aaaa') minus hold-out;
# validation = own-source hold-out + all rows of the other source (source defines the split only, never an input)
SPLITS.update({201: 'experiments/lane_backbone_20261108/data/split_srcA_20261110.npz', 202: 'experiments/lane_backbone_20261108/data/split_srcB_20261110.npz'})
SPLITS.update({301: "experiments/lane_cd_20261108/data/split_f0_srcold_20261113.npz", 302: "experiments/lane_cd_20261108/data/split_f0_srcnew_20261113.npz"})  # sweep3: fold-0 source specialists (train = own-source fold-0 training rows; validation = all fold-0 validation rows)
FAMILY = 'cpdino_b8'  # set from --family in main


def log(o):
    print(json.dumps(o), flush=True)


def arrays(fold):
    """fold -1 = full data for the rebuild: every frame with three positive labels trains; 'validation' is a fixed 256-row
    in-sample monitor (no held-out score exists in this mode)."""
    with np.load(b.META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    if fold == -1:
        tr = np.where(np.isfinite(y).all(1) & (y > 0).all(1))[0].astype(np.int64)
        va = np.sort(np.random.default_rng(12013).choice(tr, 256, replace=False))
        return ids, y, tr, va
    with np.load(SPLITS[fold]) as z:
        tr, va = z['train'].astype(np.int64), z['validation'].astype(np.int64)
    # training labels must be positive; validation rows with a zero label (fold 2 has two HEX = 0 rows) stay in, are predicted,
    # and are left out of that metric's score, as the official metric does
    assert not np.intersect1d(tr, va).size and (y[tr] > 0).all() and np.isfinite(y[va]).all()
    return ids, y, tr, va


def mape_score(pred, yy):
    """Mean absolute percentage error per metric over rows with a positive label (official rule: zero labels not scored)."""
    assert pred.shape == yy.shape and np.isfinite(pred).all()
    ok = yy > 0; ape = np.where(ok, np.abs(pred - yy) / np.where(ok, yy, 1.0), np.nan) * 100
    v = np.nanmean(ape, 0)
    return dict(CD=float(v[0]), CV=float(v[1]), HEX=float(v[2]), mean=float(v.mean()), n=len(yy), unscored=int((~ok).sum()))


def side_table(path, n, k_expected=None):
    """npz {indices, <key>} -> dense (n, k) float32 table, NaN where missing."""
    with np.load(path, allow_pickle=True) as z:
        key = 'Z' if 'Z' in z.files else 'pred'
        ix, v = z['indices'].astype(np.int64), z[key].astype(np.float32)
    v = v.reshape(len(v), -1); out = np.full((n, v.shape[1]), np.nan, np.float32); out[ix] = v
    if k_expected:
        assert v.shape[1] == k_expected
    return out


def enc_blocks(enc):
    """Transformer block list: timm / CPDINO models keep it in .blocks, HF DINOv3ViTModel in .layer."""
    return enc.blocks if hasattr(enc, 'blocks') else enc.layer


def add_drop_path(enc, pmax):
    blocks = enc_blocks(enc); L = len(blocks); rates = []
    for i, blk in enumerate(blocks):
        p = pmax * i / max(1, L - 1); rates.append(round(p, 4))
        if p <= 0:
            continue
        def hook(mod, inp, out, p=p):
            if not mod.training:
                return out
            single = not isinstance(out, list)
            xs, ys = ([inp[0]], [out]) if single else (inp[0], out)
            new = []
            for x, y in zip(xs, ys):
                keep = (torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), device=x.device) >= p).to(y.dtype) / (1 - p)
                new.append(x + (y - x) * keep)
            return new[0] if single else new
        blk.register_forward_hook(hook)
    return rates


def param_groups(net, a, L):
    groups = {}
    for name, p in net.named_parameters():
        if not name.startswith('0.'):
            key, lr = 'head', a.head_lr
        else:
            if '.blocks.' in name:
                lid = int(name.split('.blocks.')[1].split('.')[0]) + 1
            elif '.original.layer.' in name:  # HF DINOv3ViTModel
                lid = int(name.split('.original.layer.')[1].split('.')[0]) + 1
            elif any(s in name for s in ('patch_embed', 'cls_token', 'storage_tokens', 'reg_token', 'mask_token', 'rope', 'pos_embed', '.embeddings.')):
                lid = 0
            else:
                lid = L + 1
            key, lr = f'enc{lid}', a.enc_lr * (a.llrd ** (L - min(lid, L)) if a.llrd else 1.0)  # last block and final norm at enc_lr
        groups.setdefault(key, dict(params=[], base_lr=lr, lr=lr, name=key))['params'].append(p)
    return list(groups.values())


def swap(params, src):
    with torch.no_grad():
        backup = [q.detach().clone() for q in params]; torch._foreach_copy_(params, src)
    return backup


def restore(params, backup):
    with torch.no_grad():
        torch._foreach_copy_(params, backup)


class Track:
    """Averaged copy of the parameters: EMA with decay d, or a uniform running mean (d = None)."""
    def __init__(self, name, params, decay=None, start_epoch=0):
        self.name, self.decay, self.start, self.n = name, decay, start_epoch, 0
        self.avg = [p.detach().clone().float() for p in params]

    def update(self, params, epoch):
        if epoch < self.start:
            return
        self.n += 1
        # both kinds start by copying the weights at their start epoch (n == 1); EMA then uses its fixed decay
        w = (1 - self.decay) if self.decay is not None else 1.0 / self.n
        with torch.no_grad():
            if self.n == 1:
                torch._foreach_copy_(self.avg, [p.detach().float() for p in params]); return
            torch._foreach_lerp_(self.avg, [p.detach().float() for p in params], w)

    def active(self, epoch, eval_from=1):
        return epoch >= max(self.start, eval_from) and self.n > 0


class AppearFrames(R.Frames):
    """Stacked appearance augmentation (zhang2020bigaug style, no geometry change) on the raw uint8 frame, then recipe A's CLAHE +
    percentile and R.Frames' flips. Each op with probability p, magnitudes uniform in fixed ranges (not tuned): gamma 1/1.5..1.5
    (log-uniform), contrast about the frame mean 1/1.5..1.5 (log-uniform), Gaussian blur sigma 0.5..1.5 or unsharp-mask sharpening
    amount 0.5..1.5 (sigma 1, equal odds), additive Gaussian noise sigma 1..6 grey levels. Draws keyed by (seed, epoch, index).
    Evaluation rows (augment False) take R.Frames unchanged."""
    def __init__(self, indices, targets, total, epoch, augment, seed, p):
        super().__init__(indices, targets, total, epoch, augment, seed); self.p = p

    def __getitem__(self, row):
        if not self.use_augmentation:
            return R.Frames.__getitem__(self, row)
        import cv2
        cv2.setNumThreads(1)
        if self.raw is None:
            self.raw = np.memmap(b.RAW, mode='r', dtype=np.uint8, shape=(self.total, 972, 1296))
        idx = int(self.indices[row])
        g = np.random.default_rng(np.random.SeedSequence([self.seed, 4242, self.epoch, idx]))
        u, v = g.random(4), g.random(5)  # all decisions drawn up front
        a = np.asarray(self.raw[idx]).astype(np.float32)
        if u[0] < self.p:
            a = 255. * (a / 255.) ** float(np.exp(np.log(1.5) * (2 * v[0] - 1)))
        if u[1] < self.p:
            m = float(a.mean()); a = m + (a - m) * float(np.exp(np.log(1.5) * (2 * v[1] - 1)))
        if u[2] < self.p:
            if v[2] < 0.5:
                a = cv2.GaussianBlur(a, (0, 0), 0.5 + float(v[3]))
            else:
                a = a + (0.5 + float(v[3])) * (a - cv2.GaussianBlur(a, (0, 0), 1.0))
        if u[3] < self.p:
            a = a + g.standard_normal(a.shape, dtype=np.float32) * (1. + 5. * float(v[4]))
        a = np.clip(np.rint(a), 0, 255).astype(np.uint8)
        c = cv2.createCLAHE(2.0, (1296 // 32, 972 // 32)).apply(a)
        lo, hi = np.percentile(c, (1, 99))
        x = torch.from_numpy(np.ascontiguousarray(np.clip((c.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)[None]))
        horizontal, vertical = R.flip_decisions(self.seed, self.epoch, idx)
        if horizontal:
            x = torch.flip(x, (-1,))
        if vertical:
            x = torch.flip(x, (-2,))
        return x.contiguous(), torch.from_numpy(self.targets[idx]), idx


def shade_correct(a):
    """Flat-field correction: a * mean(bg) / max(bg, 8), bg = Gaussian blur (sigma 64 px) computed at 1/8 scale and upsampled."""
    import cv2
    h, w = a.shape
    small = cv2.resize(a, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), 8.0), (w, h), interpolation=cv2.INTER_LINEAR)
    return a * (float(bg.mean()) / np.maximum(bg, 8.0))


class ShadeFrames(R.Frames):
    """Recipe A frames with flat-field correction (shade_correct) on the raw uint8 frame before CLAHE + percentile, for every row;
    flips as R.Frames on training rows."""
    def __getitem__(self, row):
        import cv2
        cv2.setNumThreads(1)
        if self.raw is None:
            self.raw = np.memmap(b.RAW, mode='r', dtype=np.uint8, shape=(self.total, 972, 1296))
        idx = int(self.indices[row])
        a = np.clip(np.rint(shade_correct(np.asarray(self.raw[idx]).astype(np.float32))), 0, 255).astype(np.uint8)
        c = cv2.createCLAHE(2.0, (1296 // 32, 972 // 32)).apply(a)
        lo, hi = np.percentile(c, (1, 99))
        x = torch.from_numpy(np.ascontiguousarray(np.clip((c.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)[None]))
        if self.use_augmentation:
            horizontal, vertical = R.flip_decisions(self.seed, self.epoch, idx)
            if horizontal:
                x = torch.flip(x, (-1,))
            if vertical:
                x = torch.flip(x, (-2,))
        return x.contiguous(), torch.from_numpy(self.targets[idx]), idx


class WindowFrames(R.Frames):
    """Full-frame CLAHE + percentile as recipe A, then a crop around the image's window centre; flips as R.Frames (same decisions)."""
    def __init__(self, indices, targets, total, epoch, augment, seed, centres, size, jitter):
        super().__init__(indices, targets, total, epoch, augment, seed)
        self.centres, (self.cw, self.ch), self.jitter = centres, size, jitter

    def __getitem__(self, row):
        x, y, index = b.Frames.__getitem__(self, row)  # (1, 972, 1296), no augmentation
        H, W = x.shape[-2:]
        cx, cy = self.centres[index]
        if not np.isfinite(cx) or not np.isfinite(cy):
            cx, cy = W / 2, H / 2
        if self.use_augmentation and self.jitter:
            j = np.random.default_rng(np.random.SeedSequence([self.seed, 991, self.epoch, index])).uniform(-self.jitter, self.jitter, 2)
            cx, cy = cx + j[0], cy + j[1]
        x0 = int(round(min(max(cx - self.cw / 2, 0), W - self.cw))); y0 = int(round(min(max(cy - self.ch / 2, 0), H - self.ch)))
        x = x[..., y0:y0 + self.ch, x0:x0 + self.cw]
        if self.use_augmentation:
            horizontal, vertical = R.flip_decisions(self.seed, self.epoch, index)
            if horizontal:
                x = torch.flip(x, (-1,))
            if vertical:
                x = torch.flip(x, (-2,))
        return x.contiguous(), y, index


WINDOW = dict(scale=1.0)


def inputs_window(x):
    f = F.FAMILIES[FAMILY]
    x = x.cuda(non_blocking=True)
    if WINDOW['scale'] != 1.0:
        x = torch.nn.functional.interpolate(x, scale_factor=WINDOW['scale'], mode='bilinear', align_corners=False)
    ph, pw = (-x.shape[-2]) % 32, (-x.shape[-1]) % 32
    x = torch.nn.functional.pad(x.expand(-1, 3, -1, -1), (0, pw, 0, ph), mode='replicate')
    if f['norm']:
        x = (x - x.new_tensor(F.MEAN)[None, :, None, None]) / x.new_tensor(F.STD)[None, :, None, None]
    return x.contiguous()


def inputs_geo(x, scale, transpose):
    """Training-only geometric view: resize by scale, optional transpose, replicate-pad right/bottom to multiples of 32 as recipe A."""
    f = F.FAMILIES[FAMILY]
    x = x.cuda(non_blocking=True)
    if scale != 1.0:
        h, w = int(round(x.shape[-2] * scale)), int(round(x.shape[-1] * scale))
        x = torch.nn.functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=False, antialias=scale < 1)
    if transpose:
        x = x.transpose(-1, -2)
    ph, pw = (-x.shape[-2]) % 32, (-x.shape[-1]) % 32  # recipe A: 972 -> 992, 1296 -> 1312
    x = torch.nn.functional.pad(x.expand(-1, 3, -1, -1), (0, pw, 0, ph), mode='replicate')
    if f['norm']:
        x = (x - x.new_tensor(F.MEAN)[None, :, None, None]) / x.new_tensor(F.STD)[None, :, None, None]
    return x.contiguous()


def evaluate(net, dl, mu, sigma, expected, tta=False):
    net.eval(); preds, seen = [], []
    with torch.inference_mode():
        for x, _, ix in dl:
            views = [x] if not tta else [x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2)]  # flip before padding, as in training
            zs = []
            for v in views:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    zs.append(net(inputs_window(v) if WINDOW.get('on') else F.inputs(v, FAMILY)).float()[:, :3])
            z = torch.stack(zs).mean(0)
            preds.append(torch.exp(z * sigma + mu).cpu().numpy()); seen.append(ix.numpy())
    np.testing.assert_array_equal(np.concatenate(seen), expected)
    return np.concatenate(preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=int, required=True); p.add_argument('--seed', type=int, default=46); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--ssl', default='auto'); p.add_argument('--epochs', type=int, default=12); p.add_argument('--schedule-epochs', type=int, default=10)
    p.add_argument('--enc-lr', type=float, default=1e-5); p.add_argument('--head-lr', type=float, default=1e-3); p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--llrd', type=float, default=0.0); p.add_argument('--drop-path', type=float, default=0.0)
    p.add_argument('--ema', default=''); p.add_argument('--ema-start', type=int, default=3); p.add_argument('--swa-start', type=int, default=0)
    p.add_argument('--aux', default=''); p.add_argument('--aux-w', type=float, default=0.2)
    p.add_argument('--row-weight', default='');
    p.add_argument('--teacher', default=''); p.add_argument('--teacher-a', type=float, default=0.5)
    p.add_argument('--window', default=''); p.add_argument('--window-size', default='576,448'); p.add_argument('--window-scale', type=float, default=2.0)
    p.add_argument('--window-jitter', type=float, default=32.0)
    p.add_argument('--transpose-p', type=float, default=0.0); p.add_argument('--zoom', type=float, default=1.0)
    p.add_argument('--appear', type=float, default=0.0); p.add_argument('--shading', action='store_true')
    p.add_argument('--rope-fixed', action='store_true'); p.add_argument('--tta-final', action='store_true')
    p.add_argument("--micro", type=int, default=4); p.add_argument("--workers", type=int, default=4); p.add_argument("--batch", type=int, default=16); p.add_argument("--clip", type=float, default=1.0); p.add_argument("--target-w", default="")
    p.add_argument('--smoke', type=int, default=0, help='stop after this many updates, write nothing')
    p.add_argument('--track-eval-from', type=int, default=4, help='first epoch at which averaged tracks are evaluated')
    p.add_argument('--keep-epochs', default='', help='comma list of epochs whose raw weights are saved')
    p.add_argument('--family', default='cpdino_b8', choices=list(F.FAMILIES))
    a = p.parse_args(); TW = torch.tensor([float(v) for v in a.target_w.split(",")], device="cuda") if a.target_w else None
    global FAMILY
    FAMILY = a.family
    assert torch.cuda.device_count() == 1, 'one visible GPU'
    root = a.root; tracks_dir = {}
    ids, y, tr, va = arrays(a.fold); n = len(ids)
    if a.ssl == 'auto':
        assert FAMILY == 'cpdino_b8' and a.fold >= 0, 'auto SSL5 checkpoints exist only for recipe A folds; pass --ssl PATH or none'
    ssl = SSL5[a.fold] if a.ssl == 'auto' else (None if a.ssl == 'none' else a.ssl)
    fa = argparse.Namespace(family=FAMILY)
    net = F.build_net(fa, a.seed, ssl)
    tok = net[0].tokens; enc = tok.original; L = len(enc_blocks(enc))
    aux = side_table(a.aux, n) if a.aux else None; k_aux = aux.shape[1] if aux is not None else 0
    if k_aux:  # widen the last layer; first three rows keep recipe A's init draw
        old = net[-1]; new = nn.Linear(old.in_features, 3 + k_aux).cuda()
        with torch.no_grad():
            torch.manual_seed(a.seed + 1); nn.init.normal_(new.weight, std=1e-3); nn.init.zeros_(new.bias)
            new.weight[:3].copy_(old.weight); new.bias[:3].copy_(old.bias)
        net[-1] = new
        am = np.nanmean(aux[tr], 0); asd = np.nanstd(aux[tr], 0) + 1e-6
        aux_t = torch.tensor((aux - am) / asd, dtype=torch.float32)
    teacher = side_table(a.teacher, n, 3) if a.teacher else None
    if teacher is not None:
        assert np.isfinite(teacher[tr]).all() and (teacher[tr] > 0).all(), 'teacher must cover every training row'
        ly = np.log(np.where(y > 0, y, 1.0))
        ymix = np.exp((1 - a.teacher_a) * ly + a.teacher_a * np.log(np.where(np.isfinite(teacher), teacher, np.where(y > 0, y, 1.0)))).astype(np.float32)
        ymix[va] = y[va]  # validation targets are never mixed (they are never used for training anyway)
    else:
        ymix = y
    if a.row_weight:
        rw = side_table(a.row_weight, n, 1)[:, 0]
        assert np.isfinite(rw[tr]).all() and (rw[tr] > 0).all(), 'row weights must cover every training row'
        rw_t = torch.tensor(np.nan_to_num(rw, nan=1.0), dtype=torch.float32); rw_mean = float(rw[tr].mean())
    if a.rope_fixed and hasattr(enc, 'rope_embed'):
        enc.rope_embed.rescale_coords = None
    dp_rates = add_drop_path(enc, a.drop_path) if a.drop_path > 0 else []
    groups = param_groups(net, a, L)
    opt = torch.optim.AdamW(groups, weight_decay=a.wd, fused=True)
    params = [q for g in groups for q in g['params']]
    tracks = []
    for d in [float(v) for v in a.ema.split(',') if v]:
        tracks.append(Track(f'ema{str(d)[2:]}', params, decay=d, start_epoch=a.ema_start))
    if a.swa_start:
        tracks.append(Track(f'swa{a.swa_start}', params, decay=None, start_epoch=a.swa_start))
    logy = np.log(y[tr].astype(np.float64)); mu = torch.tensor(logy.mean(0), device='cuda', dtype=torch.float32); sigma = torch.tensor(logy.std(0), device='cuda', dtype=torch.float32)
    R.ARGS = argparse.Namespace(seed=a.seed)
    updates = math.ceil(len(tr) / a.batch); step = 0; wall = time.monotonic()
    config = dict(vars(a), root=str(root), ssl=str(ssl), drop_path_rates=dp_rates, groups=[(g['name'], g['base_lr'], len(g['params'])) for g in groups],
                  tracks=[t.name for t in tracks], k_aux=k_aux, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), pid=os.getpid(), started_utc=time.strftime('%FT%TZ', time.gmtime()))
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in config.items()}
    if not a.smoke:
        for name in ['raw'] + [t.name for t in tracks]:
            d = root / name / 'runs' / f'fold{a.fold}' / FAMILY
            if (d / 'history.json').exists():
                raise FileExistsError(f'no overwrite or implicit resume: {d}')
            d.mkdir(parents=True, exist_ok=True); tracks_dir[name] = d
        pre.save(root / 'launch.json', config)
    log(dict(launch=config))
    if a.window:
        assert not (a.transpose_p or a.zoom > 1.0 or a.appear > 0 or a.shading), 'window view and geometric / appearance / shading arms are separate'
        WINDOW.update(on=True, scale=a.window_scale)
        wsize = tuple(int(v) for v in a.window_size.split(',')); centres = side_table(a.window, n, 2)
    def frames(rows, targets, epoch, augment):
        if a.window:
            return WindowFrames(rows, targets, n, epoch, augment, a.seed, centres, wsize, a.window_jitter)
        if a.shading:
            assert a.appear == 0, 'shading and appearance augmentation are separate arms'
            return ShadeFrames(rows, targets, n, epoch, augment, a.seed)
        if a.appear > 0:
            return AppearFrames(rows, targets, n, epoch, augment, a.seed, a.appear)
        return R.Frames(rows, targets, n, epoch, augment, a.seed)
    def eval_loader(rows, bs=None):
        return DataLoader(frames(rows, y, 0, False), batch_size=bs or a.micro, shuffle=False, num_workers=a.workers, pin_memory=True, prefetch_factor=2, drop_last=False)
    val = eval_loader(va); probe = np.sort(np.random.default_rng(12012).choice(tr, min(256, len(tr)), replace=False)); pl = eval_loader(probe)
    hist = {name: [] for name in ['raw'] + [t.name for t in tracks]}
    keep_epochs = {int(v) for v in a.keep_epochs.split(',') if v}
    for epoch in range(a.epochs):
        order = np.random.default_rng(a.seed + epoch).permutation(tr)
        dl = DataLoader(frames(order, ymix, epoch, True), batch_size=a.batch, shuffle=False, num_workers=a.workers, pin_memory=True, prefetch_factor=2, drop_last=False)
        net.train(); tick = time.monotonic(); torch.cuda.reset_peak_memory_stats(); stats = torch.zeros(5, dtype=torch.float64, device='cuda'); nsum = nmax = clipped = 0.; seen = 0
        geo_rng = np.random.default_rng([a.seed, 777, epoch])
        for x, t, ix in dl:
            geo = (a.transpose_p > 0 or a.zoom > 1.0)
            if geo:
                scale = float(np.exp(geo_rng.uniform(-np.log(a.zoom), np.log(a.zoom)))) if a.zoom > 1.0 else 1.0
                transpose = bool(geo_rng.random() < a.transpose_p)
                if scale != 1.0:
                    t = t.clone(); t[:, 0] = t[:, 0] / scale ** 2
            ratio = R.lr_ratio(step, updates, a.schedule_epochs)
            for g in opt.param_groups:
                g['lr'] = g['base_lr'] * ratio
            opt.zero_grad(set_to_none=True); m = len(t)
            for s in range(0, m, a.micro):
                xx, yy = x[s:s + a.micro], t[s:s + a.micro].cuda(non_blocking=True); c = len(yy)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z = net(inputs_geo(xx, scale, transpose) if geo else inputs_window(xx) if a.window else F.inputs(xx, FAMILY))
                    if a.row_weight:  # same per-row terms as b.loss_fn, weighted
                        d = z[:, :3].float() - (yy.log() - mu) / sigma
                        lv = 0.5 * d.abs().mean(1) + 2.5 * torch.expm1(d * sigma).abs().mean(1)
                        wv = rw_t[ix[s:s + a.micro]].cuda(non_blocking=True)
                        loss = (wv * lv).mean() / rw_mean
                    else:
                        loss = b.loss_fn(z[:, :3], yy, mu, sigma) if TW is None else ((0.5 * (z[:, :3].float() - (yy.log() - mu) / sigma).abs() + 2.5 * torch.expm1((z[:, :3].float() - (yy.log() - mu) / sigma) * sigma).abs()) * TW).sum(1).mean() / TW.sum()
                    if k_aux:
                        zt = aux_t[ix[s:s + a.micro]].cuda(non_blocking=True); ok = torch.isfinite(zt)
                        la = nn.functional.smooth_l1_loss(z[:, 3:].float()[ok], zt[ok]) if ok.any() else z.sum() * 0
                        total = loss + a.aux_w * la
                    else:
                        la = torch.zeros((), device='cuda'); total = loss
                if not torch.isfinite(total):
                    raise ValueError('Nonfinite loss')
                with torch.no_grad():
                    rel = torch.expm1((z[:, :3].float() - (yy.log() - mu) / sigma) * sigma).abs()
                    stats[0] += loss.detach().double() * c; stats[1:4] += rel.double().sum(0) * 100; stats[4] += la.detach().double() * c
                (total * (c / m)).backward()
            norm = nn.utils.clip_grad_norm_(net.parameters(), a.clip if a.clip > 0 else float("inf"), error_if_nonfinite=True).item()
            nsum += norm; nmax = max(nmax, norm); clipped += norm > 1.
            opt.step(); step += 1; seen += m
            for tk in tracks:
                tk.update(params, epoch + 1)
            if step % 25 == 0 or step == 1:
                log(dict(status='training', epoch=epoch + 1, updates=step, seen=seen, gradient_norm=norm, top_enc_lr=next(g['lr'] for g in opt.param_groups if g['name'] == f'enc{L}'),
                         seconds_this_epoch=round(time.monotonic() - tick, 1), peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                         loss=(stats[0] / seen).item(), aux_loss=(stats[4] / seen).item()))
            if a.smoke and step >= a.smoke:
                log(dict(smoke='done', seconds_per_update=(time.monotonic() - tick) / step, peak_gib=torch.cuda.max_memory_allocated() / 2**30))
                pv = evaluate(net, eval_loader(va[:16]), mu, sigma, va[:16])
                for tk in tracks:
                    if tk.active(epoch + 1):
                        backup = swap(params, tk.avg)
                        pe = evaluate(net, eval_loader(va[:16]), mu, sigma, va[:16]); restore(params, backup)
                        log(dict(smoke_track=tk.name, n=tk.n, max_rel_diff=float(np.abs(pe / pv - 1).max())))
                if a.tta_final:
                    pt = evaluate(net, eval_loader(va[:8]), mu, sigma, va[:8], tta=True); log(dict(smoke_tta=float(np.abs(pt / pv[:8] - 1).max())))
                return
        assert seen == len(tr)
        train_seconds = time.monotonic() - tick
        vals = (stats / len(tr)).cpu().tolist()
        base = dict(epoch=epoch + 1, train_seconds=train_seconds, online_train_loss=vals[0], online_train_mape=dict(zip(('CD', 'CV', 'HEX'), vals[1:4])), online_aux_loss=vals[4],
                    gradient_norm_mean=nsum / updates, gradient_norm_max=nmax, gradient_clip_fraction=clipped / updates, updates=step, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        last = epoch + 1 == a.epochs
        for name, tk in [('raw', None)] + [(t.name, t) for t in tracks]:
            if tk is not None and not tk.active(epoch + 1, a.track_eval_from):
                continue
            if tk is not None:
                backup = swap(params, tk.avg)
            t0 = time.monotonic()
            pred = evaluate(net, val, mu, sigma, va); score = mape_score(pred, y[va])
            item = dict(base, score=score, eval_seconds=time.monotonic() - t0, averaged_updates=(tk.n if tk else None))
            if tk is None or last:
                pp = evaluate(net, pl, mu, sigma, probe); item['train_probe_score'] = mape_score(pp, y[probe])
            d = tracks_dir[name]
            np.savez_compressed(d / f'predictions_epoch_{epoch + 1:02d}.npz', indices=va, ids=ids[va], target=y[va], pred=pred)
            if last:
                state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
                torch.save(dict(model=state, epoch=epoch + 1, family=FAMILY, fold=a.fold, seed=a.seed, track=name, config=config,
                                log_mean=logy.mean(0).tolist(), log_std=logy.std(0).tolist()), d / 'final.pt')
                if a.tta_final:
                    pt = evaluate(net, val, mu, sigma, va, tta=True); item['tta_score'] = mape_score(pt, y[va])
                    np.savez_compressed(d / 'predictions_final_tta.npz', indices=va, ids=ids[va], target=y[va], pred=pt)
            elif tk is None and (epoch + 1) in keep_epochs:
                torch.save(dict(model={k: v.detach().cpu() for k, v in net.state_dict().items()}, epoch=epoch + 1, fold=a.fold, seed=a.seed, track=name,
                                log_mean=logy.mean(0).tolist(), log_std=logy.std(0).tolist()), d / f'epoch_{epoch + 1:02d}.pt')
            if tk is not None:
                restore(params, backup); del backup
            hist[name].append(item); pre.save(d / 'history.json', hist[name])
            log(dict(epoch_complete=epoch + 1, track=name, mean=round(score['mean'], 4), CD=round(score['CD'], 3), CV=round(score['CV'], 3), HEX=round(score['HEX'], 3),
                     probe=round(item['train_probe_score']['mean'], 3) if 'train_probe_score' in item else None, train_seconds=round(train_seconds)))
    for name, h in hist.items():
        if not h:
            continue
        w = lambda lo, hi: {k: float(np.mean([e['score'][k] for e in h if lo <= e['epoch'] <= hi])) for k in ('CD', 'CV', 'HEX', 'mean')} if any(lo <= e['epoch'] <= hi for e in h) else None
        pre.save(tracks_dir[name] / 'result.json', dict(track=name, fold=a.fold, seed=a.seed, terminal=h[-1]['score'], last5_mean=w(a.epochs - 4, a.epochs), window_8_12=w(8, 12),
                                                         seconds=time.monotonic() - wall, config=config))
        pre.save(tracks_dir[name] / 'complete.json', dict(status='completed'))
    pre.save(root / 'exit.json', dict(status='completed', seconds=time.monotonic() - wall, finished_utc=time.strftime('%FT%TZ', time.gmtime())))


if __name__ == '__main__':
    main()
