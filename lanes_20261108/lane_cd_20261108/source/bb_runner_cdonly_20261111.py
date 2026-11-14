#!/usr/bin/env python3
"""CD lane copy for D-CDTRAIN-001 (2026-11-11): bb_runner_cd_20261108.py plus TARGET_W (per-target loss weights; CD only = 1,0,0). Otherwise unchanged.
Backbone lane runner (2026-11-08). Recipe A regression (CPDINO ViT-B kernel 16 stride 8, std pool, LN-128-GELU-3 head,
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
  --transpose-p P   per batch, transpose the frames (portrait input) with probability P; labels are rotation invariant
  --zoom Z          per batch, resize by s, log-uniform in [1/Z, Z]; CD label x 1/s^2 (CD = 1e6 / mean cell area), CV and HEX kept

Every track (raw, ema<D>, swa) is written like a separate run, so read tools treat each as one model:
  <root>/<track>/runs/fold<f>/cpdino_b8/{history.json, predictions_epoch_XX.npz, final.pt, result.json}
Run from /workspace/clear-ec-research with one visible GPU."""
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, 'experiments/lane_cd_20261108/source')  # CD lane copy (2026-11-08) of lane_backbone bb_runner_20261108.py; only zero-label handling changed
import family_bb_cd_20261108 as F  # noqa: E402  lane copy of lane GPU 3's family runner (+ timm DINOv3 ViT-L); puts the frozen recipe modules on sys.path

b, R, pre = F.b, F.R, F.pre
SSL5 = {0: 'experiments/phase2_crossfold_20261015/fold0/ssl5/run/last.pt', 4: 'experiments/phase2_crossfold_20261015/fold4/ssl5/run/last.pt',
        **{f: f'experiments/oof_recipeA_20261027/fold{f}/ssl5/run/last.pt' for f in (1, 2, 3)}}
SPLITS = {0: 'experiments/ssl_screen_20260911/preflight/fold0.npz', 1: 'experiments/ssl_screen_20260911/preflight/fold1.npz',
          2: 'experiments/overnight_target_20260912/preflight/fold2.npz', 3: 'experiments/overnight_target_20260912/preflight/fold3.npz',
          4: 'experiments/target_fold4_20260912/preflight/fold4.npz',
          99: 'experiments/fulldata_recipeA_20261028/fulldata_regression_split.npz'}  # CD lane 11-09: full data (train 8952, the two HEX-zero rows held)  # the recipe A OOF split files
FAMILY = 'cpdino_b8'  # set from --family in main
_TW = os.environ.get('TARGET_W')   # D-CDTRAIN-001 copy (2026-11-11): e.g. TARGET_W=1,0,0 trains on the CD loss only; unset = recipe loss
TARGET_W = torch.tensor([float(v) for v in _TW.split(',')], device='cuda') if _TW else None


def log(o):
    print(json.dumps(o), flush=True)


def arrays(fold):
    with np.load(b.META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    with np.load(SPLITS[fold]) as z:
        tr, va = z['train'].astype(np.int64), z['validation'].astype(np.int64)
    assert not np.intersect1d(tr, va).size and (y[tr] > 0).all() and (y[va, 0] > 0).all() and (y[va, 1] > 0).all()
    return ids, y, tr, va


def side_table(path, n, k_expected=None):
    """npz {indices, <key>} -> dense (n, k) float32 table, NaN where missing."""
    with np.load(path, allow_pickle=True) as z:
        key = 'Z' if 'Z' in z.files else 'pred'
        ix, v = z['indices'].astype(np.int64), z[key].astype(np.float32)
    v = v.reshape(len(v), -1); out = np.full((n, v.shape[1]), np.nan, np.float32); out[ix] = v
    if k_expected:
        assert v.shape[1] == k_expected
    return out


def add_drop_path(enc, pmax):
    L = len(enc.blocks); rates = []
    for i, blk in enumerate(enc.blocks):
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
            elif any(s in name for s in ('patch_embed', 'cls_token', 'storage_tokens', 'reg_token', 'mask_token', 'rope', 'pos_embed')):
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
                    zs.append(net(F.inputs(v, FAMILY)).float()[:, :3])
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
    p.add_argument('--teacher', default=''); p.add_argument('--teacher-a', type=float, default=0.5)
    p.add_argument('--transpose-p', type=float, default=0.0); p.add_argument('--zoom', type=float, default=1.0)
    p.add_argument('--rope-fixed', action='store_true'); p.add_argument('--tta-final', action='store_true')
    p.add_argument('--micro', type=int, default=4); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--smoke', type=int, default=0, help='stop after this many updates, write nothing')
    p.add_argument('--track-eval-from', type=int, default=4, help='first epoch at which averaged tracks are evaluated')
    p.add_argument('--keep-epochs', default='', help='comma list of epochs whose raw weights are saved')
    p.add_argument('--family', default='cpdino_b8', choices=list(F.FAMILIES))
    a = p.parse_args()
    global FAMILY
    FAMILY = a.family
    assert torch.cuda.device_count() == 1, 'one visible GPU'
    root = a.root; tracks_dir = {}
    ids, y, tr, va = arrays(a.fold); n = len(ids)
    if a.ssl == 'auto':
        assert FAMILY == 'cpdino_b8', 'auto SSL5 checkpoints exist only for recipe A; pass --ssl PATH or none'
    ssl = SSL5[a.fold] if a.ssl == 'auto' else (None if a.ssl == 'none' else a.ssl)
    fa = argparse.Namespace(family=FAMILY)
    net = F.build_net(fa, a.seed, ssl)
    tok = net[0].tokens; enc = tok.original; L = len(enc.blocks)
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
        ymix = np.exp((1 - a.teacher_a) * np.log(y) + a.teacher_a * np.log(np.where(np.isfinite(teacher), teacher, y))).astype(np.float32)
        ymix[va] = y[va]  # validation targets are never mixed (they are never used for training anyway)
    else:
        ymix = y
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
    updates = math.ceil(len(tr) / 16); step = 0; wall = time.monotonic()
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
    def eval_loader(rows, bs=None):
        return DataLoader(R.Frames(rows, y, n, 0, False, a.seed), batch_size=bs or a.micro, shuffle=False, num_workers=a.workers, pin_memory=True, prefetch_factor=2, drop_last=False)
    val = eval_loader(va); probe = np.sort(np.random.default_rng(12012).choice(tr, min(256, len(tr)), replace=False)); pl = eval_loader(probe)
    hist = {name: [] for name in ['raw'] + [t.name for t in tracks]}
    keep_epochs = {int(v) for v in a.keep_epochs.split(',') if v}
    for epoch in range(a.epochs):
        order = np.random.default_rng(a.seed + epoch).permutation(tr)
        dl = DataLoader(R.Frames(order, ymix, n, epoch, True, a.seed), batch_size=16, shuffle=False, num_workers=a.workers, pin_memory=True, prefetch_factor=2, drop_last=False)
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
                    z = net(inputs_geo(xx, scale, transpose) if geo else F.inputs(xx, FAMILY))
                    if TARGET_W is None:
                        loss = b.loss_fn(z[:, :3], yy, mu, sigma)
                    else:   # D-CDTRAIN-001: recipe loss per target column, weighted (CD only = 1,0,0)
                        zt = (yy.log() - mu) / sigma; dd = z[:, :3].float() - zt
                        per = 0.5 * dd.abs() + 2.5 * torch.expm1(dd * sigma).abs()
                        loss = (per * TARGET_W).sum(1).mean() / TARGET_W.sum()
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
            norm = nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True).item()
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
            pred = evaluate(net, val, mu, sigma, va); score = F.safe_metrics(pred, y[va])
            item = dict(base, score=score, eval_seconds=time.monotonic() - t0, averaged_updates=(tk.n if tk else None))
            if tk is None or last:
                pp = evaluate(net, pl, mu, sigma, probe); item['train_probe_score'] = F.safe_metrics(pp, y[probe])
            d = tracks_dir[name]
            np.savez_compressed(d / f'predictions_epoch_{epoch + 1:02d}.npz', indices=va, ids=ids[va], target=y[va], pred=pred)
            if last:
                state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
                torch.save(dict(model=state, epoch=epoch + 1, family=FAMILY, fold=a.fold, seed=a.seed, track=name, config=config,
                                log_mean=logy.mean(0).tolist(), log_std=logy.std(0).tolist()), d / 'final.pt')
                if a.tta_final:
                    pt = evaluate(net, val, mu, sigma, va, tta=True); item['tta_score'] = F.safe_metrics(pt, y[va])
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
