"""Green day 2 / G3 (pre-registered 2026-11-01 21:10 UTC): hex6-auxiliary multi-task fine-tune of the recipe A regressor.
Per fold f: start from the recipe A seed-46 fold-f terminal checkpoint (encoder + std pooling + head), fine-tune for
--epochs epochs on the training-fold rows with  loss = recipe loss (standardised log L1 + relative error, as
phase2_foundation_train.loss_fn) + lambda * L1(aux(h), standardised log hex6_target)  where h is the 128-d hidden
layer and hex6_target is the fold-clean ACTIVE ten-family mean-log hex6 (experiments/green_day2_20261101/d2b_active_gate/
active_family_features.npz).  Control arm: lambda = 0, identical schedule.  Writes TTA4 predictions of CD/CV/HEX and of
the distilled hex6 for the held fold.
  train --fold F --arm {aux,control} [--epochs 2] [--smoke]
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, 'experiments/phase2_pooling_replication_20261026/f0_s46_std/source')
sys.path.insert(0, 'experiments/green_track_20261028/source')
import phase2_foundation_train_20261002 as b
import phase2_foundation_preflight_20261002 as preflight
import phase2_pooling_20261019 as pooling
import green_selftrain_20261028 as ST
import green_point_features_20261027 as G

ROOT = Path('experiments/green_day2_20261101'); OUT = ROOT/'g3_hex6aux'
ARCH = 'cpdino_vitb'
RECIPE46 = {0: 'experiments/phase2_pooling_replication_20261026/f0_s46_std', 1: 'experiments/oof_recipeA_20261027/fold1/regression_std_s46',
            2: 'experiments/oof_recipeA_20261027/fold2/regression_std_s46_v2', 3: 'experiments/oof_recipeA_20261027/fold3/regression_std_s46',
            4: 'experiments/phase2_pooling_20261020/arms/std'}
HEX6 = ROOT/'d2b_active_gate'/'active_family_features.npz'
ACTIVE = ['s8_s46', 's8_s47', 's8_s48', 'sg2_s46', 'sg2_s47', 'sg2_s48', 'sg2_s49', 'sg2_s50', 's8multi_s47', 'sg2pre_s46']
LAMBDA = {'aux': 1.0, 'control': 0.0, 'aux3': 3.0}
FLIPS = ((False, False), (True, False), (False, True), (True, True))


def log(msg, **kw):
    print(json.dumps(dict(utc=G.now(), msg=msg, **kw)), flush=True)


def build(fold):
    net, multiple, details = preflight.build(ARCH, False, native_convnext_norm=True)
    net[0] = pooling.SpatialPoolEncoder(net[0].original, 'std')
    dim = 768*len(pooling.MODES['std'])
    if dim != 768:
        net[1] = torch.nn.LayerNorm(dim); net[2] = torch.nn.Linear(dim, 128)
    ck = torch.load(Path(RECIPE46[fold])/'runs'/f'fold{fold}'/ARCH/'epoch_20.pt', map_location='cpu', weights_only=False)
    net.load_state_dict(ck['model'], strict=True)
    return net, np.array(ck['log_mean'], dtype=np.float64), np.array(ck['log_std'], dtype=np.float64)


def hex6_target(n):
    with np.load(HEX6) as z:
        A = np.log(np.stack([z[f'hex6_{k}'] for k in ACTIVE])); seen = z['seen']
    with np.errstate(all='ignore'):
        t = np.nanmean(A, 0)
    t[seen] = np.nan                                   # the 20 train-green rows: detector-leaked, never an aux target
    assert len(t) == n
    return t


def flip(x, h, v):
    if h: x = torch.flip(x, (-1,))
    if v: x = torch.flip(x, (-2,))
    return x


def forward(net, aux, x):
    h = net[3](net[2](net[1](net[0](b.inputs(x, ARCH)))))
    return net[4](h), aux(h).squeeze(-1)


def train(a):
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    out = OUT/a.arm/f'fold{a.fold}'; out.mkdir(parents=True, exist_ok=True)
    ann, binding, ids, y, tr, va = ST.green(a.fold)
    n = len(ids); y = np.asarray(y, dtype=np.float32)
    net, mu_np, sigma_np = build(a.fold)
    net = net.cuda().train()
    torch.manual_seed(46+a.fold)
    aux = torch.nn.Linear(128, 1).cuda(); torch.nn.init.normal_(aux.weight, std=1e-3); torch.nn.init.zeros_(aux.bias)
    mu = torch.tensor(mu_np, device='cuda', dtype=torch.float32); sigma = torch.tensor(sigma_np, device='cuda', dtype=torch.float32)
    ht = hex6_target(n); m_h, s_h = float(np.nanmean(ht[tr])), float(np.nanstd(ht[tr]))
    ht_std = (ht-m_h)/s_h
    lam = LAMBDA[a.arm]
    if a.smoke:
        tr = tr[:32]; va = va[:8]
    opt = torch.optim.AdamW([{'params': net[0].parameters(), 'lr': a.encoder_lr}, {'params': list(net[1:].parameters()), 'lr': a.head_lr},
                             {'params': aux.parameters(), 'lr': 1e-3}], weight_decay=0.05, fused=True)
    rng = np.random.default_rng(46+a.fold)
    micro, accum = 4, 4                               # 4 x 4 = batch 16 as in the recipe
    updates = math.ceil(len(tr)/(micro*accum))*a.epochs
    warm = min(50, max(1, updates//10)); step = 0; t0 = time.time(); hist = []
    for epoch in range(a.epochs):
        order = rng.permutation(tr)
        dl = DataLoader(b.Frames(order, y, n), batch_size=micro, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2, drop_last=False)
        acc = np.zeros(4); k = 0
        for i, (x, t, ix) in enumerate(dl):
            hflip, vflip = rng.random() < .5, rng.random() < .5
            x = flip(x, hflip, vflip)
            if i % accum == 0:
                ratio = min(1.0, (step+1)/warm)
                opt.param_groups[0]['lr'] = a.encoder_lr*ratio; opt.param_groups[1]['lr'] = a.head_lr*ratio; opt.param_groups[2]['lr'] = 1e-3*ratio
                opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z, za = forward(net, aux, x)
            tt = t.cuda(non_blocking=True)
            loss_main = b.loss_fn(z, tt, mu, sigma)
            tgt = torch.tensor(ht_std[ix.numpy()], device='cuda', dtype=torch.float32); ok_t = torch.isfinite(tgt)
            loss_aux = (za.float()[ok_t]-tgt[ok_t]).abs().mean() if bool(ok_t.any()) else za.float().sum()*0
            loss = loss_main+lam*loss_aux
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            (loss/accum).backward()
            acc += [float(loss_main), float(loss_aux), float(len(t)), 1]; k += 1
            if (i+1) % accum == 0 or (i+1) == len(dl):
                torch.nn.utils.clip_grad_norm_(list(net.parameters())+list(aux.parameters()), 1.0, error_if_nonfinite=True)
                opt.step(); step += 1
                if step % 25 == 0 or step == 1:
                    log('train', arm=a.arm, fold=a.fold, epoch=epoch+1, step=step, of=updates, loss_main=acc[0]/acc[3], loss_aux=acc[1]/acc[3], seconds=round(time.time()-t0, 1), peak_gib=round(torch.cuda.max_memory_allocated()/2**30, 1))
                    hist.append(dict(step=step, loss_main=acc[0]/acc[3], loss_aux=acc[1]/acc[3])); acc[:] = 0
    # held-fold TTA4 predictions
    net.eval(); aux.eval()
    dl = DataLoader(b.Frames(va, y, n), batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    zs, zas, seen = [], [], []
    with torch.no_grad():
        for x, t, ix in dl:
            zv, zav = [], []
            for h, v in FLIPS:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z, za = forward(net, aux, flip(x, h, v))
                zv.append(z.float().cpu().numpy()); zav.append(za.float().cpu().numpy())
            zs.append(np.stack(zv, 1)); zas.append(np.stack(zav, 1)); seen.append(ix.numpy())
    zs = np.concatenate(zs); zas = np.concatenate(zas); seen = np.concatenate(seen)
    np.testing.assert_array_equal(seen, va)
    pred_tta4 = np.exp(zs.mean(1)*sigma_np+mu_np); pred_single = np.exp(zs[:, 0]*sigma_np+mu_np)
    hex6_tta4 = np.exp(zas.mean(1)*s_h+m_h)
    np.savez_compressed(out/'predictions_tta4.npz', indices=va, ids=ids[va], target=y[va], z_views=zs.astype(np.float32), pred_single=pred_single, pred_tta4=pred_tta4,
                        hex6_z_views=zas.astype(np.float32), hex6_pred_tta4=hex6_tta4, flips=np.array(FLIPS), log_mean=mu_np, log_std=sigma_np, hex6_log_mean=m_h, hex6_log_std=s_h)
    torch.save(dict(model=net.state_dict(), aux=aux.state_dict(), fold=a.fold, arm=a.arm, lam=lam, epochs=a.epochs, log_mean=mu_np.tolist(), log_std=sigma_np.tolist(), hex6_log_mean=m_h, hex6_log_std=s_h,
                    init=str(Path(RECIPE46[a.fold])/'runs'/f'fold{a.fold}'/ARCH/'epoch_20.pt'), note='green day 2 G3 hex6-auxiliary fine-tune'), out/'model.pt')
    mape = lambda p, t: float((np.abs(p-t)/t*100)[t > 0].mean())
    res = dict(created_utc=G.now(), arm=a.arm, fold=a.fold, lam=lam, epochs=a.epochs, updates=step, n_train=int(len(tr)), n_val=int(len(va)), seconds=round(time.time()-t0, 1), smoke=bool(a.smoke),
               raw_tta4_mape={k: mape(pred_tta4[:, j], y[va, j]) for j, k in enumerate(('CD', 'CV', 'HEX'))},
               hex6_distilled_r=float(np.corrcoef(np.log(hex6_tta4[np.isfinite(ht[va])]), ht[va][np.isfinite(ht[va])])[0, 1]) if np.isfinite(ht[va]).sum() > 10 else None, history=hist[-5:])
    (out/'result.json').write_text(json.dumps(res, indent=1)); log('done', **{k: v for k, v in res.items() if k != 'history'})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['train']); ap.add_argument('--fold', type=int, required=True); ap.add_argument('--arm', choices=['aux', 'control', 'aux3'], required=True)
    ap.add_argument('--epochs', type=int, default=2); ap.add_argument('--encoder-lr', type=float, default=3e-6); ap.add_argument('--head-lr', type=float, default=3e-4); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); train(a)
