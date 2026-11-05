"""One backbone on one fold, one GPU, for the combination sweep (2026-11-05). The foundation recipe of 2026-10-02 (20 epochs, encoder LR 1e-5, head LR 1e-3, flips, terminal epoch), so the existing fold-3 runs of dinov3_vitb and convnext_base can join: their add-at-20 % deltas on B were -0.023 and -0.030, CI crossing 0, at about 1 h per fold.
Rows with a zero label leave training; validation scores every row with the official zero rule. Writes history.json, predictions.npz, predictions_tta4.npz (four flips, log mean) and last.pt under ROOT/fold{F}/{arch}_s{seed}. usage: sweep_backbone_20261105.py --arch A --fold F --root R [--seed 42]"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, 'experiments/meta_ceiling_20261103/source')
import phase2_foundation_train_20261002 as b
FLIPS = ((False, False), (True, False), (False, True), (True, True))
RECIPE = dict(epochs=20, encoder_lr=1e-5, head_lr=1e-3, weight_decay=1e-4, gradient_clip=1.0)


def split_path(f):
    return Path('experiments/target_fold4_20260912/preflight/fold4.npz') if f == 4 else Path(f'experiments/ssl_screen_20260911/preflight/fold{f}.npz')


def official(p, y):
    s = y > 0
    v = np.nanmean(np.where(s, np.abs(p - y) / np.where(s, y, 1.0), np.nan), 0) * 100
    return dict(CD=float(v[0]), CV=float(v[1]), HEX=float(v[2]), mean=float(v.mean()), n=int(len(y)))


def predict(net, arch, dl, mu, sigma):
    net.eval(); Z = {i: [] for i in range(4)}; seen = []
    with torch.inference_mode():
        for x, _, ix in dl:
            seen.append(ix.numpy())
            for i, (h, v) in enumerate(FLIPS):
                q = x
                if h: q = torch.flip(q, (-1,))
                if v: q = torch.flip(q, (-2,))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    Z[i].append(net(b.inputs(q, arch)).float().cpu().numpy())
    Z = np.stack([np.concatenate(Z[i]) for i in range(4)], 1) * sigma.cpu().numpy() + mu.cpu().numpy()
    return np.concatenate(seen), np.exp(Z[:, 0]), np.exp(Z.mean(1))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--arch', required=True); p.add_argument('--fold', type=int, required=True); p.add_argument('--root', required=True); p.add_argument('--seed', type=int, default=42); a = p.parse_args()
    assert torch.cuda.device_count() == 1
    out = Path(a.root) / f'fold{a.fold}' / f'{a.arch}_s{a.seed}'
    if (out / 'complete.json').exists():
        print('already complete', out); return
    out.mkdir(parents=True, exist_ok=True)
    with np.load(b.META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    with np.load(split_path(a.fold)) as z:
        tr, va = z['train'].copy(), z['validation'].copy()
    tr = tr[(y[tr] > 0).all(1)]; log_y = np.log(y[tr].astype(np.float64))
    m = dict(recipe=RECIPE, log_mean=log_y.mean(0).tolist(), log_std=log_y.std(0).tolist())
    net, opt, mu, sigma, details = b.setup(a.arch, m); torch.manual_seed(a.seed); r = RECIPE
    (out / 'manifest.json').write_text(json.dumps(dict(arch=a.arch, fold=a.fold, seed=a.seed, split=str(split_path(a.fold)), train_n=len(tr), validation_n=len(va), recipe=r, log_mean=m['log_mean'], log_std=m['log_std'], details=details, created_utc=time.strftime('%FT%TZ', time.gmtime())), indent=1, default=str))
    updates = math.ceil(len(tr) / 16); step = 0; history = []; t0 = time.monotonic(); val = b.loader(va, y, len(ids))
    for epoch in range(r['epochs']):
        order = np.random.default_rng(a.seed + epoch).permutation(tr); dl = b.loader(order, y, len(ids), epoch, True); net.train(); opt.zero_grad(set_to_none=True); total = 0.0; seen = 0; t1 = time.monotonic()
        for micro, (x, t, ix) in enumerate(dl):
            group_n = min(16, len(tr) - (micro // 4) * 16)
            if micro % 4 == 0:
                ratio = (step + 1) / updates if step < updates else .1 + .9 * .5 * (1 + math.cos(math.pi * (step - updates) / max(1, updates * (r['epochs'] - 1) - 1)))
                opt.param_groups[0]['lr'] = r['encoder_lr'] * ratio; opt.param_groups[1]['lr'] = r['head_lr'] * ratio
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = b.loss_fn(net(b.inputs(x, a.arch)), t.cuda(non_blocking=True), mu, sigma)
            if not torch.isfinite(loss).item():
                raise ValueError('Nonfinite objective')
            (loss * (len(t) / group_n)).backward(); total += loss.item() * len(t); seen += len(t)
            if (micro + 1) % 4 == 0 or micro + 1 == len(dl):
                torch.nn.utils.clip_grad_norm_(net.parameters(), r['gradient_clip'], error_if_nonfinite=True); opt.step(); opt.zero_grad(set_to_none=True); step += 1
        assert seen == len(tr)
        ix, single, _ = predict_one(net, a.arch, val, mu, sigma); assert np.array_equal(ix, va); s = official(single, y[va])
        history.append(dict(epoch=epoch + 1, score=s, train_loss=total / seen, seconds=time.monotonic() - t1)); (out / 'history.json').write_text(json.dumps(history, indent=1)); print(json.dumps(dict(arch=a.arch, fold=a.fold, epoch=epoch + 1, mean=round(s['mean'], 3), CD=round(s['CD'], 3), CV=round(s['CV'], 3), HEX=round(s['HEX'], 3), seconds=round(history[-1]['seconds']))), flush=True)
    torch.save(dict(model=net.state_dict(), arch=a.arch, fold=a.fold, seed=a.seed, log_mean=m['log_mean'], log_std=m['log_std'], recipe=r), out / 'last.pt')
    ix, single, tta = predict(net, a.arch, val, mu, sigma); assert np.array_equal(ix, va)
    np.savez_compressed(out / 'predictions.npz', indices=va, ids=ids[va], target=y[va], pred=single); np.savez_compressed(out / 'predictions_tta4.npz', indices=va, ids=ids[va], target=y[va], pred_single=single, pred_tta4=tta)
    (out / 'complete.json').write_text(json.dumps(dict(single=official(single, y[va]), tta4=official(tta, y[va]), hours=(time.monotonic() - t0) / 3600), indent=1)); print('complete', (out / 'complete.json').read_text().replace('\n', ' '), flush=True)


def predict_one(net, arch, dl, mu, sigma):
    net.eval(); Z = []; seen = []
    with torch.inference_mode():
        for x, _, ix in dl:
            seen.append(ix.numpy())
            with torch.autocast('cuda', dtype=torch.bfloat16):
                Z.append(net(b.inputs(x, arch)).float())
    z = torch.cat(Z); return np.concatenate(seen), torch.exp(z * sigma + mu).cpu().numpy(), None


if __name__ == '__main__':
    main()
