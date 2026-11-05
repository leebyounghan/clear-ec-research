"""Train-fit and weight-averaging inference for completed recipe A fold roots (ceiling meta-analysis, 2026-11-03).

For one completed recipe A regression root (one GPU, no weights written back):
  e20  : terminal checkpoint, one whole-frame view, on every train row and every validation row. The validation
         part must reproduce the stored predictions_epoch_20.npz within 1 % (proves the train-row pass uses the
         same model and preprocessing).
  swa  : uniform average of the epoch 11-20 weights (the constant-low-LR tail of the 10-epoch schedule), one view
         on train rows, four flips on validation rows.
  eXX  : optional extra terminal-style checkpoints (--epochs 5 10 15), one view on train and validation rows.
Output: <out>/fold<F>_<tag>.npz with indices/ids/target for train and validation and decoded predictions.
Why: train-row error versus validation error separates "cannot fit" (bias, information or label noise) from
"fits but does not generalise" (variance), and SWA is the zero-inference-cost version of the snapshot ensemble that
gave -0.036 at member level.
"""
import argparse, datetime, json
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import phase2_foundation_train_20261002 as b
import phase2_foundation_preflight_20261002 as preflight
import phase2_pooling_20261019 as pooling

FLIPS = ((False, False), (True, False), (False, True), (True, True))
BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def flip(x, h, v):
    if h:
        x = torch.flip(x, (-1,))
    if v:
        x = torch.flip(x, (-2,))
    return x


def official(pred, y):
    s = y > 0
    v = np.nanmean(np.where(s, np.abs(pred - y) / np.where(s, y, 1.), np.nan), 0) * 100
    return dict(CD=float(v[0]), CV=float(v[1]), HEX=float(v[2]), mean=float(v.mean()), n=int(len(y)))


def build(m, state):
    net, _, _ = preflight.build('cpdino_vitb', False, native_convnext_norm=True)
    net[0] = pooling.SpatialPoolEncoder(net[0].original, m['recipe']['pooling'])
    net.load_state_dict(state, strict=True)
    return net.cuda().eval().requires_grad_(False)


def predict(net, idx, y, n, flips):
    dl = torch.utils.data.DataLoader(b.Frames(idx, y, n), batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    zs, seen = [], []
    for x, _, ix in dl:
        views = []
        for h, v in flips:
            with torch.no_grad(), sdpa_kernel(BACKENDS), torch.autocast('cuda', dtype=torch.bfloat16):
                views.append(net(b.inputs(flip(x, h, v), 'cpdino_vitb')).float())
        zs.append(torch.stack(views, 1).cpu().numpy()); seen.append(ix.numpy())
    seen = np.concatenate(seen)
    np.testing.assert_array_equal(seen, idx)
    return np.concatenate(zs).astype(np.float64)


def main(a):
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    root = a.root
    m = json.loads((root / 'manifest.json').read_text())
    f = m['fold']
    run = root / 'runs' / f'fold{f}' / 'cpdino_vitb'
    E = m['recipe']['epochs']
    with np.load(b.META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    tr = np.asarray(m['train_indices'], dtype=np.int64)
    with np.load(run / f'predictions_epoch_{E:02d}.npz') as z:
        va, stored = z['indices'].astype(np.int64), z['pred'].astype(np.float64)
    assert len(tr) == m['train_n'] and not np.intersect1d(tr, va).size
    a.out.mkdir(parents=True, exist_ok=True)
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), root=str(root), fold=f, seed=m['seed'])

    def load(ep):
        ck = torch.load(run / f'epoch_{ep:02d}.pt', map_location='cpu', weights_only=True)
        assert ck['manifest_sha'] == preflight.digest(root / 'manifest.json')
        return ck

    def decode(z, ck):
        return np.exp(z * np.array(ck['log_std']) + np.array(ck['log_mean']))

    for ep in sorted(set(a.epochs) | {E}):
        out = a.out / f'fold{f}_e{ep:02d}.npz'
        if out.exists():
            continue
        ck = load(ep)
        net = build(m, ck['model'])
        ztr, zva = predict(net, tr, y, len(ids), FLIPS[:1])[:, 0], predict(net, va, y, len(ids), FLIPS[:1])[:, 0]
        ptr, pva = decode(ztr, ck), decode(zva, ck)
        if ep == E:
            rel = np.abs(pva - stored) / stored
            assert rel.max() < 0.01, f'validation recompute deviates from stored predictions: {rel.max()}'
            report['e20_val_recompute_max_rel_dev'] = float(rel.max())
        np.savez_compressed(out, train_indices=tr, train_pred=ptr, val_indices=va, val_pred=pva, epoch=ep)
        report[f'e{ep:02d}'] = dict(train=official(ptr, y[tr].astype(np.float64)), val=official(pva, y[va].astype(np.float64)))
        print(json.dumps({f'fold{f}_e{ep:02d}': report[f'e{ep:02d}']}), flush=True)
        del net

    out = a.out / f'fold{f}_swa{a.swa_from:02d}_{E:02d}.npz'
    if not out.exists():
        eps = list(range(a.swa_from, E + 1))
        acc, ck = None, None
        for ep in eps:
            ck = load(ep)
            sd = {k: v.double() for k, v in ck['model'].items()}
            acc = sd if acc is None else {k: acc[k] + sd[k] for k in acc}
        state = {k: (acc[k] / len(eps)).to(ck['model'][k].dtype) for k in acc}
        net = build(m, state)
        ztr = predict(net, tr, y, len(ids), FLIPS[:1])[:, 0]
        zva = predict(net, va, y, len(ids), FLIPS)
        ptr, pva1, pva4 = decode(ztr, ck), decode(zva[:, 0], ck), decode(zva.mean(1), ck)
        np.savez_compressed(out, train_indices=tr, train_pred=ptr, val_indices=va, val_pred_single=pva1, val_pred_tta4=pva4,
                            val_z_views=zva.astype(np.float32), epochs=np.array(eps))
        report['swa'] = dict(epochs=eps, train=official(ptr, y[tr].astype(np.float64)),
                             val_single=official(pva1, y[va].astype(np.float64)), val_tta4=official(pva4, y[va].astype(np.float64)))
        print(json.dumps({f'fold{f}_swa': report['swa']}), flush=True)
    (a.out / f'fold{f}_report.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--epochs', type=int, nargs='*', default=[])
    ap.add_argument('--swa-from', type=int, default=11)
    main(ap.parse_args())
