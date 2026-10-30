"""Four-flip TTA inference for one completed recipe A regression root; one GPU, no weights changed.

Reads the frozen manifest and terminal checkpoint, recomputes the single-view
prediction (must match the stored predictions.npz within tolerance), then writes
predictions_tta4.npz: per-view standardized-log outputs, their mean decoded with
the training log mean/std, validation indices/ids/targets, hashes. bf16 autocast,
memory-efficient or math SDPA only. Scoring uses the official zero rule.
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import phase2_foundation_train_20261002 as b
import phase2_foundation_preflight_20261002 as preflight
import phase2_pooling_20261019 as pooling

FLIPS = ((False, False), (True, False), (False, True), (True, True))
BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def flip(x, h, v):
    if h:
        x = torch.flip(x, (-1,))
    if v:
        x = torch.flip(x, (-2,))
    return x


def official(pred, y):
    scored = y > 0
    v = np.nanmean(np.where(scored, np.abs(pred-y)/np.where(scored, y, 1.), np.nan), 0)*100
    return dict(CD=float(v[0]), CV=float(v[1]), HEX=float(v[2]), mean=float(v.mean()), n=int(len(y)))


def main(root):
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    m = json.loads((root/'manifest.json').read_text())
    run = root/'runs'/f'fold{m["fold"]}'/'cpdino_vitb'
    out = run/'predictions_tta4.npz'
    if out.exists():
        raise FileExistsError(out)
    ck = torch.load(run/f'epoch_{m["recipe"]["epochs"]:02d}.pt', map_location='cpu', weights_only=True)
    assert ck['manifest_sha'] == preflight.digest(root/'manifest.json')
    net, _, _ = preflight.build('cpdino_vitb', False, native_convnext_norm=True)
    net[0] = pooling.SpatialPoolEncoder(net[0].original, m['recipe']['pooling'])
    dim = 768*len(pooling.MODES[m['recipe']['pooling']])
    if dim != 768:
        net[1], net[2] = torch.nn.LayerNorm(dim), torch.nn.Linear(dim, 128)
    net.load_state_dict(ck['model'], strict=True)
    net = net.cuda().eval().requires_grad_(False)
    mu = torch.tensor(ck['log_mean'], device='cuda', dtype=torch.float32)
    sigma = torch.tensor(ck['log_std'], device='cuda', dtype=torch.float32)
    with np.load(b.META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    with np.load(run/'predictions.npz') as z:
        va, stored = z['indices'].copy(), z['pred'].astype(np.float64)
    dl = torch.utils.data.DataLoader(b.Frames(va, y, len(ids)), batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    zs, seen = [], []
    for x, _, ix in dl:
        views = []
        for h, v in FLIPS:
            with torch.no_grad(), sdpa_kernel(BACKENDS), torch.autocast('cuda', dtype=torch.bfloat16):
                views.append(net(b.inputs(flip(x, h, v), 'cpdino_vitb')).float())
        zs.append(torch.stack(views, 1).cpu().numpy())
        seen.append(ix.numpy())
    seen, zs = np.concatenate(seen), np.concatenate(zs).astype(np.float64)
    np.testing.assert_array_equal(seen, va)
    lm, ls = np.array(ck['log_mean']), np.array(ck['log_std'])
    single = np.exp(zs[:, 0]*ls+lm)
    tta = np.exp(zs.mean(1)*ls+lm)
    rel = np.abs(single-stored)/stored
    assert rel.max() < 0.01, f'Single-view recompute deviates from stored predictions: {rel.max()}'
    np.savez_compressed(out, indices=va, ids=ids[va], target=y[va], z_views=zs.astype(np.float32), pred_single=single, pred_tta4=tta,
                        flips=np.array(FLIPS), log_mean=lm, log_std=ls)
    report = dict(created_utc=now(), root=str(root), fold=m['fold'], seed=m['seed'], pooling=m['recipe']['pooling'],
                  checkpoint_sha=preflight.digest(run/f'epoch_{m["recipe"]["epochs"]:02d}.pt'), predictions_tta4_sha=preflight.digest(out),
                  single_view_recompute_max_rel_dev=float(rel.max()), stored_score=official(stored, y[va].astype(np.float64)),
                  single_score=official(single, y[va].astype(np.float64)), tta4_score=official(tta, y[va].astype(np.float64)))
    report['tta_delta_pp'] = report['tta4_score']['mean']-report['single_score']['mean']
    (run/'tta4_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(root=str(root), single=report['single_score']['mean'], tta4=report['tta4_score']['mean'], delta=report['tta_delta_pp'])), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('roots', nargs='+', type=Path)
    for r in ap.parse_args().roots:
        main(r)
