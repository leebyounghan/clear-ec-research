"""Inference-only probe (2026-10-31): does excluding near-black patch tokens from recipe A's std pooling help on
black-frame images? Same trained checkpoint, same single view, two poolings on identical tokens; no weights changed.
Hypothesis: recipe A loses its advantage over Phase-1 on any image with a black frame (A - P1: -0.21 without frame,
+0.04..+0.17 with), not in proportion to the frame size, and the 1/99 percentile window changes only 4 %; std
pooling over all tokens is the remaining mechanism. A gain here (despite the train/test pooling mismatch) would
justify retraining with masked pooling; a loss would be inconclusive (mismatch).
"""
import json, math, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

SRC = 'experiments/tta_recipeA_20261027/source'
sys.path.insert(0, SRC)
import phase2_foundation_train_20261002 as b
import phase2_foundation_preflight_20261002 as preflight
import phase2_pooling_20261019 as pooling

BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def official(pred, y, rows=None):
    if rows is not None:
        pred, y = pred[rows], y[rows]
    s = y > 0
    v = np.nanmean(np.where(s, np.abs(pred - y) / np.where(s, y, 1.), np.nan), 0) * 100
    return dict(CD=round(float(v[0]), 4), CV=round(float(v[1]), 4), HEX=round(float(v[2]), 4), mean=round(float(v.mean()), 4), n=int(len(y)))


def run(root, out_dir):
    m = json.loads((root / 'manifest.json').read_text())
    run_dir = root / 'runs' / f'fold{m["fold"]}' / 'cpdino_vitb'
    ck = torch.load(run_dir / f'epoch_{m["recipe"]["epochs"]:02d}.pt', map_location='cpu', weights_only=True)
    net, _, _ = preflight.build('cpdino_vitb', False, native_convnext_norm=True)
    net[0] = pooling.SpatialPoolEncoder(net[0].original, m['recipe']['pooling'])
    assert m['recipe']['pooling'] == 'std'
    net.load_state_dict(ck['model'], strict=True)
    net = net.cuda().eval().requires_grad_(False)
    assert isinstance(net, torch.nn.Sequential)
    enc, head = net[0].original, net[1:]
    with np.load(b.META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    with np.load(run_dir / 'predictions.npz') as z:
        va, stored = z['indices'].copy(), z['pred'].astype(np.float64)
    raw = np.memmap(b.RAW, mode='r', dtype=np.uint8, shape=(len(ids), 972, 1296))
    dl = torch.utils.data.DataLoader(b.Frames(va, y, len(ids)), batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    za, zm, seen, kept = [], [], [], []
    for x, _, ix in dl:
        xin = b.inputs(x, 'cpdino_vitb')
        with torch.no_grad(), sdpa_kernel(BACKENDS), torch.autocast('cuda', dtype=torch.bfloat16):
            tok = enc.forward_features(xin)['x_norm_patchtokens']
            B, N, C = tok.shape
            p = int(round(math.sqrt(xin.shape[-2] * xin.shape[-1] / N)))
            gh, gw = xin.shape[-2] // p, xin.shape[-1] // p
            assert gh * gw == N, (gh, gw, N)
            r = np.stack([np.asarray(raw[int(i)]) for i in ix.numpy()])
            r = np.pad(r, ((0, 0), (0, xin.shape[-2] - 972), (0, xin.shape[-1] - 1296)), mode='edge')
            dark = (r < 5).reshape(B, gh, p, gw, p).mean((2, 4)) > 0.5
            keep = torch.from_numpy(~dark.reshape(B, N)).cuda()
            keep[keep.sum(1) == 0] = True
            std_all = pooling.pool_tokens(tok, 'std')
            std_mask = torch.stack([(tok[i][keep[i]].float().var(0, unbiased=False) + pooling.STD_EPS).sqrt().to(tok.dtype)
                                    for i in range(B)])
            za.append(head(std_all).float().cpu().numpy()); zm.append(head(std_mask).float().cpu().numpy())
        seen.append(ix.numpy()); kept.append(keep.float().mean(1).cpu().numpy())
    seen = np.concatenate(seen); np.testing.assert_array_equal(seen, va)
    lm, ls = np.array(ck['log_mean']), np.array(ck['log_std'])
    pa, pm = np.exp(np.concatenate(za) * ls + lm), np.exp(np.concatenate(zm) * ls + lm)
    kept = np.concatenate(kept)
    rel = np.abs(pa - stored) / stored
    assert rel.max() < 0.01, f'standard pooling does not reproduce stored predictions: {rel.max()}'
    yy = y[va].astype(np.float64)
    frame = kept < 0.99
    res = dict(root=str(root), fold=m['fold'], seed=m['seed'], recompute_max_rel_dev=float(rel.max()),
               frame_fraction=float(frame.mean()), dark_token_fraction_median_in_frame=float(np.median(1 - kept[frame])),
               all=dict(standard=official(pa, yy), masked=official(pm, yy)),
               frame=dict(standard=official(pa, yy, frame), masked=official(pm, yy, frame)),
               no_frame=dict(standard=official(pa, yy, ~frame), masked=official(pm, yy, ~frame)))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f'fold{m["fold"]}_s{m["seed"]}.npz', indices=va, pred_standard=pa, pred_masked=pm, kept_fraction=kept)
    (out_dir / f'fold{m["fold"]}_s{m["seed"]}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


if __name__ == '__main__':
    out = Path(sys.argv[1])
    for r in sys.argv[2:]:
        run(Path(r), out)
