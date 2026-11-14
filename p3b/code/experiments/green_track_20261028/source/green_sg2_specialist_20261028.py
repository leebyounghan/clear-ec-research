"""GREEN stride-2 point specialist (one GPU, per fold): same recipe as the stride-8 specialist
(fold-clean recipe A encoder copy, encoder lr 1e-5, head lr 1e-3, AdamW wd 1e-4, clip 1, bf16, batch 2,
four exact flips, ROI-masked Gaussian BCE, fixed 1000 updates, train-green 20 only) but the head is
Linear 768 -> 16 followed by a 4x4 pixel shuffle, giving a stride-2 heatmap (496 x 656) with
Gaussian targets rasterised at stride 2 (sigma 8 px as before).  Purpose: sub-grid localisation for
the spacing-based CD estimator.  Writes, for every validation row of the fold, the sub-pixel peaks
(x, y, p) at threshold >= 0.2 (NMS radius 12 px on the stride-2 grid) plus the stride-8 max-pooled map
for compatibility; the 5 validation-green images are reported once at the fixed threshold 0.35.

commands: train --fold F --tag T ; infer --fold F --tag T
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import maximum_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import green_point_features_20261027 as G
import green_point_spatial_20261023 as spatial
import green_selftrain_20261028 as ST

ROOT = Path('experiments/green_track_20261028')
UP = 4                                    # pixel-shuffle factor: stride 8 -> stride 2
GRID2 = (G.GRID[0]*UP, G.GRID[1]*UP)      # 496 x 656
STRIDE2, OFFSET2 = spatial.STRIDE/UP, spatial.OFFSET-spatial.STRIDE/2+spatial.STRIDE/UP/2   # 2.0, 0.5
ROWS2, COLS2 = int((G.HEIGHT-1-OFFSET2)//STRIDE2)+1, int((G.WIDTH-1-OFFSET2)//STRIDE2)+1   # in-frame cells
UPDATES, BATCH = 1000, 2
EVAL_THRESHOLD, STORE_THRESHOLD, NMS2 = 0.35, 0.2, 12.0


def log(msg, **kw):
    print(json.dumps(dict(utc=G.now(), msg=msg, **kw)), flush=True)


class Head(torch.nn.Module):
    def __init__(self, seed=G.SEED):
        super().__init__()
        state = torch.random.get_rng_state()
        try:
            torch.random.default_generator.manual_seed(seed)
            self.lin = torch.nn.Linear(768, UP*UP)
        finally:
            torch.random.set_rng_state(state)

    def forward(self, tok):                     # tok: B, N, 768 -> B, 496, 656 logits
        b = tok.shape[0]
        x = self.lin(tok.float()).reshape(b, G.GRID[0], G.GRID[1], UP*UP).permute(0, 3, 1, 2)
        return F.pixel_shuffle(x, UP).squeeze(1)


def target2(points, roi):
    """Gaussian max target on the stride-2 grid and the ROI support (inclusive box)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    box = np.asarray(roi, dtype=np.float64)
    yy, xx = np.indices(GRID2, dtype=np.float64)
    xx = OFFSET2+STRIDE2*xx
    yy = OFFSET2+STRIDE2*yy
    valid = (xx >= box[0]) & (xx <= box[2]) & (yy >= box[1]) & (yy <= box[3])
    t = np.zeros(GRID2, dtype=np.float64)
    r = int(4*spatial.SIGMA/STRIDE2)+1
    for px, py in pts:
        cx, cy = int(round((px-OFFSET2)/STRIDE2)), int(round((py-OFFSET2)/STRIDE2))
        x0, x1, y0, y1 = max(0, cx-r), min(GRID2[1], cx+r+1), max(0, cy-r), min(GRID2[0], cy+r+1)
        if x1 <= x0 or y1 <= y0:
            continue
        g = np.exp(-((xx[y0:y1, x0:x1]-px)**2+(yy[y0:y1, x0:x1]-py)**2)/(2*spatial.SIGMA**2))
        t[y0:y1, x0:x1] = np.maximum(t[y0:y1, x0:x1], g)
    assert valid.any()
    return t.astype(np.float32), valid


def peaks2(prob, support, threshold, radius=NMS2):
    """Local maxima within +-2 cells (+-4 px) >= threshold inside support, then greedy NMS at radius px.
    Position refined by the probability-weighted 3x3 centroid.  Returns (n, 3): x, y, p."""
    p = prob.astype(np.float32)
    cand = support & (p >= threshold) & (p == maximum_filter(p, size=5, mode='nearest'))
    yy, xx = np.nonzero(cand)
    if len(yy) == 0:
        return np.zeros((0, 3))
    order = np.lexsort((xx, yy, -p[yy, xx]))
    yy, xx = yy[order], xx[order]
    pts = []
    for iy, ix in zip(yy, xx):
        y0, y1, x0, x1 = max(0, iy-1), min(GRID2[0], iy+2), max(0, ix-1), min(GRID2[1], ix+2)
        w = p[y0:y1, x0:x1].astype(np.float64)
        gy, gx = np.mgrid[y0:y1, x0:x1]
        pts.append([OFFSET2+STRIDE2*(w*gx).sum()/w.sum(), OFFSET2+STRIDE2*(w*gy).sum()/w.sum(), float(p[iy, ix])])
    pts = np.array(pts)
    keep = np.ones(len(pts), bool)
    # greedy NMS on candidates sorted by probability; vectorised distance to kept points
    kept = []
    for i in range(len(pts)):
        if kept:
            d = np.hypot(pts[kept, 0]-pts[i, 0], pts[kept, 1]-pts[i, 1])
            if (d <= radius).any():
                keep[i] = False
                continue
        kept.append(i)
    return pts[keep]


def flip_xy(x, h, v):
    if h:
        x = torch.flip(x, (-1,))
    if v:
        x = torch.flip(x, (-2,))
    return x


def unflip_map(m, h, v):
    """Map a stride-2 heatmap of a flipped image back to unflipped coordinates.  The grid covers the padded
    frame (right/bottom padding), so a flipped image's cell k corresponds to pixel 2k+0.5 of the flipped image =
    pixel W-1-(2k+0.5) of the original, which is not a grid centre; nearest-cell resampling (offset by one cell
    where needed) is used.  Exact for the in-frame region up to a half-cell shift, which the 3x3 centroid absorbs."""
    m = m.copy()
    if h:
        # original pixel for cell k: W-1-(0.5+2k) = 1294.5-2k -> cell index (1294.5-2k-0.5)/2 = 647-k
        src = np.arange(GRID2[1])
        dst = 647-src
        ok = (dst >= 0) & (dst < GRID2[1])
        out = np.zeros_like(m)
        out[:, :, dst[ok]] = m[:, :, src[ok]]
        m = out
    if v:
        src = np.arange(GRID2[0])
        dst = 485-src
        ok = (dst >= 0) & (dst < GRID2[0])
        out = np.zeros_like(m)
        out[:, dst[ok], :] = m[:, src[ok], :]
        m = out
    return m


def frame_support():
    s = np.zeros(GRID2, bool)
    s[:ROWS2, :COLS2] = True
    return s


def heat(encoder, head, x):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        tok = encoder.forward_features(G.b.inputs(x, 'cpdino_vitb'))['x_norm_patchtokens']
    with torch.no_grad():
        return head(tok).sigmoid().cpu().numpy()


def green_targets(ann, names):
    out = {}
    for n in names:
        for f, (h, v) in enumerate(G.FLIPS):
            pts, roi = G.flip_annotation(ann[n]['markers_mha'], ann[n]['roi_box_mha'], h, v)
            out[(n, f)] = target2(pts, roi)
    return out


def rows_for(encoder, head, names, ann, ids, y, threshold):
    lookup = {n: i for i, n in enumerate(ids)}
    rows = []
    for n in names:
        x = G.b.Frames(np.array([lookup[n]]), y, len(ids))[0][0][None]
        p = heat(encoder, head, x)[0]
        _, support = target2(ann[n]['markers_mha'], ann[n]['roi_box_mha'])
        found = peaks2(p, support, threshold)[:, :2]
        rows.append(dict(id=n, predicted=len(found), annotated=len(ann[n]['markers_mha']), **spatial.match_points(found, ann[n]['markers_mha'])))
    return dict(rows=rows, mean_f1=float(np.mean([r['f1'] for r in rows])), mean_recall=float(np.mean([r['recall'] for r in rows])),
                mean_precision=float(np.mean([r['precision'] for r in rows])))


def cmd_train(a):
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    out = ROOT/a.tag/f'fold{a.fold}'
    out.mkdir(parents=True, exist_ok=True)
    ann, binding, ids, y, tr, va = ST.green(a.fold)
    names = binding['train_ids']
    lookup = {n: i for i, n in enumerate(ids)}
    frames = G.b.Frames(np.array([lookup[n] for n in names]), y, len(ids))
    images = {n: frames[i][0][None] for i, n in enumerate(names)}
    targets = {k: (torch.from_numpy(t).cuda(), torch.from_numpy(s).cuda()) for k, (t, s) in green_targets(ann, names).items()}
    encoder = G.load_encoder(ST.ENCODERS[a.fold]).train().requires_grad_(True)
    head = Head(seed=a.seed).cuda()
    opt = torch.optim.AdamW([{'params': encoder.parameters(), 'lr': 1e-5}, {'params': head.parameters(), 'lr': 1e-3}], weight_decay=1e-4)
    rng = np.random.default_rng(a.seed)
    items = [(n, f) for n in names for f in range(4)]
    losses, t0 = [], time.time()
    for step in range(UPDATES):
        pick = [items[i] for i in rng.choice(len(items), BATCH, replace=False)]
        x = torch.cat([G.flip_image(images[n], f) for n, f in pick])
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tok = encoder.forward_features(G.b.inputs(x, 'cpdino_vitb'))['x_norm_patchtokens']
        logits = head(tok)
        loss = sum(F.binary_cross_entropy_with_logits(logits[i][targets[k][1]], targets[k][0][targets[k][1]]) for i, k in enumerate(pick))/len(pick)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(head.parameters()), 1.0, error_if_nonfinite=True)
        opt.step()
        losses.append(float(loss))
        if step % 100 == 0:
            log('train', step=step, loss=float(loss), seconds=round(time.time()-t0, 1))
    encoder.eval().requires_grad_(False)
    head.eval()
    val = rows_for(encoder, head, binding['validation_ids'], ann, ids, y, EVAL_THRESHOLD)
    torch.save(dict(encoder=encoder.state_dict(), head=head.state_dict(), threshold=EVAL_THRESHOLD, init_encoder=ST.ENCODERS[a.fold], up=UP), out/'specialist_sg2.pt')
    G.save(out/'train.json', dict(created_utc=G.now(), fold=a.fold, init_encoder=ST.ENCODERS[a.fold], init_sha=G.sha(ST.ENCODERS[a.fold]),
           updates=UPDATES, batch=BATCH, threshold=EVAL_THRESHOLD, validation=val, final_train_loss_last100=float(np.mean(losses[-100:])), seed=a.seed,
           seconds=round(time.time()-t0, 1), head='Linear 768->16 + pixel_shuffle 4 (stride 2)', stride2_grid=list(GRID2)))
    log('trained', fold=a.fold, val_f1=val['mean_f1'])


def cmd_infer(a):
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    out = ROOT/a.tag/f'fold{a.fold}'
    ann, binding, ids, y, tr, va = ST.green(a.fold)
    ck = torch.load(out/'specialist_sg2.pt', map_location='cpu', weights_only=False)
    encoder = G.load_encoder(ST.ENCODERS[a.fold])
    encoder.load_state_dict(ck['encoder'], strict=True)
    head = Head().cuda()
    head.load_state_dict(ck['head'])
    head.eval()
    support = frame_support()
    dl = DataLoader(G.b.Frames(va, y, len(ids)), batch_size=4, shuffle=False, num_workers=3, pin_memory=True)
    peaks, offsets, seen, low = [], [0], [], []
    t0 = time.time()
    for i, (x, _, ix) in enumerate(dl):
        if a.flips == 1:
            p = heat(encoder, head, x)
        else:
            acc = None
            for h, v in G.FLIPS[:a.flips]:
                m = heat(encoder, head, G.flip_image(G.flip_image(x, 0), 0) if False else flip_xy(x, h, v))
                m = unflip_map(m, h, v)
                acc = m if acc is None else acc+m
            p = acc/a.flips
        for m in p:
            pk = peaks2(m, support, STORE_THRESHOLD)
            peaks.append(pk.astype(np.float32))
            offsets.append(offsets[-1]+len(pk))
            low.append(m.reshape(G.GRID[0], UP, G.GRID[1], UP).max((1, 3)).astype(np.float16))
        seen.append(ix.numpy())
        if i % 100 == 0:
            log('infer', batch=i, of=len(dl), seconds=round(time.time()-t0, 1))
    seen = np.concatenate(seen)
    np.testing.assert_array_equal(seen, va)
    train_green = set(binding['train_ids'])
    path = out/(f'fold{a.fold}_sg2_peaks.npz' if a.flips == 1 else f'fold{a.fold}_sg2_peaks_tta{a.flips}.npz')
    np.savez_compressed(path, indices=va, ids=ids[va], peaks=np.concatenate(peaks) if peaks else np.zeros((0, 3), np.float32),
                        offsets=np.array(offsets), probability_stride8_max=np.stack(low),
                        point_labels_seen=np.array([n in train_green for n in ids[va]]), store_threshold=STORE_THRESHOLD, nms_radius=NMS2)
    G.save(out/('infer.json' if a.flips == 1 else f'infer_tta{a.flips}.json'), dict(created_utc=G.now(), fold=a.fold, n=int(len(va)), flips=a.flips, peaks_sha=G.sha(path), seconds=round(time.time()-t0, 1),
           peaks_per_image=dict(q10=float(np.quantile(np.diff(offsets), .1)), q50=float(np.quantile(np.diff(offsets), .5)), q90=float(np.quantile(np.diff(offsets), .9)))))
    log('inferred', fold=a.fold, path=str(path))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['train', 'infer'])
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--tag', default='sg2_specialist')
    ap.add_argument('--seed', type=int, default=G.SEED)
    ap.add_argument('--flips', type=int, default=1, choices=[1, 2, 4])
    a = ap.parse_args()
    {'train': cmd_train, 'infer': cmd_infer}[a.command](a)
