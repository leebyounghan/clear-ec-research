"""Green day 2 / G1 + G2: full-frame (pre-trained-only) detectors.
  infer-pre-tta  --family {s46,s47,s48,aug_s46} --fold F [--flips 4] [--limit N] [--tag T]
      four-flip TTA stride-2 heatmap of a pre-only detector on the held fold rows; peaks stored at 0.2 (sg2 format).
  pretrain-aug   --fold F --seed S --updates 2000
      srep pre-training (as green_external_srep_20261030.pretrain) with synthetic black frames / elliptical vignettes
      (blacked-out cells become negatives), Gaussian blur and contrast jitter; saves green_day2_20261101/sg2pre_aug/fold{F}/sg2_pre.pt
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, 'experiments/green_track_20261028/source')
import green_point_features_20261027 as G
import green_selftrain_20261028 as ST
import green_sg2_specialist_20261028 as SG
import green_external_srep_20261030 as E

ROOT = Path('experiments/green_day2_20261101'); GT = Path('experiments/green_track_20261028')
PRE = {'s46': lambda f: GT/'external_srep'/f'fold{f}'/'sg2_pre.pt',
       's47': lambda f: GT/'external_srep'/'s47'/f'fold{f}'/'sg2_pre.pt',
       's48': lambda f: GT/'external_srep'/'s48'/f'fold{f}'/'sg2_pre.pt',
       's49': lambda f: GT/'external_srep'/'s49'/f'fold{f}'/'sg2_pre.pt',
       's50': lambda f: GT/'external_srep'/'s50'/f'fold{f}'/'sg2_pre.pt',
       'aug_s46': lambda f: ROOT/'sg2pre_aug'/f'fold{f}'/'sg2_pre.pt',
       'aug_s47': lambda f: ROOT/'sg2pre_aug_s47'/f'fold{f}'/'sg2_pre.pt',
       'aug_s48': lambda f: ROOT/'sg2pre_aug_s48'/f'fold{f}'/'sg2_pre.pt'}


def log(msg, **kw):
    print(json.dumps(dict(utc=G.now(), msg=msg, **kw)), flush=True)


def flip_xy(x, h, v):
    if h:
        x = torch.flip(x, (-1,))
    if v:
        x = torch.flip(x, (-2,))
    return x


def infer_pre_tta(a):
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    ck = torch.load(PRE[a.family](a.fold), map_location='cpu', weights_only=False)
    ann, binding, ids, y, tr, va = ST.green(a.fold)
    if a.limit:
        va = va[:a.limit]
    encoder = G.load_encoder(ST.ENCODERS[a.fold]); encoder.load_state_dict(ck['encoder'], strict=True); encoder.eval().requires_grad_(False)
    head = SG.Head(seed=int(ck.get('seed', G.SEED))).cuda(); head.load_state_dict(ck['head']); head.eval()
    support = SG.frame_support()
    out = ROOT/'preonly_tta'/(a.family+('_'+a.tag if a.tag else ''))/f'fold{a.fold}'
    out.mkdir(parents=True, exist_ok=True)
    dl = DataLoader(G.b.Frames(va, y, len(ids)), batch_size=4, shuffle=False, num_workers=3, pin_memory=True)
    peaks, offsets, seen = [], [0], []
    t0 = time.time()
    for i, (x, _, ix) in enumerate(dl):
        acc = None
        for h, v in G.FLIPS[:a.flips]:
            m = SG.unflip_map(SG.heat(encoder, head, flip_xy(x, h, v)), h, v)
            acc = m if acc is None else acc+m
        p = acc/a.flips
        for m in p:
            pk = SG.peaks2(m, support, 0.2)
            peaks.append(pk.astype(np.float32)); offsets.append(offsets[-1]+len(pk))
        seen.append(ix.numpy())
        if i % 100 == 0:
            log('infer_tta', family=a.family, fold=a.fold, batch=i, of=len(dl), seconds=round(time.time()-t0, 1))
    np.testing.assert_array_equal(np.concatenate(seen), va)
    path = out/f'fold{a.fold}_sg2_peaks.npz'
    np.savez_compressed(path, indices=va, ids=ids[va], peaks=np.concatenate(peaks) if peaks else np.zeros((0, 3), np.float32), offsets=np.array(offsets),
                        point_labels_seen=np.zeros(len(va), bool), store_threshold=0.2, nms_radius=12.0)
    d = np.diff(offsets)
    (out/'infer_tta.json').write_text(json.dumps(dict(created_utc=G.now(), family=a.family, fold=a.fold, flips=a.flips, n=int(len(va)), checkpoint=str(PRE[a.family](a.fold)),
                                                      seconds=round(time.time()-t0, 1), seconds_per_image=round((time.time()-t0)/max(1, len(va)), 3),
                                                      peaks_per_image=dict(q10=float(np.quantile(d, .1)), q50=float(np.quantile(d, .5)), q90=float(np.quantile(d, .9)))), indent=1))
    log('infer_tta done', family=a.family, fold=a.fold, n=int(len(va)), seconds=round(time.time()-t0, 1))


def frame_aug(img, rng):
    """img float32 [0,1] (H, W).  Returns augmented image and the blacked-out pixel mask."""
    Hh, Ww = img.shape
    out = img.copy(); mask = np.zeros((Hh, Ww), bool)
    if rng.random() < 0.5:
        out = np.clip(out*rng.uniform(0.8, 1.25)+rng.uniform(-0.08, 0.08), 0, 1)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.5, 1.5)))
    r = rng.random()
    if r < 0.45:
        t = [int(rng.uniform(0, 0.22)*Hh), int(rng.uniform(0, 0.22)*Hh), int(rng.uniform(0, 0.22)*Ww), int(rng.uniform(0, 0.22)*Ww)]
        if t[0]: mask[:t[0], :] = True
        if t[1]: mask[Hh-t[1]:, :] = True
        if t[2]: mask[:, :t[2]] = True
        if t[3]: mask[:, Ww-t[3]:] = True
    elif r < 0.7:
        cy, cx = Hh/2+rng.uniform(-0.1, 0.1)*Hh, Ww/2+rng.uniform(-0.1, 0.1)*Ww
        ay, ax = rng.uniform(0.45, 0.75)*Hh, rng.uniform(0.45, 0.75)*Ww
        yy, xx = np.ogrid[:Hh, :Ww]
        mask = (((yy-cy)/ay)**2+((xx-cx)/ax)**2) > 1
    out[mask] = 0.0
    return out.astype(np.float32), mask


def pretrain_aug(a):
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    out = ROOT/('sg2pre_aug' if a.seed == 46 else f'sg2pre_aug_s{a.seed}')/f'fold{a.fold}'
    out.mkdir(parents=True, exist_ok=True)
    imgs, pts, masks, shape = E.load_cache(a.scale)
    stride, offset = 2, 0.5
    init = Path(ST.ENCODERS[a.fold])
    encoder = G.load_encoder(init).train().requires_grad_(True)
    head = SG.Head(seed=a.seed).cuda()
    opt = torch.optim.AdamW([{'params': encoder.parameters(), 'lr': 1e-5}, {'params': head.parameters(), 'lr': 1e-3}], weight_decay=1e-4)
    rng = np.random.default_rng(a.seed+1000*a.fold)
    gh2, gw2 = shape[0]//2, shape[1]//2
    ix = np.clip(np.rint(offset+stride*np.arange(gw2)).astype(int), 0, shape[1]-1); iy = np.clip(np.rint(offset+stride*np.arange(gh2)).astype(int), 0, shape[0]-1)
    n = len(imgs); losses = []; t0 = time.time(); cache = {}; n_aug = 0
    for step in range(a.updates):
        idx = rng.choice(n, 2, replace=False); flips = rng.integers(4, size=2)
        xs, ts, ss = [], [], []
        for i, f in zip(idx, flips):
            h, v = G.FLIPS[f]
            key = (int(i), int(f))
            if key not in cache:
                p, m = E.flip_pts(pts[i], masks[i], shape, h, v)
                t, sup = E.rasterise(p, m, shape, stride, offset)
                cache[key] = (torch.from_numpy(t).cuda(), torch.from_numpy(sup).cuda())
                if len(cache) > 400:
                    cache.pop(next(iter(cache)))
            img = np.asarray(imgs[i], dtype=np.float32)
            if h:
                img = img[:, ::-1]
            if v:
                img = img[::-1, :]
            img = np.ascontiguousarray(img)
            img, bm = frame_aug(img, rng)
            cm = torch.from_numpy(np.ascontiguousarray(bm[iy][:, ix])).cuda()
            t, sup = cache[key]
            if bool(cm.any()):
                n_aug += 1
            ts.append(torch.where(cm, torch.zeros_like(t), t)); ss.append(sup | cm)
            xs.append(torch.from_numpy(img)[None, None])
        x = torch.cat(xs)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tok = encoder.forward_features(G.b.inputs(x, 'cpdino_vitb'))['x_norm_patchtokens']
        gh, gw = (shape[0]+20)//8, (shape[1]+16)//8
        b = tok.shape[0]
        logits = F.pixel_shuffle(head.lin(tok.float()).reshape(b, gh, gw, 16).permute(0, 3, 1, 2), 4).squeeze(1)
        loss = 0
        for j in range(2):
            t, sup = ts[j], ss[j]
            lg = logits[j][:t.shape[0], :t.shape[1]]
            loss = loss+F.binary_cross_entropy_with_logits(lg[sup], t[sup])
        loss = loss/2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(head.parameters()), 1.0, error_if_nonfinite=True)
        opt.step()
        losses.append(float(loss))
        if step % 200 == 0:
            log('pretrain_aug', fold=a.fold, seed=a.seed, step=step, loss=float(loss), seconds=round(time.time()-t0, 1))
    torch.save(dict(encoder=encoder.state_dict(), head=head.state_dict(), kind='sg2', features='last', seed=a.seed, threshold=0.35, scale=a.scale, updates=a.updates,
                    init_encoder=str(init), init_sha=G.sha(init), mode='pre-only (srep) + frame augmentation, green day 2'), out/'sg2_pre.pt')
    (out/'sg2_pretrain.json').write_text(json.dumps(dict(created_utc=G.now(), fold=a.fold, seed=a.seed, updates=a.updates, n_images=int(n), scale=a.scale, init_encoder=str(init),
                                                         frac_batches_with_blackout=round(n_aug/(2*a.updates), 3), loss_last100=float(np.mean(losses[-100:])), seconds=round(time.time()-t0, 1)), indent=1))
    log('pretrain_aug done', fold=a.fold, seed=a.seed, seconds=round(time.time()-t0, 1), loss_last100=float(np.mean(losses[-100:])))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['infer-pre-tta', 'pretrain-aug'])
    ap.add_argument('--family', default='s46'); ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--flips', type=int, default=4, choices=[1, 2, 4]); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--tag', default='')
    ap.add_argument('--seed', type=int, default=46); ap.add_argument('--updates', type=int, default=2000); ap.add_argument('--scale', type=float, default=E.SCALE_NOMINAL)
    a = ap.parse_args()
    {'infer-pre-tta': infer_pre_tta, 'pretrain-aug': pretrain_aug}[a.command](a)
