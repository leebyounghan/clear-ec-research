#!/usr/bin/env python3
"""Synthetic-image pretraining stage for DINOv3-B recipe A (backbone lane, 2026-11-11; coordinator; KG ds-simulated-tessellations,
bucht2011simulation, kucharski2023gan). From a fold's DINOv3-B SSL5 weights, regress the exact (CD, area CV, HEX) of synthetic
frames (synth_specular_20261111.py) with recipe A's network (std pool, LN-128-GELU-3 head), loss (0.5 |dz| + 2.5 |expm1(dz sigma)|
on standardised log targets), input pipeline (CLAHE + 1-99 percentile, H/V flips, replicate pad, ImageNet normalisation),
AdamW (encoder 1e-5, head 1e-3, wd 1e-4), effective batch 16, bf16, clip 1. One pass over the training frames, LR warm-up 5 %
then cosine to 0. The last --holdout frames are a synthetic read (never trained). Saves {'tokens': encoder state} that
bb_runner_20261108.py loads with --ssl for recipe A fine-tuning on our labels (the head is re-initialised there, as always).
Usage: synth_pretrain_20261111.py --ssl SSL.pt --data PREFIX --out OUT.pt [--seed 46] [--micro 8] [--workers 4] [--holdout 1000]
Run from /workspace/clear-ec-research with one visible GPU."""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, 'experiments/lane_backbone_20261108/source')
import family_bb_20261108 as F  # noqa: E402
b = F.b
FAMILY = 'dinov3_b16'  # default; --family overrides (D-SYN-002: dinov3_l16t ViT-L)


class SynthFrames(Dataset):
    def __init__(self, path, n_total, rows, labels, augment, seed, epoch=0):
        self.path, self.n, self.rows, self.labels, self.augment, self.seed, self.epoch, self.mm = path, n_total, rows, labels, augment, seed, epoch, None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        import cv2
        cv2.setNumThreads(1)
        if self.mm is None:
            self.mm = np.memmap(self.path, mode='r', dtype=np.uint8, shape=(self.n, 972, 1296))
        i = int(self.rows[k])
        a = cv2.createCLAHE(2.0, (1296 // 32, 972 // 32)).apply(np.asarray(self.mm[i]))
        lo, hi = np.percentile(a, (1, 99))
        a = np.clip((a.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
        if self.augment:
            g = np.random.default_rng(np.random.SeedSequence([self.seed, 7171, self.epoch, i]))
            if g.random() < .5:
                a = a[:, ::-1]
            if g.random() < .5:
                a = a[::-1, :]
        return torch.from_numpy(np.ascontiguousarray(a[None])), torch.from_numpy(self.labels[i]), i


def log(o):
    print(json.dumps(o), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ssl', required=True); p.add_argument('--data', required=True); p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=46); p.add_argument('--micro', type=int, default=8); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--holdout', type=int, default=1000); p.add_argument('--smoke', type=int, default=0)
    p.add_argument('--family', default='dinov3_b16', choices=list(F.FAMILIES))
    a = p.parse_args()
    global FAMILY
    FAMILY = a.family
    with np.load(a.data + '_labels.npz', allow_pickle=True) as z:
        Y = z['labels'].astype(np.float32); n = int(z['shape'][0])
    assert np.isfinite(Y).all() and (Y > 0).all()
    tr, ho = np.arange(n - a.holdout), np.arange(n - a.holdout, n)
    net = F.build_net(argparse.Namespace(family=FAMILY), a.seed, a.ssl)
    logy = np.log(Y[tr].astype(np.float64)); mu = torch.tensor(logy.mean(0), device='cuda', dtype=torch.float32); sigma = torch.tensor(logy.std(0), device='cuda', dtype=torch.float32)
    opt = torch.optim.AdamW([{'params': net[0].parameters(), 'lr': 1e-5, 'base': 1e-5}, {'params': list(net[1:].parameters()), 'lr': 1e-3, 'base': 1e-3}], weight_decay=1e-4, fused=True)
    order = np.random.default_rng(a.seed).permutation(tr)
    dl = DataLoader(SynthFrames(a.data + '_frames.u8', n, order, Y, True, a.seed), batch_size=16, shuffle=False, num_workers=a.workers, pin_memory=True, prefetch_factor=2, drop_last=True)
    updates = len(order) // 16 if not a.smoke else a.smoke; warm = max(1, int(0.05 * updates))
    log(dict(launch=dict(family=FAMILY, ssl=a.ssl, data=a.data, out=a.out, seed=a.seed, train=len(tr), holdout=len(ho), updates=updates, mu=logy.mean(0).tolist(), sigma=logy.std(0).tolist())))
    net.train(); step = 0; t0 = time.monotonic(); run = []
    for x, t, ix in dl:
        ratio = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, updates - warm)))
        for g in opt.param_groups:
            g['lr'] = g['base'] * ratio
        opt.zero_grad(set_to_none=True); m = len(t)
        for s in range(0, m, a.micro):
            xx, yy = x[s:s + a.micro], t[s:s + a.micro].cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = net(F.inputs(xx, FAMILY))
                loss = b.loss_fn(z[:, :3], yy, mu, sigma)
            with torch.no_grad():
                rel = torch.expm1((z[:, :3].float() - (yy.log() - mu) / sigma) * sigma).abs().mean(0) * 100
            run.append(torch.cat([loss.detach().float().reshape(1), rel]).cpu().numpy())
            (loss * (len(yy) / m)).backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        opt.step(); step += 1
        if step % 100 == 0 or step == updates:
            r = np.mean(run, 0); run = []
            log(dict(step=step, updates=updates, loss=round(float(r[0]), 4), mape=dict(zip(('CD', 'CVa', 'HEX'), np.round(r[1:], 3).tolist())), sec=round(time.monotonic() - t0), peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)))
        if step >= updates:
            break
    if a.smoke:
        log(dict(smoke='done', sec_per_update=round((time.monotonic() - t0) / step, 3))); return
    # synthetic hold-out read
    net.eval(); P = []
    hl = DataLoader(SynthFrames(a.data + '_frames.u8', n, ho, Y, False, a.seed), batch_size=a.micro, shuffle=False, num_workers=a.workers)
    with torch.inference_mode():
        for x, t, ix in hl:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = net(F.inputs(x, FAMILY)).float()[:, :3]
            P.append(torch.exp(z * sigma + mu).cpu().numpy())
    P = np.concatenate(P); T = Y[ho]
    ho_read = {k: dict(mape=round(float(np.mean(np.abs(P[:, j] / T[:, j] - 1)) * 100), 3), r=round(float(np.corrcoef(np.log(P[:, j]), np.log(T[:, j]))[0, 1]), 4)) for j, k in enumerate(('CD', 'CV_area', 'HEX'))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(tokens={k: v.detach().cpu() for k, v in net[0].tokens.state_dict().items()}, head={k: v.detach().cpu() for k, v in net[1:].state_dict().items()},
                    log_mean=logy.mean(0).tolist(), log_std=logy.std(0).tolist(), family=FAMILY, ssl=a.ssl, data=a.data, seed=a.seed, updates=updates, holdout_read=ho_read), a.out)
    log(dict(done=True, out=a.out, holdout_read=ho_read, sec=round(time.monotonic() - t0)))


if __name__ == '__main__':
    main()
