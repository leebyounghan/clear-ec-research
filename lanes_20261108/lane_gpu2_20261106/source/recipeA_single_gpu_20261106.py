"""Run the frozen two-rank recipe A runner in one process on one GPU (lane GPU 0, 2026-11-06).

The frozen runner hard-codes two ranks, each doing two micro-batches of 4 out of a global batch of 16, weighted by count.
The objective is a per-sample mean and the network has no batch norm or dropout, so the gradient is the sample-weighted mean over the 16 frames however they are grouped. The smoke step checks that against a micro-batch-4 reference (relative L2 < 0.002).
Patches: init_dist, dist, DDP, distributed_backward, and the train loader's worker count (4 in one loader instead of 2 in each of two ranks, both of which decoded all 16 frames). Order, flips, schedule, clip, loss and validation are unchanged.
Import this after the runner, then set R.main as usual."""
import os
from pathlib import Path
import types

import numpy as np
import torch
from torch.utils.data import DataLoader

import phase2_foundation_train_20261002 as b
import phase2_pooling_regression_20261019 as R

SELF = Path(__file__).resolve()
MICRO = 8
_SOURCES = R.source_files
_SAVE = R.save


def init_dist():
    torch.cuda.set_device(0)
    assert torch.cuda.device_count() == 1, 'expects exactly one visible GPU'
    return 0


def _gather(out, value):
    out[:] = [value] * len(out)


R.dist = types.SimpleNamespace(barrier=lambda: None, destroy_process_group=lambda: None, broadcast=lambda *a, **k: None, all_reduce=lambda *a, **k: None,
                               all_gather_object=_gather, broadcast_object_list=lambda *a, **k: None)


def backward(ddp, x, t, mu, sigma, rank):
    n = len(t)
    stats = torch.zeros(7, device=mu.device, dtype=torch.float64)
    for start in range(0, n, MICRO):
        xx, yy = x[start:start + MICRO], t[start:start + MICRO]
        count = len(yy)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = ddp(b.inputs(xx, R.ARCH))
            loss = b.loss_fn(z, yy.cuda(non_blocking=True), mu, sigma)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        with torch.no_grad():
            zt = (yy.cuda(non_blocking=True).log() - mu) / sigma
            rel = torch.expm1((z.float() - zt) * sigma).abs()
            stats[0] += loss.detach().double() * count
            stats[1:4] += rel.double().sum(0) * 100
            stats[4:7] += (z.float() - zt).abs().double().sum(0)
        (loss * (count / n)).backward()
    return stats


def loader(indices, targets, total, epoch=0, augment=False, batch_size=4):
    return DataLoader(R.Frames(indices, targets, total, epoch, augment, R.ARGS.seed), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                      prefetch_factor=2, drop_last=False)


def source_files():
    return _SOURCES() + [SELF]


def save(path, obj):
    if Path(path).name == 'manifest.json' and isinstance(obj, dict) and 'recipe' in obj:
        obj['recipe']['micro_batch'] = MICRO
        obj['execution'] = dict(world_size=1, micro_batch_per_gpu=MICRO, global_batch=16, accumulation_rounds=2,
                                partial_groups='sample weighted', single_gpu_note='same gradient as the two-rank runner; see recipeA_single_gpu_20261106.py')
    return _SAVE(path, obj)


R.init_dist = init_dist
R.DDP = lambda net, **kwargs: net
R.distributed_backward = backward
R.loader = loader
R.source_files = source_files
R.save = save
