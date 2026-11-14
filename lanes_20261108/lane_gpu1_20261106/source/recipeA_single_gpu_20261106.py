"""Run the frozen two-rank recipe A runner in one process on one GPU (lane GPU 0, 2026-11-06).

The frozen runner hard-codes two ranks, each doing two micro-batches of 4 out of a global batch of 16, weighted by count.
The objective is a per-sample mean and the network has no batch norm or dropout, so the gradient is the sample-weighted mean over the 16 frames. The same micro-batches run in the same order of random draws as the two ranks; see backward(). The smoke step checks the gradient against a micro-batch-4 reference (relative L2 < 0.002).
Patches: init_dist, dist, DDP, distributed_backward, and the loader's worker count (2 for training as in each frozen rank, 2 for evaluation instead of 4, for the lane's CPU limit). Order, flips, schedule, clip, loss and validation are unchanged.
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
MICRO = 4  # the same micro-batches as the two ranks; grouping by 8 differed by 0.0023 relative L2 under bf16 and failed the smoke gate
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
    """The two ranks of the frozen runner, one after the other. The encoder draws a random RoPE coordinate rescale in [1/2, 2] at every forward in train mode, from the GPU generator, and both ranks hold the same generator state, so rank 1 repeats rank 0's draws. Each rank here starts from the same state and makes the same two forwards, including the zero-weight dummy, so the draws match the reference run."""
    n = len(t)
    stats = torch.zeros(7, device=mu.device, dtype=torch.float64)
    state = torch.cuda.get_rng_state()
    for r in (0, 1):
        torch.cuda.set_rng_state(state)
        for j in range(2):
            start = (2 * j + r) * MICRO
            count = max(0, min(MICRO, n - start))
            xx, yy = (x[start:start + count], t[start:start + count]) if count else (x[:1], t[:1])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = ddp(b.inputs(xx, R.ARCH))
                loss = b.loss_fn(z, yy.cuda(non_blocking=True), mu, sigma)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            if count:
                with torch.no_grad():
                    zt = (yy.cuda(non_blocking=True).log() - mu) / sigma
                    rel = torch.expm1((z.float() - zt) * sigma).abs()
                    stats[0] += loss.detach().double() * count
                    stats[1:4] += rel.double().sum(0) * 100
                    stats[4:7] += (z.float() - zt).abs().double().sum(0)
            (loss * (count / n)).backward()
    return stats


def loader(indices, targets, total, epoch=0, augment=False, batch_size=4):
    return DataLoader(R.Frames(indices, targets, total, epoch, augment, R.ARGS.seed), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
                      prefetch_factor=2, drop_last=False)


def source_files():
    return _SOURCES() + [SELF]


def save(path, obj):
    if Path(path).name == 'manifest.json' and isinstance(obj, dict) and 'recipe' in obj:
        obj['recipe']['micro_batch'] = MICRO
        obj['execution'] = dict(world_size=1, micro_batch_per_gpu=MICRO, global_batch=16, accumulation_rounds=4,
                                partial_groups='sample weighted', single_gpu_note='same gradient as the two-rank runner; see recipeA_single_gpu_20261106.py')
    return _SAVE(path, obj)


R.init_dist = init_dist
R.DDP = lambda net, **kwargs: net
R.distributed_backward = backward
R.loader = loader
R.source_files = source_files
R.save = save
