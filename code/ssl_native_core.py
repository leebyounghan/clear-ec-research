"""Native-scale masked EMA features + explicit encoder regularization.

Custom JEPA-inspired ablation, not an official I-JEPA/LeJEPA/VICReg reproduction.
No data paths, labels or validation access in this module.
"""
import copy
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint


def normalized_tokens(features):
    return F.layer_norm(features.float().permute(0, 2, 3, 1), (features.shape[1],))


def penalties(features):
    tokens = normalized_tokens(features)[:, ::8, ::8].reshape(-1, features.shape[1])
    variance = F.relu(.5 - torch.sqrt(tokens.var(0, unbiased=False) + 1e-4)).mean()
    image_features = normalized_tokens(features).mean((1, 2))
    variance = variance + F.relu(.2 - torch.sqrt(image_features.var(0, unbiased=False) + 1e-4)).mean()
    centered = tokens - tokens.mean(0)
    covariance = centered.T @ centered / max(1, len(tokens)-1)
    offdiag = covariance - torch.diag(torch.diag(covariance))
    return variance, offdiag.square().sum() / features.shape[1]


def feature_diagnostics(features):
    values = normalized_tokens(features)
    tokens = values[:, ::8, ::8].reshape(-1, features.shape[1])
    def rank(x):
        s = torch.linalg.svdvals(x - x.mean(0))
        p = s.square() / s.square().sum().clamp_min(1e-12)
        return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))
    image_values = values.mean((1, 2))
    return dict(token_std=float(tokens.std(0, unbiased=False).mean()),
                image_std=float(image_values.std(0, unbiased=False).mean()),
                token_effective_rank=rank(tokens), image_effective_rank=rank(image_values))


def require_healthy(stats, initial):
    if not all(math.isfinite(v) for v in stats.values()):
        raise FloatingPointError('Nonfinite SSL representation diagnostics')
    for key in ('token_std', 'image_std'):
        if stats[key] < max(1e-4, .2 * initial[key]):
            raise FloatingPointError(f'Severe representation contraction: {key}')
    for key in ('token_effective_rank', 'image_effective_rank'):
        if stats[key] < .25 * initial[key]:
            raise FloatingPointError(f'Severe effective rank contraction: {key}')


def mask_batch(batch, size=32):
    masks = np.zeros((batch, 1, size, size), np.float32)
    for m in masks:
        target = int(np.random.uniform(.4, .6) * size * size)
        while m.sum() < target:
            h, w = np.random.randint(4, 11, size=2)
            r, c = np.random.randint(size-h+1), np.random.randint(size-w+1)
            m[:, r:r+h, c:c+w] = 1
    return masks


def pretrain(encoder, views, train_indices, destination, steps, shared):
    """Only train_indices can be sampled. Targets are SAME-crop teacher features."""
    shared.seed(20260911)
    teacher = copy.deepcopy(encoder).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    predictor = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.GroupNorm(32, 512),
                              nn.GELU(), nn.Conv2d(512, 256, 3, padding=1)).cuda()
    for block in encoder.blocks:
        original = block.forward
        block.forward = lambda x, fn=original: checkpoint.checkpoint(fn, x, use_reentrant=False)
    encoder.train()
    parameters = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=.05)
    permitted = set(map(int, train_indices))
    batch = 32
    # Deterministic native crops from TRAIN only; never use downstream validation.
    probe_indices = np.random.default_rng(910).choice(train_indices, size=32, replace=False)
    probe = np.stack([views.tile(int(i), views.selected[i, 0], 0) for i in probe_indices])
    @torch.no_grad()
    def diagnose():
        features = []
        for offset in range(0, len(probe), 8):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                features.append(teacher(torch.from_numpy(probe[offset:offset+8]).cuda()).float())
        return feature_diagnostics(torch.cat(features))
    initial = diagnose()
    require_healthy(initial, initial)
    shared.save_json(destination/'probe_initial.json', initial)
    config = dict(method='custom_masked_EMA_with_online_variance_covariance', updates=steps,
                  batch=batch, lr=1e-5, weight_decay=.05, variance_weight=.1,
                  covariance_weight=.01, token_std_floor=.5, image_std_floor=.2,
                  mask_range=[.4, .6], native256=True, resize=False,
                  clear_ec_train_rows=len(train_indices), external_images=0,
                  validation_images_used=0, same_crop_context_target=True,
                  probe_train_only=True, checkpoint_selection='fixed_terminal')
    shared.save_json(destination/'config.json', config)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    history = [dict(step=0, **initial)]
    for step in range(1, steps+1):
        chosen = np.random.choice(train_indices, size=batch, replace=False)
        assert set(map(int, chosen)) <= permitted
        array = np.stack([views.train(int(i), 1)[0] for i in chosen])
        clean = torch.from_numpy(array).cuda()
        mask = torch.from_numpy(mask_batch(batch)).cuda()
        pixel_mask = F.interpolate(mask, size=(256, 256), mode='nearest')
        gain = torch.empty(batch, 1, 1, 1, device='cuda').uniform_(.95, 1.05)
        offset = torch.empty_like(gain).uniform_(-.02, .02)
        context = (clean * gain + offset).clamp(0, 1) * (1-pixel_mask) + .5 * pixel_mask
        progress = (step-1) / max(1, steps-1)
        lr = 1e-5 * min(step/200, 1) * (.05+.95*.5*(1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            target = teacher(clean)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            features = encoder(context)
            prediction = predictor(features)
        errors = F.smooth_l1_loss(normalized_tokens(prediction), normalized_tokens(target), reduction='none').mean(-1)
        prediction_loss = (errors * mask[:, 0]).sum() / mask.sum()
        variance, covariance = penalties(features)
        loss = prediction_loss + .1*variance + .01*covariance
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite SSL loss')
        loss.backward()
        assert any(p.grad is not None for p in encoder.parameters())
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        ema = .996 + (.9999-.996)*progress
        with torch.no_grad():
            for online, target_parameter in zip(encoder.parameters(), teacher.parameters()):
                target_parameter.mul_(ema).add_(online, alpha=1-ema)
            for online, target_buffer in zip(encoder.buffers(), teacher.buffers()):
                target_buffer.copy_(online)
        if step % 25 == 0 or step == steps:
            elapsed = time.perf_counter()-start
            record = dict(stage='pretraining', step=step, total_steps=steps,
                          elapsed_seconds=elapsed, seconds_per_step=elapsed/step,
                          loss=float(loss), prediction_loss=float(prediction_loss),
                          variance_penalty=float(variance), covariance_penalty=float(covariance),
                          peak_gib=torch.cuda.max_memory_allocated()/2**30,
                          updated_at=time.strftime('%FT%T%z'))
            shared.save_json(destination/'progress.json', record)
            print(record, flush=True)
        if step % 500 == 0 or step == steps:
            stats = diagnose()
            require_healthy(stats, initial)
            history.append(dict(step=step, **stats))
            shared.save_json(destination/'probe_history.json', history)
        if step % 2000 == 0 or step == steps:
            torch.save(dict(teacher_encoder=teacher.state_dict(), config=config, step=step),
                       destination/f'encoder_step{step:04d}.pt')
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    final = destination/f'encoder_step{steps:04d}.pt'
    result = dict(updates=steps, elapsed_seconds=elapsed,
                  peak_gib=torch.cuda.max_memory_allocated()/2**30,
                  final_checkpoint=str(final), checkpoint_sha256=shared.digest(final),
                  initial_probe=initial, final_probe=history[-1], validation_images_used=0)
    shared.save_json(destination/'result.json', result)
    return result
