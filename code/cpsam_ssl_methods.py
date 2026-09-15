"""DINO and I-JEPA objectives on the SAME existing CPSAM encoder.

Training-only adapters preserve all encoder parameter names and relative positions.
No official DINO/I-JEPA backbone or weights are instantiated. The inference neck
is retained, frozen during SSL, and trained normally in the downstream model.
"""
import copy
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def sparse_attention(attn, x, indices, size):
    """SAM attention on only visible tokens, at their ORIGINAL 2-D positions."""
    from segment_anything.modeling.image_encoder import get_rel_pos
    b, n, c = x.shape
    q, k, v = attn.qkv(x).reshape(b, n, 3, attn.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
    scores = (q * attn.scale) @ k.transpose(-2, -1)
    if attn.use_rel_pos:
        row, col = indices // size, indices % size
        rh = get_rel_pos(size, size, attn.rel_pos_h)[row]
        rw = get_rel_pos(size, size, attn.rel_pos_w)[col]
        bh = torch.einsum('bhnd,bnrd->bhnr', q, rh)
        bw = torch.einsum('bhnd,bnrd->bhnr', q, rw)
        keys = (b, attn.num_heads, n, n)
        scores = scores + bh.gather(-1, row[:, None, None, :].expand(keys))
        scores = scores + bw.gather(-1, col[:, None, None, :].expand(keys))
    out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, n, c)
    return attn.proj(out)


def token_forward(encoder, images, masks=None, grad_checkpoint=False):
    """Gather BEFORE any attention. Masked image patches have no path to context."""
    x = encoder.patch_embed(images)
    b, h, w, c = x.shape
    assert h == w
    pos = encoder.pos_embed
    if pos.shape[1:3] != (h, w):
        pos = F.interpolate(pos.permute(0, 3, 1, 2), size=(h, w), mode='bicubic', align_corners=False).permute(0, 2, 3, 1)
    x = (x + pos).flatten(1, 2)
    indices = torch.arange(h*w, device=x.device)[None].expand(b, -1) if masks is None else masks
    if masks is not None:
        x = x.gather(1, masks[..., None].expand(-1, -1, c))
    for block in encoder.blocks:
        assert block.window_size == 0, 'Only the existing global-attention CPSAM is supported'
        def apply(z, block=block):
            z = z + sparse_attention(block.attn, block.norm1(z), indices, h)
            return z + block.mlp(block.norm2(z))
        x = checkpoint(apply, x, use_reentrant=False) if grad_checkpoint else apply(x)
    return x


def configure_encoder(encoder):
    # No neck output is used by either SSL loss. Keep it unchanged for inference.
    for p in encoder.parameters():
        p.requires_grad_(True)
    for p in encoder.neck.parameters():
        p.requires_grad_(False)


def ema_update(student, teacher, momentum):
    with torch.no_grad():
        for (name, p), (other, q) in zip(student.named_parameters(), teacher.named_parameters()):
            assert name == other
            if p.requires_grad:
                q.mul_(momentum).add_(p, alpha=1-momentum)
            else:
                q.copy_(p)
        for p, q in zip(student.buffers(), teacher.buffers()):
            q.copy_(p)


def make_crops(views, indices, rng, dino):
    arrays = [[], [], [], []] if dino else [[]]
    for i in indices:
        # One native field, selected without targets; local views retain pixel scale.
        origin = views.selected[int(i), rng.integers(8)] + rng.integers(-32, 33, size=2)
        origin = np.clip(origin, [0, 0], [views.height-256, views.width-256])
        image = views.tile(int(i), origin, 0)
        for j, size in enumerate((256, 256, 128, 128) if dino else (256,)):
            r, c = (0, 0) if size == 256 else tuple(rng.integers(0, 129, size=2))
            a = np.rot90(image[:, r:r+size, c:c+size], rng.integers(4), axes=(-2, -1))
            if rng.random() < .5:
                a = a[:, :, ::-1]
            a = np.clip(a*rng.uniform(.95, 1.05)+rng.uniform(-.02, .02), 0, 1)
            arrays[j].append(np.ascontiguousarray(a))
    return [torch.from_numpy(np.stack(a)).cuda() for a in arrays]


@torch.no_grad()
def diagnose(encoder, probe):
    vals = []
    for batch in probe.split(4):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            vals.append(token_forward(encoder, batch).float().mean(1))
    x = torch.cat(vals)
    x = x-x.mean(0)
    s = torch.linalg.svdvals(x)
    p = s.square()/s.square().sum().clamp_min(1e-12)
    return dict(std=float(x.std(0, unbiased=False).mean()), rank=float(torch.exp(-(p*p.clamp_min(1e-12).log()).sum())))


def healthy(stats, initial):
    if not all(math.isfinite(v) for v in stats.values()):
        raise FloatingPointError('Nonfinite SSL diagnostic')
    if stats['std'] < max(1e-4, .2*initial['std']) or stats['rank'] < .25*initial['rank']:
        raise FloatingPointError('Severe feature contraction')


def train_ssl(encoder, views, destination, config, save, digest, upstream):
    import sys
    sys.path.insert(0, str(upstream/'dino'))
    from vision_transformer import DINOHead
    from main_dino import DINOLoss
    sys.path.insert(0, str(upstream/'ijepa'))
    from src.models.vision_transformer import vit_predictor
    from src.masks.multiblock import MaskCollator
    from src.masks.utils import apply_masks

    class LocalDINOLoss(DINOLoss):
        @torch.no_grad()
        def update_center(self, outputs):
            self.center.mul_(self.center_momentum).add_(outputs.mean(0, keepdim=True), alpha=1-self.center_momentum)

    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    save(out/'config.json', config)
    steps, batch_size, lr = config['steps'], config['batch'], config['lr']
    random.seed(914); np.random.seed(914); torch.manual_seed(914); torch.cuda.manual_seed_all(914)
    configure_encoder(encoder)
    teacher = copy.deepcopy(encoder).cuda().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    encoder.cuda().train()
    is_dino = config['method'] == 'dino'
    dim = encoder.pos_embed.shape[-1]
    if is_dino:
        auxiliary = DINOHead(dim, 8192, hidden_dim=2048, bottleneck_dim=256).cuda()
        # Legacy weight_norm exposes a non-leaf weight: do not deepcopy this head.
        target_head = DINOHead(dim, 8192, hidden_dim=2048, bottleneck_dim=256).cuda().eval()
        target_head.load_state_dict(auxiliary.state_dict(), strict=True)
        for p in target_head.parameters():
            p.requires_grad_(False)
        criterion = LocalDINOLoss(8192, 4, .04, .04, 0, 1).cuda()
        collator = None
    else:
        auxiliary = vit_predictor(num_patches=1024, embed_dim=dim, predictor_embed_dim=384, depth=6, num_heads=12).cuda()
        target_head = criterion = None
        collator = MaskCollator(input_size=256, patch_size=8, enc_mask_scale=(.85, 1.),
            pred_mask_scale=(.15, .2), aspect_ratio=(.75, 1.5), nenc=1, npred=4,
            min_keep=10, allow_overlap=False)
    params = [p for p in encoder.parameters() if p.requires_grad]
    head_params = [p for p in auxiliary.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([dict(params=params, lr=lr), dict(params=head_params, lr=lr*config['head_multiplier'])], weight_decay=.04)
    rng = np.random.default_rng(914)
    order_rng = np.random.default_rng(913)
    order, cursor, seen = order_rng.permutation(9000), 0, set()
    probe = torch.from_numpy(np.stack([views.tile(i, views.selected[i, 0], 0) for i in range(32)])).cuda()
    initial = diagnose(teacher, probe)
    history = [dict(step=0, **initial)]
    started = time.monotonic()
    sample_hash = __import__('hashlib').sha256()
    for step in range(1, steps+1):
        if cursor+batch_size > 9000:
            tail = order[cursor:]; order = order_rng.permutation(9000)
            indices = np.concatenate([tail, order[:batch_size-len(tail)]]); cursor = batch_size-len(tail)
        else:
            indices = order[cursor:cursor+batch_size]; cursor += batch_size
        sample_hash.update(indices.astype(np.int64).tobytes()); seen.update(map(int, indices))
        xs = make_crops(views, indices, rng, is_dino)
        progress = (step-1)/max(1, steps-1)
        scheduled = lr*min(step/200, 1)*(.05+.95*.5*(1+math.cos(math.pi*progress)))
        optimizer.param_groups[0]['lr'] = scheduled
        optimizer.param_groups[1]['lr'] = scheduled*config['head_multiplier']
        optimizer.zero_grad(set_to_none=True)
        if is_dino:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                targets = torch.cat([target_head(token_forward(teacher, x).mean(1)) for x in xs[:2]]).float()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                students = torch.cat([auxiliary(token_forward(encoder, x, grad_checkpoint=True).mean(1)) for x in xs]).float()
            loss = criterion(students, targets, 0)
        else:
            # The official collator masks have equal token counts across a batch.
            _, enc_masks, pred_masks = collator([(torch.zeros(1), 0) for _ in indices])
            enc_masks, pred_masks = [m.cuda() for m in enc_masks], [m.cuda() for m in pred_masks]
            for a in enc_masks:
                for b in pred_masks:
                    assert not (a[:, :, None] == b[:, None, :]).any(), 'Context/target overlap'
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                targets = F.layer_norm(token_forward(teacher, xs[0]).float(), (dim,))
                targets = apply_masks(targets, pred_masks)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                context = token_forward(encoder, xs[0], enc_masks[0], grad_checkpoint=True)
                prediction = auxiliary(context, enc_masks, pred_masks)
                loss = F.smooth_l1_loss(prediction.float(), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite SSL loss')
        loss.backward()
        assert any(p.grad is not None for p in params)
        assert all(p.grad is None for p in teacher.parameters())
        if is_dino and step <= 200:
            for p in auxiliary.last_layer.parameters():
                p.grad = None
        norm = torch.nn.utils.clip_grad_norm_(params+head_params, 1., error_if_nonfinite=True)
        optimizer.step()
        momentum = 1-(1-.996)*(.5*(1+math.cos(math.pi*progress)))
        ema_update(encoder, teacher, momentum)
        if is_dino:
            ema_update(auxiliary, target_head, momentum)
        if step % 25 == 0 or step == steps:
            torch.cuda.synchronize()
            elapsed = time.monotonic()-started
            row = dict(stage='ssl', step=step, steps=steps, loss=float(loss.detach()), lr=scheduled,
                grad_norm=float(norm), elapsed_seconds=elapsed, seconds_per_step=elapsed/step,
                unique_images_seen=len(seen), at=time.strftime('%FT%T%z'))
            save(out/'progress.json', row); print(row, flush=True)
        if step % 500 == 0 or step == steps:
            stats = diagnose(teacher, probe); healthy(stats, initial)
            history.append(dict(step=step, **stats)); save(out/'diagnostics.json', history)
        if step % 2000 == 0 or step == steps:
            final = out/f'encoder_{step:04d}.pt'
            torch.save(dict(teacher_encoder=teacher.state_dict(), step=step, config=config), final)
    if not config['smoke']:
        assert len(seen) == 9000
    ck = torch.load(final, map_location='cpu', weights_only=True)
    teacher.load_state_dict(ck['teacher_encoder'], strict=True)
    result = dict(steps=steps, unique_images_seen=len(seen), sample_order_sha256=sample_hash.hexdigest(),
        strict_reload=True, checkpoint=str(final), checkpoint_sha256=digest(final),
        elapsed_seconds=time.monotonic()-started, initial=initial, final=history[-1],
        inference_backbones=1, labels_used=False, all9000=True)
    save(out/'result.json', result)
    return result
