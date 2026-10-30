#!/usr/bin/env python3
"""Disposable, train-only resource probes. No fitted checkpoint or score is produced.

Workers use one GPU and fresh processes per configuration. Results are hardware
feasibility evidence, not a ranking of model accuracy or a T4 compatibility test.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(os.environ.get('CLEAR_EC_PREFLIGHT_ROOT', 'experiments/phase2_foundation_20261002'))
DATA = Path('datasets/clear_ec_fullframe_20260904')
SHAPES = {'central512': (512, 512), 'full_resized576x768': (576, 768),
          'full_native972x1296': (972, 1296)}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def prepare():
    import numpy as np
    target = ROOT / 'preflight_manifest.json'
    if target.exists():
        raise FileExistsError(target)
    previous = json.loads(Path('experiments/phase2_metric35_ssl_20260928/manifest.json').read_text())
    splits = {}
    with np.load('datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True) as z:
        ids, labels = z['ids'].astype(str), z['Y'].copy()
    with np.load(DATA / 'fullframe_meta.npz', allow_pickle=True) as z:
        np.testing.assert_array_equal(ids, z['ids'].astype(str))
    for fold in ('2', '3'):
        path = Path(previous['folds'][fold]['split_path'])
        with np.load(path) as z:
            train, val = z['train'].copy(), z['validation'].copy()
        assert not np.intersect1d(train, val).size
        splits[fold] = dict(path=str(path), sha256=digest(path), train_n=len(train), val_n=len(val),
                            train_zero_counts=(labels[train] == 0).sum(0).tolist(),
                            val_zero_counts=(labels[val] == 0).sum(0).tolist())
        if fold == '2':
            # No validation image is used for speed tests or annotation pilot selection.
            probe_indices = train[:8].tolist()
    manifest = dict(created_utc=time.strftime('%FT%TZ', time.gmtime()),
                    source_sha256=digest(__file__), splits=splits,
                    probe_indices=probe_indices, probe_ids=ids[probe_indices].tolist(),
                    total_images=len(ids), full_shape=[972, 1296],
                    full_manifest_sha256=digest(DATA / 'fullframe_manifest.json'),
                    raw_meta_sha256=digest('datasets/clear_ec_official/cache_native_meta_512.npz'),
                    status='RESOURCE_PREFLIGHT_ONLY', official_release_untouched=True,
                    gpu_parallelism=2, maximum_gpu_parallelism=4,
                    metric_policy='Not scored. Zero-label rule must be verified before accuracy trials.',
                    validation_limits='Reused development folds; patient/slide grouping needs separate audit.',
                    runtime={k: importlib.metadata.version(k) for k in ('torch', 'timm', 'cellpose')})
    save(target, manifest)
    print(json.dumps(manifest), flush=True)


def build(architecture, checkpointing, native_convnext_norm=False):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint
    if architecture == 'cpsam':
        from cellpose.vit import CPSAM
        path = Path('cpsam_models/cpsam_v2')
        foundation = CPSAM()
        enc = foundation.encoder
        state = torch.load(path, map_location='cpu', weights_only=True)
        # The external artifact predates two non-trainable diameter metadata
        # entries. Strictly load ALL encoder tensors; segmentation readout and
        # diameter metadata are not part of this regression model.
        enc.load_state_dict({k[len('encoder.'):]: v for k, v in state.items()
                             if k.startswith('encoder.')}, strict=True)
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.original = enc
            def forward(self, x):
                x = self.original.patch_embed(x)
                pos = self.original.pos_embed
                if pos.shape[1:3] != x.shape[1:3]:
                    pos = F.interpolate(pos.permute(0, 3, 1, 2), size=x.shape[1:3],
                                        mode='bicubic', align_corners=False).permute(0, 2, 3, 1)
                x = x + pos
                for block in self.original.blocks:
                    x = checkpoint(block, x, use_reentrant=False) if checkpointing and self.training else block(x)
                return self.original.neck(x.permute(0, 3, 1, 2)).mean((-2, -1))
        encoder, dim, multiple = Encoder(), 256, 8
        details = dict(weight_path=str(path), weight_sha256=digest(path),
                       patch_kernel=list(enc.patch_embed.proj.kernel_size),
                       patch_stride=list(enc.patch_embed.proj.stride),
                       position_shape=list(enc.pos_embed.shape),
                       attention='original CPSAM global attention; no kernel replacement')
    elif architecture == 'cpdino_vitb':
        from cellpose.vit import CPDINO
        path = Path('experiments/phase2_foundation_20261002/weights/cellpose/cpdino-vitb')
        foundation = CPDINO(model_name='vitb', ps=8, bsize=384)
        enc = foundation.encoder
        state = torch.load(path, map_location='cpu', weights_only=True)
        enc.load_state_dict({k[len('encoder.'):]: v for k, v in state.items()
                             if k.startswith('encoder.')}, strict=True)
        class CheckpointBlock(nn.Module):
            def __init__(self, block):
                super().__init__()
                self.original = block
            def forward(self, *inputs, **kwargs):
                if self.training:
                    return checkpoint(self.original, *inputs, use_reentrant=False, **kwargs)
                return self.original(*inputs, **kwargs)
        if checkpointing:
            enc.blocks = nn.ModuleList([CheckpointBlock(b) for b in enc.blocks])
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.original = enc
            def forward(self, x):
                return self.original.forward_features(x)['x_norm_patchtokens'].mean(1)
        encoder, dim, multiple = Encoder(), 768, 16
        details = dict(weight_path=str(path), weight_sha256=digest(path),
                       patch_kernel=list(enc.patch_embed.proj.kernel_size),
                       patch_stride=list(enc.patch_embed.proj.stride),
                       patch_padding=list(enc.patch_embed.proj.padding),
                       pretrained_source='CellposeDINO ViT-B, revision 7c61431b5fbb078f3296754bd15d9f51b320f837')
    elif architecture == 'dinov3_vitb':
        from transformers import AutoModel
        path = Path('experiments/phase2_foundation_20261002/weights/dinov3-vitb16')
        artifact = json.loads((path/'artifact_manifest.json').read_text())
        weight = path/'model.safetensors'
        if digest(weight) != artifact['files']['model.safetensors']['sha256']:
            raise ValueError('DINOv3 artifact checksum changed')
        enc = AutoModel.from_pretrained(path, local_files_only=True, dtype=torch.float32)
        if checkpointing:
            enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        prefix = 1 + enc.config.num_register_tokens
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.original = enc
            def forward(self, x):
                tokens = self.original(pixel_values=x).last_hidden_state[:, prefix:]
                assert tokens.shape[1] == (x.shape[-2]//16)*(x.shape[-1]//16)
                return tokens.mean(1)
        encoder, dim, multiple = Encoder(), enc.config.hidden_size, 16
        details = dict(weight_path=str(weight), weight_sha256=digest(weight),
                       pretrained_source=artifact['repo'], revision=artifact['revision'],
                       patch_kernel=[16,16], patch_stride=[16,16])
    elif architecture == 'convnext_base':
        import timm
        from safetensors.torch import load_file
        path = Path('experiments/phase2_backbone_screen_20261001/hf_cache/hub/models--timm--convnext_base.fb_in22k_ft_in1k/snapshots/67758fcbcf9f007ec7630bf8636b74a8f24e52a0/model.safetensors')
        # Load the unchanged external foundation, never a CLEAR-EC trained checkpoint.
        base = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=False)
        base.load_state_dict(load_file(str(path)), strict=True)
        base.reset_classifier(0)
        base.set_grad_checkpointing(checkpointing)
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.original = base
            def forward(self, x):
                features = self.original.forward_features(x)
                if native_convnext_norm:
                    return self.original.forward_head(features, pre_logits=True)
                return features.mean((-2, -1))
        encoder, dim, multiple = Encoder(), 1024, 32
        details = dict(weight_path=str(path), weight_sha256=digest(path),
                       pretraining='ImageNet22k supervised -> ImageNet1k; historical reference version',
                       pretrained_head_norm_preserved=native_convnext_norm)
    else:
        raise ValueError(architecture)
    model = nn.Sequential(encoder, nn.LayerNorm(dim), nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 3))
    details['parameter_count'] = sum(p.numel() for p in model.parameters())
    details['trainable_parameter_count'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, multiple, details


def one(args):
    import cv2
    import numpy as np
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(42)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Need exactly one visible GPU')
    manifest = json.loads((ROOT / 'preflight_manifest.json').read_text())
    if not 1 <= args.batch <= len(manifest['probe_indices']):
        raise ValueError('Requested batch exceeds the available distinct probe inputs')
    if digest(__file__) != manifest['source_sha256']:
        raise ValueError('Preflight source changed')
    model, multiple, details = build(args.architecture, args.checkpointing)
    raw = np.memmap(DATA / 'img_full_raw.dat', mode='r', dtype=np.uint8,
                    shape=(manifest['total_images'], 972, 1296))
    images = []
    t0 = time.perf_counter()
    h, w = SHAPES[args.arm]
    for index in manifest['probe_indices'][:args.batch]:
        image = np.asarray(raw[index])
        if args.arm == 'central512':
            image = image[(972-512)//2:(972+512)//2, (1296-512)//2:(1296+512)//2]
        elif args.arm == 'full_resized576x768':
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        image = cv2.createCLAHE(2.0, (max(1, w//32), max(1, h//32))).apply(np.ascontiguousarray(image))
        lo, hi = np.percentile(image, (1, 99))
        image = np.clip((image.astype(np.float32)-lo)/(hi-lo+1e-6), 0, 1)
        images.append(np.stack([image]*3))
    preprocess_seconds = time.perf_counter()-t0
    x = torch.from_numpy(np.stack(images)).cuda()
    assert x.shape[0] == args.batch
    x = F.pad(x, (0, (-w)%multiple, 0, (-h)%multiple), mode='replicate')
    if args.architecture in ('convnext_base', 'dinov3_vitb'):
        x = (x - x.new_tensor([.485,.456,.406])[None,:,None,None]) / x.new_tensor([.229,.224,.225])[None,:,None,None]
    model = model.cuda()
    if args.channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
    dtype = torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    details.update(input_shape=list(x.shape), original_view_shape=[h,w],
                   resized=args.arm == 'full_resized576x768', cropped=args.arm == 'central512',
                   batch=args.batch, checkpointing=args.checkpointing, channels_last=args.channels_last,
                   dtype='bfloat16', gpu=torch.cuda.get_device_name(),
                   preprocess_batch_seconds=preprocess_seconds,
                   disclaimer='Repeated train-only inputs, synthetic objective, no accuracy claim; GPU-only timing excludes data loading; not T4.')
    if args.architecture in ('cpsam', 'cpdino_vitb'):
        with torch.no_grad():
            actual = model[0].original.patch_embed.proj(x)
        details['actual_patch_grid'] = list(actual.shape[-2:])
        del actual
    elif args.architecture == 'dinov3_vitb':
        details['actual_patch_grid'] = [x.shape[-2]//16, x.shape[-1]//16]
    # Measure inference before allocating gradients/optimizer states.
    model.eval()
    with torch.inference_mode():
        for _ in range(2):
            with torch.autocast('cuda', dtype=dtype):
                model(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(4):
            with torch.autocast('cuda', dtype=dtype):
                model(x)
        torch.cuda.synchronize()
        details['inference_ms_per_image'] = (time.perf_counter()-t0)*1000/(4*args.batch)
        details['inference_peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
    model.train()
    durations = []
    for step in range(6):
        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if step == 2:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.autocast('cuda', dtype=dtype):
            pred = model(x)
            loss = (pred.float()-0.1).square().mean()
        loss.backward()
        optim.step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter()-t0)
        # Run diagnostics OUTSIDE the timed region. One host sync, not one per
        # parameter, avoids distorting the measured optimizer-step throughput.
        finite = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()
                              if p.grad is not None] + [torch.isfinite(loss)]).all()
        if not finite.item():
            raise ValueError('Non-finite gradient/loss')
    details.update(train_seconds_per_step=float(np.median(durations[2:])),
                   train_images_per_second=args.batch/float(np.median(durations[2:])),
                   train_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   train_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                   note='Finite-gradient checks are outside timed regions; accuracy trials must benchmark their actual objective/loader.',
                   status='success')
    save(args.output, details)
    print(json.dumps(details), flush=True)


def worker(args):
    out = ROOT / 'probes' / args.architecture
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        raise FileExistsError(out/'complete.json')
    outcomes = []
    configs = []
    for arm in SHAPES:
        configs.append((arm, 1, True, False))
        if args.architecture == 'convnext_base':
            configs.append((arm, 4, False, True))
        elif arm == 'central512':
            configs.append((arm, 2, False, False))
    for arm, batch, ckpt, layout in configs:
        name = f'{arm}_b{batch}_ckpt{int(ckpt)}_cl{int(layout)}'
        path = out / f'{name}.json'
        cmd = [sys.executable, __file__, 'one', '--architecture', args.architecture,
               '--arm', arm, '--batch', str(batch), '--output', str(path)]
        if ckpt:
            cmd.append('--checkpointing')
        if layout:
            cmd.append('--channels-last')
        print(f'START {name}', flush=True)
        with (out/f'{name}.log').open('w') as log:
            try:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
                status = 'success' if result.returncode == 0 else 'failed'
                entry = dict(name=name, status=status, returncode=result.returncode)
            except subprocess.TimeoutExpired:
                entry = dict(name=name, status='timeout')
        outcomes.append(entry)
        save(out/'progress.json', dict(outcomes=outcomes))
        print(json.dumps(entry), flush=True)
    save(out/'complete.json', dict(outcomes=outcomes, all_success=all(x['status']=='success' for x in outcomes),
                                  status='preflight_finished', no_trained_checkpoint_saved=True))
    if not any(x['status']=='success' for x in outcomes):
        raise RuntimeError('All resource probes failed; see logs')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare','worker','one'])
    p.add_argument('--architecture', choices=['cpsam','convnext_base','cpdino_vitb','dinov3_vitb'])
    p.add_argument('--arm', choices=list(SHAPES))
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--checkpointing', action='store_true')
    p.add_argument('--channels-last', action='store_true')
    p.add_argument('--output')
    args = p.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'worker':
        worker(args)
    else:
        one(args)
