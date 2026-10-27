"""T4-bound inference dry run for recipe A on an H200: memory cap, fp16, no flash/cudnn SDPA.

Loads the fold4 recipe A terminal checkpoint, caps the CUDA allocator at 15 GiB,
disables the flash and cuDNN SDPA backends (unavailable on sm75) and measures
peak memory and time per image for one whole-frame view and for the four-flip
TTA, under fp16 autocast. Then compares fp32 / bf16 / fp16 predictions on the
first 200 fold4 validation images: MAPE of each and the fp16-vs-bf16 deviation.
Also reports the same under the math-only SDPA backend when it fits the cap.
No weights are modified; nothing is promoted.
"""
import datetime
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import phase2_foundation_train_20261002 as b
import phase2_foundation_preflight_20261002 as preflight
import phase2_pooling_20261019 as pooling

ROOT = Path('experiments/t4_dry_run_20261027')
CKPT = Path('experiments/phase2_pooling_20261020/arms/std/runs/fold4/cpdino_vitb/epoch_20.pt')
SPLIT = Path('experiments/target_fold4_20260912/preflight/fold4.npz')
CAP_GIB = 15
N_COMPARE = 200
FLIPS = ((False, False), (True, False), (False, True), (True, True))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def flip(x, h, v):
    if h:
        x = torch.flip(x, (-1,))
    if v:
        x = torch.flip(x, (-2,))
    return x


def load():
    ck = torch.load(CKPT, map_location='cpu', weights_only=True)
    net, _, _ = preflight.build('cpdino_vitb', False, native_convnext_norm=True)
    net[0] = pooling.SpatialPoolEncoder(net[0].original, 'std')
    net.load_state_dict(ck['model'], strict=True)
    mu = torch.tensor(ck['log_mean'], dtype=torch.float32)
    sigma = torch.tensor(ck['log_std'], dtype=torch.float32)
    return net.eval().requires_grad_(False), mu, sigma


def predict(net, x, dtype, mu, sigma, backends):
    with torch.no_grad(), sdpa_kernel(backends):
        if dtype is None:
            z = net(b.inputs(x, 'cpdino_vitb')).float()
        else:
            with torch.autocast('cuda', dtype=dtype):
                z = net(b.inputs(x, 'cpdino_vitb')).float()
    return z, torch.exp(z*sigma.cuda()+mu.cuda())


def main():
    assert torch.cuda.device_count() == 1
    ROOT.mkdir(parents=True, exist_ok=False)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(CAP_GIB*2**30/total)
    torch.set_num_threads(4)
    net, mu, sigma = load()
    net = net.cuda()
    with np.load(b.META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    with np.load(SPLIT) as z:
        va = z['validation'][:N_COMPARE].copy()
    frames = b.Frames(va, y, len(ids))
    xs = [frames[i][0][None] for i in range(len(va))]
    report = dict(created_utc=now(), checkpoint=str(CKPT), gpu=torch.cuda.get_device_name(), allocator_cap_gib=CAP_GIB,
                  torch=torch.__version__, cuda=torch.version.cuda, tokens=124*164, sdpa_note='flash and cuDNN disabled to mimic sm75',
                  timing={}, precision={}, errors={})
    t4_backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    for name, dtype, flips, backends in [('fp16_single_view_memeff', torch.float16, [(False, False)], t4_backends),
                                         ('fp16_four_flip_tta_memeff', torch.float16, list(FLIPS), t4_backends),
                                         ('fp16_single_view_math_only', torch.float16, [(False, False)], [SDPBackend.MATH]),
                                         ('fp32_single_view_memeff', None, [(False, False)], t4_backends)]:
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            predict(net, xs[0], dtype, mu, sigma, backends)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for x in xs[:20]:
                zs = [predict(net, flip(x, h, v), dtype, mu, sigma, backends)[0] for h, v in flips]
                torch.stack(zs).mean(0)
            torch.cuda.synchronize()
            report['timing'][name] = dict(seconds_per_image_h200=(time.perf_counter()-t0)/20,
                                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                                          peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, views=len(flips))
        except torch.OutOfMemoryError as e:
            report['errors'][name] = f'OOM under {CAP_GIB} GiB cap: {str(e)[:200]}'
        torch.cuda.empty_cache()
    preds = {}
    for name, dtype in (('fp32', None), ('bf16', torch.bfloat16), ('fp16', torch.float16)):
        out = []
        for x in xs:
            _, p = predict(net, x, dtype, mu, sigma, t4_backends)
            out.append(p.cpu().numpy())
        preds[name] = np.concatenate(out)
        report['precision'][name] = dict(mape=b.metrics(preds[name], y[va]), finite=bool(np.isfinite(preds[name]).all()))
    for a, c in (('fp16', 'bf16'), ('fp16', 'fp32'), ('bf16', 'fp32')):
        rel = np.abs(preds[a]-preds[c])/preds[c]
        report['precision'][f'{a}_vs_{c}'] = dict(max_relative_deviation=rel.max(0).tolist(), mean_relative_deviation=rel.mean(0).tolist(),
                                                 mape_delta_pp=report['precision'][a]['mape']['mean']-report['precision'][c]['mape']['mean'])
    tta = []
    for x in xs:
        zs = [predict(net, flip(x, h, v), torch.bfloat16, mu, sigma, t4_backends)[0] for h, v in FLIPS]
        tta.append(torch.exp(torch.stack(zs).mean(0)*sigma.cuda()+mu.cuda()).cpu().numpy())
    tta = np.concatenate(tta)
    report['tta_bf16_200'] = dict(mape=b.metrics(tta, y[va]), delta_vs_single_view_pp=b.metrics(tta, y[va])['mean']-report['precision']['bf16']['mape']['mean'],
                                  note='200-image preview only; the OOF-wide TTA gate is computed separately')
    report['t4_estimate_note'] = 'T4 fp16 dense throughput is roughly 15-25x below H200; multiply seconds_per_image_h200 accordingly for a rough bound'
    report['finished_utc'] = now()
    (ROOT/'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(timing=report['timing'], errors=report['errors'],
                          fp16_vs_bf16=report['precision'].get('fp16_vs_bf16'), tta=report['tta_bf16_200']['delta_vs_single_view_pp'])), flush=True)


if __name__ == '__main__':
    main()
