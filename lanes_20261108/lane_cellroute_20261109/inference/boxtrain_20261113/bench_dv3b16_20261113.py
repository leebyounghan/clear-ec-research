"""Synthetic step benchmark of the dinov3_b16 recipe-A training step on the build box GPU (deploy lane, 2026-11-13). No data, random weights.
Same shapes and operations as bb_runner_20261108.py --family dinov3_b16: HF DINOv3ViTModel (config of weights/dinov3-vitb16), input
(micro, 3, 992, 1312) float32 (972 x 1296 frame padded 16 / 20), std pool over the final normed patch tokens -> LayerNorm -> Linear 128 -> GELU ->
Linear 3, bf16 autocast, effective batch 16 as micro-batches with gradient accumulation, clip 1, AdamW. Reports seconds per update (16 frames),
eval seconds per 16 frames (no grad), and peak allocated / reserved GiB. usage: python bench_dv3b16_20261113.py CONFIG_DIR [MICRO ...]"""
import sys, time, json
import torch
from torch import nn
from transformers import AutoConfig, AutoModel
cfg = AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)
R = getattr(cfg, 'num_register_tokens', 0)
res = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, total_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2), runs={})
for micro in [int(m) for m in (sys.argv[2:] or ['8'])]:
    torch.manual_seed(0); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    enc = AutoModel.from_config(cfg).cuda().train()
    head = nn.Sequential(nn.LayerNorm(cfg.hidden_size), nn.Linear(cfg.hidden_size, 128), nn.GELU(), nn.Linear(128, 3)).cuda()
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-5, weight_decay=1e-4)

    def fwd(x):
        h = enc(pixel_values=x).last_hidden_state[:, 1 + R:]
        return head(h.float().std(1))
    x = torch.rand(micro, 3, 992, 1312, device='cuda')

    def update():
        for _ in range(16 // micro):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = fwd(x).pow(2).mean() * micro / 16
            loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
    try:
        for _ in range(3): update()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(10): update()
        torch.cuda.synchronize(); tu = (time.perf_counter() - t) / 10
        enc.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(2): fwd(x)
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(10): fwd(x)
            torch.cuda.synchronize(); te = (time.perf_counter() - t) / 10 * 16 / micro
        res['runs'][micro] = dict(seconds_per_update=round(tu, 3), eval_seconds_per_16=round(te, 3),
                                  peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2), peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2))
    except torch.OutOfMemoryError as e:
        res['runs'][micro] = dict(oom=str(e)[:200])
    del enc, head, opt, params, x
print(json.dumps(res), flush=True)
