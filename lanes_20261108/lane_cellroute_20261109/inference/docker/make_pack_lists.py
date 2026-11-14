"""Packaging lists (deploy lane, 2026-11-12): docker/fp16_files.txt (repo-relative paths of the 82 fp16 copies, from deploy_fp16_sha256.txt) and
docker/base_shapes.json (names / shapes / dtypes of the base files the hooks replace; read from the safetensors header and the CPDINO state dict)."""
import json, os
import torch
from safetensors import safe_open
ROOT = '/workspace/clear-ec-research/'; D = ROOT + 'experiments/lane_cellroute_20261109/inference/docker/'
rels = [l.split()[1][2:] for l in open(D + 'deploy_fp16_sha256.txt') if l.strip()]
assert len(rels) == 82 and all(os.path.exists(ROOT + r) for r in rels)
open(D + 'fp16_files.txt', 'w').write('\n'.join(rels) + '\n')
TIMM = 'experiments/phase2_backbone_screen_20261001/hf_cache/hub/models--timm--vit_large_patch16_dinov3.lvd1689m/snapshots/30c1109559f65dea34316b0d4842d35c5771fe11/model.safetensors'
CPD = 'experiments/phase2_foundation_20261002/weights/cellpose/cpdino-vitb'
HF = 'experiments/phase2_foundation_20261002/weights/dinov3-vitb16'
st = {}
with safe_open(ROOT + TIMM, framework='pt') as f:
    st[TIMM] = {k: [list(f.get_slice(k).get_shape()), str(f.get_slice(k).get_dtype()).lower().replace('f32', 'float32').replace('bf16', 'bfloat16').replace('f16', 'float16')] for k in f.keys()}
sd = torch.load(ROOT + CPD, map_location='cpu', weights_only=True); assert isinstance(sd, dict) and all(isinstance(v, torch.Tensor) for v in sd.values())
import hashlib
DG = {CPD: hashlib.sha256(open(ROOT + CPD, 'rb').read()).hexdigest()}
spec = dict(digest=DG, torch={CPD: {k: [list(v.shape), str(v.dtype).replace('torch.', '')] for k, v in sd.items()}}, safetensors=st, hf=[HF],
            note='base files replaced at load by pack_hooks (CLEAR_EC_SKIP_BASE); only config.json of the HF directory ships')
json.dump(spec, open(D + 'base_shapes.json', 'w'))
print('fp16 files', len(rels), '| timm tensors', len(st[TIMM]), set(v[1] for v in st[TIMM].values()), '| cpdino tensors', len(spec['torch'][CPD]), set(v[1] for v in spec['torch'][CPD].values()))
