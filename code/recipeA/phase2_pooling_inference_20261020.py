"""Explicit pooling-aware inference for one audited terminal CLEAR-EC checkpoint.

Supported input: a grayscale uint8 NumPy image of exactly 972 x 1296 pixels.
This module reproduces the training transform and single-view prediction. It
does not implement an official challenge submission interface, image resizing,
ensembles, TTA, calibration, FP16 inference or T4 certification.

Typical use::

    model, mu, sigma, metadata = load_checkpoint(
        checkpoint, manifest, device='cuda:0',
        expected_checkpoint_sha256=audited_checkpoint_sha256)
    predictions = predict_images(model, images, mu, sigma,
                                 device='cuda:0', precision='bf16', batch_size=4)

For CPU inference pass device='cpu' and precision='fp32'. Selection of the
checkpoint and the provenance of its expected SHA256 belong to the caller.
"""
from collections.abc import Mapping
import hashlib
import itertools
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = 'cpdino_vitb'
NATIVE_SHAPE = (972, 1296)
PADDED_SHAPE = (992, 1312)
TARGET_NAMES = ('CD', 'CV', 'HEX')
EXPECTED_RUNTIME_SHA256 = {
    'cellpose.vit': '487f8efc4c8ee4498907b8ab8b3bf211645d8b7657fca7afe73ea3d1408dbf43',
    'dinov3.hub.backbones': '6c9a18aca0b5fcb1fce439213a94a528e072376a6023718e03dbd9fcfaa10deb',
}
SUPPORTED_RECIPE = {
    
    'view': {
        'whole native972x1296; replicate-pad bottom20/right16 to992x1312',
        'whole native972x1296; replicate-pad bottom20/right16 to992x1312 for ALL architectures',
    },
    'normalization': {
        'CLAHE+percentile01; Cellpose01',
        'common CLAHE+percentile01; fixed ImageNet normalization for DINOv3/ConvNeXt, Cellpose01 for CPDINO',
    },
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_runtime_sources():
    """Check the installed constructor implementations without creating a model.

    The pinned CPDINO constructor calls dinov3_vitb16(pretrained=False), then
    sets patch stride8/padding4. The pinned DINOv3 builder only initializes
    parameters when pretrained=False; its download branch is not entered.
    """
    import cellpose.vit
    import dinov3.hub.backbones
    modules = {'cellpose.vit': cellpose.vit,
               'dinov3.hub.backbones': dinov3.hub.backbones}
    evidence = {}
    for name, module in modules.items():
        path = Path(module.__file__).resolve()
        actual = sha256_file(path)
        if actual != EXPECTED_RUNTIME_SHA256[name]:
            raise ValueError(f'Unsupported runtime source checksum: {name} ({path})')
        evidence[name] = {'path': str(path), 'sha256': actual}
    return evidence


POOLING_MODES = {'mean': ('mean',), 'std': ('std',), 'min': ('min',), 'max': ('max',),
                 'mean_std': ('mean','std'), 'mean_std_min_max': ('mean','std','min','max')}

class SpatialPatchEncoder(nn.Module):
    def __init__(self, encoder, mode):
        super().__init__()
        if mode not in POOLING_MODES: raise ValueError('Unsupported pooling')
        self.original, self.mode = encoder, mode

    def forward(self, x):
        tokens = self.original.forward_features(x)['x_norm_patchtokens']
        values=[]
        for stat in POOLING_MODES[self.mode]:
            if stat=='mean': value=tokens.mean(1)
            elif stat=='std': value=(tokens.float().var(1,unbiased=False)+1e-12).sqrt().to(tokens.dtype)
            elif stat=='min': value=tokens.amin(1)
            else: value=tokens.amax(1)
            values.append(value)
        return values[0] if len(values)==1 else torch.cat(values,dim=-1)


def build_model(mode):
    if mode not in POOLING_MODES: raise ValueError("Unsupported pooling")
    """Construct the exact CPU architecture without reading pretrained weights.

    Every model parameter/buffer is subsequently supplied by strict loading of
    the selected regression checkpoint. No checkpoint, URL, hub or SSL artifact
    is loaded by this function.
    """
    verify_runtime_sources()
    from cellpose.vit import CPDINO
    foundation = CPDINO(model_name='vitb', ps=8, bsize=384)
    encoder = foundation.encoder
    patch = encoder.patch_embed.proj
    if (tuple(patch.kernel_size) != (16, 16) or tuple(patch.stride) != (8, 8)
            or tuple(patch.padding) != (4, 4)):
        raise ValueError('Runtime CPDINO patch embedding differs from training')
    dim = 768 * len(POOLING_MODES[mode])
    model = nn.Sequential(SpatialPatchEncoder(encoder, mode), nn.LayerNorm(dim),
                          nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 3))
    if any(parameter.device.type != 'cpu' for parameter in model.parameters()):
        raise ValueError('The inference builder must construct on CPU')
    return model


def preprocess_native(image_uint8):
    """Return CPU float32 [1,972,1296] with the exact b.Frames transform."""
    if not isinstance(image_uint8, np.ndarray):
        raise TypeError('Expected a grayscale uint8 NumPy array')
    if image_uint8.dtype != np.uint8 or image_uint8.shape != NATIVE_SHAPE:
        raise ValueError('Only grayscale uint8 images with shape (972,1296) are supported')
    import cv2
    cv2.setNumThreads(1)
    image = cv2.createCLAHE(2.0, (1296 // 32, 972 // 32)).apply(np.asarray(image_uint8))
    lo, hi = np.percentile(image, (1, 99))
    image = np.clip((image.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
    if image.dtype != np.float32 or not np.isfinite(image).all():
        raise ValueError('Preprocessing did not produce finite float32 pixels')
    return torch.from_numpy(np.ascontiguousarray(image[None]))


def _device(value):
    device = torch.device(value)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only CPU and CUDA inference are supported')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable')
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError('Requested CUDA device does not exist')
        device = torch.device('cuda', index)
    else:
        device = torch.device('cpu')
    return device


def prepare_batch(tensor, device):
    """Move normalized frames and return float32 [N,3,992,1312]."""
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
        raise TypeError('Expected a float32 tensor from preprocess_native')
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (1, *NATIVE_SHAPE) or len(tensor) == 0:
        raise ValueError('Expected a nonempty batch with shape [N,1,972,1296]')
    if not torch.isfinite(tensor).all().item() or tensor.min().item() < 0 or tensor.max().item() > 1:
        raise ValueError('Normalized pixels must be finite and within [0,1]')
    target = _device(device)
    x = tensor.to(device=target, non_blocking=True).expand(-1, 3, -1, -1)
    return F.pad(x, (0, 16, 0, 20), mode='replicate').contiguous()


def _statistics(value, name, *, positive=False):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid {name}') from exc
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain three finite values in CD/CV/HEX order')
    if positive and not (array > 0).all():
        raise ValueError(f'{name} must be strictly positive')
    return array


def _manifest_architecture(manifest):
    architecture = manifest.get('architecture')
    # Archived single-architecture regression manifests used this plural field.
    if architecture is None and manifest.get('architectures') == [ARCHITECTURE]:
        architecture = ARCHITECTURE
    if architecture != ARCHITECTURE:
        raise ValueError('Manifest is not a single CPDINO regression model')
    if 'architectures' in manifest and manifest['architectures'] != [ARCHITECTURE]:
        raise ValueError('Manifest has conflicting architecture declarations')
    recipe = manifest.get('recipe')
    if not isinstance(recipe, Mapping):
        raise ValueError('Manifest has no supported training recipe')
    mode=recipe.get('pooling')
    if mode not in POOLING_MODES:
        raise ValueError('Explicit supported pooling declaration required')
    if (recipe.get('head') != mode+' pool -> LayerNorm -> Linear128 -> GELU -> Linear3'
            or recipe.get('pooling_statistics') != list(POOLING_MODES[mode])
            or recipe.get('pooled_dimension') != 768*len(POOLING_MODES[mode])
            or recipe.get('std_population') is not True
            or recipe.get('std_epsilon') != 1e-12):
        raise ValueError('Pooling/head contract differs from the supported training schema')
    for key, allowed in SUPPORTED_RECIPE.items():
        if recipe.get(key) not in allowed:
            raise ValueError(f'Unsupported model or preprocessing recipe: {key}')


def load_checkpoint(checkpoint_path, manifest_path, device='cpu', *, expected_checkpoint_sha256):
    """Verify and strict-load one audited checkpoint, returning model,mu,sigma,metadata.

    The caller must supply the expected checkpoint SHA256 from reviewed audit
    evidence. This function neither searches sidecars nor selects a checkpoint.
    Only terminal epoch snapshots and last.pt are supported; partial deadline
    snapshots lack the required completed-epoch architecture metadata.
    """
    if not isinstance(expected_checkpoint_sha256, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', expected_checkpoint_sha256):
        raise ValueError('A 64-character audited expected_checkpoint_sha256 is required')
    checkpoint_path, manifest_path = Path(checkpoint_path), Path(manifest_path)
    actual_sha = sha256_file(checkpoint_path)
    if actual_sha != expected_checkpoint_sha256.lower():
        raise ValueError('Checkpoint SHA256 differs from the caller-provided audit evidence')
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError('Manifest must be a JSON object')
    _manifest_architecture(manifest)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Checkpoint must be a mapping')
    if checkpoint.get('architecture') != ARCHITECTURE:
        raise ValueError('Checkpoint architecture differs from CPDINO regression')
    if checkpoint.get('manifest_sha') != manifest_sha:
        raise ValueError('Checkpoint is bound to a different manifest')
    epoch = checkpoint.get('epoch')
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError('Only a completed epoch checkpoint is supported')
    if epoch != manifest.get('recipe', {}).get('epochs', 0):
        raise ValueError('Only the declared terminal epoch is supported')
    mean = _statistics(checkpoint.get('log_mean'), 'checkpoint log_mean')
    std = _statistics(checkpoint.get('log_std'), 'checkpoint log_std', positive=True)
    expected_mean = _statistics(manifest.get('log_mean'), 'manifest log_mean')
    expected_std = _statistics(manifest.get('log_std'), 'manifest log_std', positive=True)
    if not np.array_equal(mean, expected_mean) or not np.array_equal(std, expected_std):
        raise ValueError('Checkpoint and manifest log statistics differ')
    state = checkpoint.get('model')
    if not isinstance(state, Mapping) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Checkpoint must contain the complete tensor model state')
    model = build_model(manifest["recipe"]["pooling"])
    expected_state = model.state_dict()
    if set(state) != set(expected_state):
        raise ValueError('Checkpoint model state has missing or unexpected keys')
    for name, tensor in state.items():
        if tensor.shape != expected_state[name].shape or tensor.dtype != expected_state[name].dtype:
            raise ValueError(f'Checkpoint tensor shape/dtype differs: {name}')
    model.load_state_dict(state, strict=True)
    target = _device(device)
    model = model.to(target).eval()
    mu = torch.tensor(mean, device=target, dtype=torch.float32)
    sigma = torch.tensor(std, device=target, dtype=torch.float32)
    if not torch.isfinite(mu).all().item() or not torch.isfinite(sigma).all().item() or not (sigma > 0).all().item():
        raise ValueError('Log statistics are invalid after conversion to float32')
    metadata = dict(pooling=manifest["recipe"]["pooling"], pooled_dimension=manifest["recipe"]["pooled_dimension"], architecture=ARCHITECTURE, checkpoint_path=str(checkpoint_path),
                    checkpoint_sha256=actual_sha, manifest_path=str(manifest_path),
                    manifest_sha256=manifest_sha, epoch=epoch, fold=manifest.get('fold'),
                    seed=manifest.get('seed', manifest.get('recipe', {}).get('seed')),
                    device=str(target), target_names=list(TARGET_NAMES),
                    native_shape=list(NATIVE_SHAPE), padded_shape=list(PADDED_SHAPE),
                    single_model=True, runtime_sources=verify_runtime_sources(),
                    preprocessing='CLAHE2.0 grid40x30; percentile1/99; replicate-pad bottom20/right16',
                    output_transform='exp(standardized_log_prediction.float()*log_std+log_mean)',
                    scope='972x1296 grayscale uint8 research readback; not an official submission certification')
    return model, mu, sigma, metadata


def predict_images(model, images, mu, sigma, *, device, precision='bf16', batch_size=4):
    """Predict CD/CV/HEX for one view per image; returns NumPy float32 [N,3].

    CUDA BF16 is the research default. Explicit precision='fp32' supports CPU
    or CUDA. BF16 is rejected on CPU; FP16 is not implemented or certified.
    """
    if precision not in ('bf16', 'fp32'):
        raise ValueError('precision must be bf16 or fp32; FP16 is not validated')
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    target = _device(device)
    if precision == 'bf16':
        if target.type != 'cuda':
            raise ValueError('CPU inference requires explicit precision=fp32')
        with torch.cuda.device(target):
            if torch.cuda.get_device_capability(target)[0] < 8 or not torch.cuda.is_bf16_supported():
                raise ValueError('This CUDA device does not support the BF16 research path')
    if not isinstance(model, nn.Sequential) or len(model) != 5:
        raise ValueError('Expected the single regression model returned by load_checkpoint')
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        if tensor.device != target:
            raise ValueError('Model device differs from the requested inference device')
    for name, tensor in (('mu', mu), ('sigma', sigma)):
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != (3,)
                or tensor.dtype != torch.float32 or tensor.device != target
                or not torch.isfinite(tensor).all().item()):
            raise ValueError(f'{name} must be a finite float32 vector of length3 on the model device')
    if not (sigma > 0).all().item():
        raise ValueError('sigma must be strictly positive')
    if isinstance(images, np.ndarray) and images.ndim == 2:
        images = (images,)
    iterator = iter(images)
    predictions = []
    model.eval()
    with torch.inference_mode():
        while True:
            group = list(itertools.islice(iterator, batch_size))
            if not group:
                break
            batch = torch.stack([preprocess_native(image) for image in group])
            batch = prepare_batch(batch, target)
            context = (torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16'
                       else torch.autocast(target.type, enabled=False))
            with context:
                standardized = model(batch).float()
            if standardized.shape != (len(group), 3) or not torch.isfinite(standardized).all().item():
                raise ValueError('Model produced invalid standardized log predictions')
            physical = torch.exp(standardized * sigma + mu)
            if not torch.isfinite(physical).all().item() or not (physical > 0).all().item():
                raise ValueError('Model produced invalid physical predictions')
            predictions.append(physical.cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty((0, 3), dtype=np.float32)
