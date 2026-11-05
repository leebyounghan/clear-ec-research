"""Read-only CPDINO terminal-checkpoint readback; writes one new aggregate report.

Example (from the PVC workspace root):
  python source/phase2_single_model_readback_20261015.py cpu --manifest \
    experiments/phase2_lejepa5_sweep_20261013/arms/head1e4_clip3/manifest.json \
    --output experiments/phase2_crossfold_20261015/readback_cpu.json

The gpu mode is a separate invocation. Both modes use the first eight validation
rows, fixed before inference. This is implementation verification, not full MAPE
evaluation, model selection, official submission, or T4 certification.
"""
import argparse
from contextlib import ExitStack, contextmanager
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ARCH = 'cpdino_vitb'
SOURCE_SHA = {
    'phase2_foundation_train_20261002.py': '20a849cb607c3c5777102e3266c6c1b5d9ba4533f9bfac9f1f6e9d18232dac2f',
    'phase2_foundation_preflight_20261002.py': '196500b67d01ad9423ea4d7d8c1a1e4b2d886df2328d36bfb1e1c94058ca4b1e',
}
FINGERPRINT_SHA = 'dde29080a4c95a9ea55a23956500f9aee5b469919ba9730648deffa617be81f3'
TOLERANCES = {
    'cpu_small_forward': {'rtol': 1e-6, 'atol': 1e-6},
    'gpu_same_batch_saved': {'rtol': 1e-4, 'atol': 1e-6},
    'gpu_wrapper_reference': {'rtol': 1e-6, 'atol': 1e-6},
    'gpu_batch1_vs_batch4': {'rtol': 2e-3, 'atol': 1e-6},
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def manifest_input_hash(manifest, path):
    matches = {value for name, value in manifest['hashes'].items()
               if Path(name).resolve() == Path(path).resolve()}
    require(len(matches) == 1, 'Input missing or ambiguous in manifest: ' + str(path))
    return next(iter(matches))


def state_sha(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@contextmanager
def no_external_access(allowed_torch_load=None):
    """Block network/hub APIs; allow at most one specified local weight path."""
    attempts, permitted = [], []
    real_load = torch.load
    def reject(kind):
        def denied(*args, **kwargs):
            attempts.append(kind)
            raise RuntimeError('Offline probe blocked ' + kind)
        return denied
    def guarded_load(path, *args, **kwargs):
        if allowed_torch_load is not None and isinstance(path, (str, os.PathLike)):
            if Path(path).resolve() == allowed_torch_load.resolve():
                permitted.append(str(allowed_torch_load))
                return real_load(path, *args, **kwargs)
        return reject('torch.load')()
    with ExitStack() as stack:
        for obj, name in ((socket.socket, 'connect'), (socket.socket, 'connect_ex'),
                          (socket, 'create_connection'), (socket, 'getaddrinfo')):
            stack.enter_context(patch.object(obj, name, reject('socket.' + name)))
        for name in ('load', 'download_url_to_file', 'load_state_dict_from_url'):
            stack.enter_context(patch.object(torch.hub, name, reject('torch.hub.' + name)))
        stack.enter_context(patch.object(torch, 'load', guarded_load))
        yield {'blocked_attempts': attempts, 'permitted_local_loads': permitted}
    require(not attempts, 'Constructor attempted external access, even if internally caught')


def compare(actual, expected, tolerance, label, report):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, label + ': shape mismatch')
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), label + ': nonfinite')
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    passed = bool(np.allclose(actual, expected, **tolerance))
    report[label] = dict(passed=passed, tolerance=tolerance,
        maximum_absolute_error=diff.max(axis=0).tolist(),
        maximum_relative_error=(diff / np.maximum(np.abs(expected), 1e-12)).max(axis=0).tolist())
    require(passed, label + ': frozen numerical tolerance exceeded')


def artifacts(args, report):
    manifest = read(args.manifest)
    epoch = manifest['recipe']['epochs']
    require(type(epoch) is int and epoch > 0, 'Invalid terminal epoch')
    run = args.manifest.parent / 'runs' / ('fold' + str(manifest['fold'])) / ARCH
    selected = args.checkpoint or run / f'epoch_{epoch:02d}.pt'
    result, complete = read(run/'result.json'), read(run/'complete.json')
    require(complete['status'] == 'completed', 'Training incomplete')
    require(complete['result_sha'] == sha(run/'result.json'), 'Result hash changed')
    require(result['manifest_sha'] == sha(args.manifest), 'Result/manifest mismatch')
    require(result['architecture'] == ARCH, 'Unexpected architecture')
    require(sha(run/'last.pt') == result['checkpoint_sha'], 'Terminal checkpoint hash changed')
    require(sha(run/'predictions.npz') == result['predictions_sha'], 'Prediction hash changed')
    checkpoint = torch.load(selected, map_location='cpu', weights_only=True)
    terminal = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
    for name, value in (('epoch', epoch), ('architecture', ARCH),
                        ('manifest_sha', sha(args.manifest))):
        require(checkpoint[name] == terminal[name] == value, 'Checkpoint metadata mismatch: ' + name)
    for name in ('log_mean', 'log_std'):
        np.testing.assert_array_equal(checkpoint[name], manifest[name])
        np.testing.assert_array_equal(terminal[name], manifest[name])
    tensor_sha = state_sha(checkpoint['model'])
    require(tensor_sha == state_sha(terminal['model']), 'Selected/terminal model tensors differ')
    del terminal
    require(len(result['history']) == epoch, 'Terminal history incomplete')
    require(result['history'][-1]['epoch'] == epoch, 'History terminal epoch mismatch')
    report['artifacts'] = dict(checkpoint=str(selected), checkpoint_sha256=sha(selected),
        terminal_checkpoint_sha256=result['checkpoint_sha'], model_tensor_sha256=tensor_sha,
        manifest=str(args.manifest), manifest_sha256=sha(args.manifest),
        result_sha256=complete['result_sha'], predictions_sha256=result['predictions_sha'],
        fold=manifest['fold'], epoch=epoch, terminal_state_equal=True,
        log_mean=manifest['log_mean'], log_std=manifest['log_std'])
    return manifest, selected, run, checkpoint


def data_checks(args, manifest, run, b, inf, report):
    meta = Path(b.META)
    split = args.split or Path(manifest.get('split_path',
        f'experiments/overnight_target_20260912/preflight/fold{manifest["fold"]}.npz'))
    for path in (meta, split):
        require(sha(path) == manifest_input_hash(manifest, path), 'Changed input: ' + str(path))
    with np.load(meta, allow_pickle=True) as z:
        ids, targets = z['ids'].astype(str), z['Y'].astype(np.float32)
    with np.load(split) as z:
        train, val = z['train'], z['validation']
    require(len(val) == manifest['validation_n'] and len(train) == manifest['train_n'], 'Split counts differ')
    require(len(val) >= 8 and len(ids) == manifest['images'], 'Insufficient/unexpected images')
    logs = np.log(targets[train].astype(np.float64))
    np.testing.assert_array_equal(logs.mean(0), manifest['log_mean'])
    np.testing.assert_array_equal(logs.std(0), manifest['log_std'])
    require(hashlib.sha256('\n'.join(ids[val]).encode()).hexdigest() == manifest['validation_ids_sha'],
            'Validation IDs/order changed')
    with np.load(run/'predictions.npz') as z:
        for name, expected in (('indices', val), ('ids', ids[val]), ('target', targets[val])):
            np.testing.assert_array_equal(z[name], expected)
        saved = z['pred'][:8].copy()
    require(saved.shape == (8, 3) and np.isfinite(saved).all() and (saved > 0).all(), 'Invalid saved predictions')
    require(Path(b.RAW).stat().st_size == manifest['raw_bytes'] == len(ids)*972*1296, 'Raw cache size changed')
    require(sha(args.fingerprints) == FINGERPRINT_SHA, 'Historical pixel fingerprints changed')
    with np.load(args.fingerprints) as z:
        fingerprints = z['sha256'].astype(str)
    require(len(fingerprints) == len(ids), 'Fingerprint row count mismatch')
    raw = np.memmap(b.RAW, mode='r', dtype=np.uint8, shape=(len(ids), 972, 1296))
    rows = val[:8].copy()
    images = [np.array(raw[int(index)], copy=True) for index in rows]
    original = b.Frames(rows, targets, len(ids), augment=False)
    for position, (index, image) in enumerate(zip(rows, images)):
        require(hashlib.sha256(image.tobytes()).hexdigest() == fingerprints[index], 'Selected raw pixels changed')
        x, _, found = original[position]
        require(found == index and torch.equal(inf.preprocess_native(image), x), 'Native preprocessing differs')
    x = torch.stack([original[i][0] for i in range(8)])
    expected = torch.nn.functional.pad(x.expand(-1, 3, -1, -1), (0, 16, 0, 20), mode='replicate').contiguous()
    require(torch.equal(inf.prepare_batch(x, 'cpu'), expected), 'Replication/padding differs')
    report['data'] = dict(n_checked=8, selection='first eight validation rows in frozen order',
        indices_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
        pixels='all eight match pinned historical SHA256 fingerprints',
        raw_full_sha256_from_training_manifest=manifest['raw_sha256'],
        entire_raw_cache_rehashed=False, preprocessing_bitwise_equal=True, padding_bitwise_equal=True,
        train_only_stats_recomputed=True, fingerprint_file_sha256=FINGERPRINT_SHA)
    return images, saved, rows, targets, len(ids)


def edge_checks(inf, report):
    constant = np.full((972, 1296), 127, dtype=np.uint8)
    actual = inf.preprocess_native(constant)
    import cv2
    clahe = cv2.createCLAHE(2.0, (40, 30)).apply(constant)
    lo, hi = np.percentile(clahe, (1, 99))
    expected = np.clip((clahe.astype(np.float32)-lo)/(hi-lo+1e-6), 0, 1)
    require(np.isfinite(actual.numpy()).all() and np.array_equal(actual.numpy()[0], expected),
            'Constant-image normalization differs')
    rejected = []
    for name, image in [('float32', constant.astype(np.float32)), ('uint16', constant.astype(np.uint16)),
                        ('wrong_size', constant[:-1]), ('three_channels', np.stack([constant]*3)),
                        ('empty', np.empty((0, 0), dtype=np.uint8))]:
        try:
            inf.preprocess_native(image)
        except (TypeError, ValueError):
            rejected.append(name)
        else:
            raise ValueError('Invalid preprocessing input accepted: ' + name)
    report['preprocessing_edge_cases'] = dict(constant_finite_and_exact=True, rejected=rejected)


def cpu_checks(inf, b, model, checkpoint, report):
    torch.manual_seed(11)
    x = torch.rand(1, 3, 64, 80)
    mode = model[0].mode if hasattr(model[0], 'mode') else None
    with torch.inference_mode():
        actual = model(x)
        tokens = model[0].original.forward_features(x)['x_norm_patchtokens']
        parts = []
        for stat in inf.POOLING_MODES[report['pooling']]:
            if stat == 'mean': value = tokens.mean(1)
            elif stat == 'std': value = (tokens.float().var(1, unbiased=False)+1e-12).sqrt().to(tokens.dtype)
            elif stat == 'min': value = tokens.amin(1)
            else: value = tokens.amax(1)
            parts.append(value)
        pooled = parts[0] if len(parts)==1 else torch.cat(parts, -1)
        expected = model[1:](pooled)
    compare(actual.numpy(), expected.numpy(), TOLERANCES['cpu_small_forward'], 'cpu_small_forward', report)


def gpu_checks(inf, b, model, mu, sigma, images, saved, rows, targets, total, report):
    require(torch.cuda.device_count() == 1, 'GPU mode requires exactly one visible GPU')
    require(torch.cuda.is_bf16_supported(), 'BF16 readback requires a BF16-capable GPU; no precision fallback')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    model.to('cuda'); mu = mu.to('cuda'); sigma = sigma.to('cuda')
    actual = inf.predict_images(model, images, mu, sigma, device='cuda', precision='bf16', batch_size=4)
    compare(actual, saved, TOLERANCES['gpu_same_batch_saved'], 'gpu_same_batch_saved', report)
    reference_loader = b.loader(rows, targets, total, augment=False)
    reference = b.evaluate(model, ARCH, reference_loader, mu, sigma, rows)
    compare(actual, reference, TOLERANCES['gpu_wrapper_reference'], 'gpu_wrapper_reference', report)
    one = inf.predict_images(model, images, mu, sigma, device='cuda', precision='bf16', batch_size=1)
    compare(one, actual, TOLERANCES['gpu_batch1_vs_batch4'], 'gpu_batch1_vs_batch4', report)
    report['gpu'] = dict(name=torch.cuda.get_device_name(), precision='bfloat16',
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        scope='Eight fixed cases only; memory observation is not a deployment benchmark')


def run(args, report):
    global np, torch
    if args.mode == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    torch.set_num_threads(2 if args.mode == 'cpu' else 4)
    report['runtime'] = dict(torch=torch.__version__, cuda_build=torch.version.cuda,
                           numpy=np.__version__, torch_cpu_threads=torch.get_num_threads())
    source = args.source_dir.resolve()
    for name, expected in SOURCE_SHA.items():
        require(sha(source/name) == expected, 'Frozen reference source changed: ' + name)
    sys.path.insert(0, str(source))
    report['sources'] = {name: sha(source/name) for name in (*SOURCE_SHA,
        'phase2_pooling_inference_20261020.py', Path(__file__).name)}
    guard = ExitStack()
    with guard:
        if args.mode == 'cpu':
            require(not torch.cuda.is_initialized(), 'CUDA initialized before CPU verification')
            guard.enter_context(patch.object(torch.cuda, '_lazy_init', side_effect=RuntimeError('CPU mode forbids CUDA initialization')))
        with no_external_access() as offline:
            inf = importlib.import_module('phase2_pooling_inference_20261020')
            probe = inf.build_model(read(args.manifest)['recipe']['pooling'])
        require(all(p.device.type == 'cpu' for p in probe.parameters()), 'Independent constructor must remain on CPU')
        del probe
        report['offline_constructor'] = dict(passed=True, **offline)
        b = importlib.import_module('phase2_foundation_train_20261002')
        manifest, selected, folder, checkpoint = artifacts(args, report)
        report['pooling'] = manifest['recipe']['pooling']
        # Permit reading only the chosen checkpoint while constructing/loading the independent model.
        with no_external_access(selected) as loading:
            model, mu, sigma, metadata = inf.load_checkpoint(selected, args.manifest, device='cpu',
                expected_checkpoint_sha256=report['artifacts']['checkpoint_sha256'])
        require(len(loading['permitted_local_loads']) == 1, 'Independent loader must read only the chosen checkpoint')
        require(state_sha(model.state_dict()) == report['artifacts']['model_tensor_sha256'], 'Loaded tensor hash differs')
        np.testing.assert_array_equal(mu.numpy(), np.asarray(manifest['log_mean'], dtype=np.float32))
        np.testing.assert_array_equal(sigma.numpy(), np.asarray(manifest['log_std'], dtype=np.float32))
        report['strict_independent_reload'] = dict(passed=True, external_weights_loaded=False)
        images, saved, rows, targets, total = data_checks(args, manifest, folder, b, inf, report)
        edge_checks(inf, report)
        if args.mode == 'cpu':
            # Reference initialization is authenticated by the selected run's own manifest.
            external = Path(b.WEIGHTS[ARCH])
            require(sha(external) == manifest_input_hash(manifest, external), 'Reference initialization changed')
            cpu_checks(inf, b, model, checkpoint, report)
            require(not torch.cuda.is_initialized(), 'CPU verification initialized CUDA')
            report['cuda_initialized'] = False
        else:
            del checkpoint
            gpu_checks(inf, b, model, mu, sigma, images, saved, rows, targets, total, report)
    report['status'] = 'passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('cpu', 'gpu'))
    parser.add_argument('--workspace', type=Path, default=Path('.'))
    parser.add_argument('--source-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--split', type=Path)
    parser.add_argument('--fingerprints', type=Path, default=Path('experiments/ssl_screen_20260911/preflight/pixel_fingerprints.npz'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(args.workspace)
    require(not args.output.exists(), 'Refusing to overwrite a readback report')
    report = dict(status='failed', mode=args.mode, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        tolerances=TOLERANCES, tolerance_policy='Frozen before observations; no CLI override or automatic relaxation',
        scope='Terminal checkpoint implementation readback, not winner selection or full-validation MAPE',
        official_submission_changed=False, t4_certified=False, raw_case_predictions_exported=False)
    try:
        run(args, report)
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
