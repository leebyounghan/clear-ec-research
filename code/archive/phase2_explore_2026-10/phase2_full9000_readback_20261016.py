"""Explicit CPU/GPU readback of one full-population terminal20 CPDINO model.

This consumes an already completed and audited run. It selects no recipe,
loads no Y/Y_train, computes no accuracy metric, and changes no training file.
The eight inputs are the predeclared ID-only memmap4/MHA4 reference cases.
CPU checks strict offline loading and real native preprocessing/padding.
GPU checks BF16 batch4 outputs exactly against this model's saved outputs and
batch1 versus batch4 at fixed rtol=.002/atol=1e-6. Existing fold-model readback
results do not satisfy this model's CPU or GPU evidence requirements.
"""
import argparse
from contextlib import ExitStack, contextmanager
import copy
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from unittest.mock import patch

HELPER_PINS = {
    'phase2_single_model_readback_20261015.py': '3e083e947b6dcb7fbf5b3ea833dfc1a94520e4d1512003b2697e9232176d33ef',
    'phase2_single_model_inference_20261015.py': '83c36203eb09fc2546346a25c6bafbf2ba041cd6b3c3a7e33e582e0cb60c86dc',
    'phase2_foundation_train_20261002.py': '20a849cb607c3c5777102e3266c6c1b5d9ba4533f9bfac9f1f6e9d18232dac2f',
    'phase2_foundation_preflight_20261002.py': '196500b67d01ad9423ea4d7d8c1a1e4b2d886df2328d36bfb1e1c94058ca4b1e',
    'phase2_lejepa_regression_20261003.py': 'f66fe903c2360c6e2392b407c55392dc1d7a45a03e624bed0acee8cbc32b33b8',
}
RUNNER_NAME = 'phase2_full9000_regression_20261016.py'
READER_NAME = 'phase2_full9000_data_20261016.py'
SCOPE = 'full9000_original_Y_positive_rows'
TOLERANCES = {'gpu_batch4_saved_exact': {'rtol': 0., 'atol': 0.},
              'gpu_batch1_vs_batch4': {'rtol': .002, 'atol': 1e-6}}
FORBIDDEN = {'fold', 'split_path', 'validation_indices', 'validation_ids_sha', 'validation_n',
             'val_n', 'score', 'retrospective_best', 'train_probe_score'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            'Input changed while hashing: ' + str(path))
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def import_pinned(directory, name, expected):
    path = directory/(name+'.py')
    require(sha(path) == expected, 'Helper source SHA differs: ' + name)
    sys.path.insert(0, str(directory))
    module = importlib.import_module(name)
    require(Path(module.__file__).resolve() == path.resolve() and sha(module.__file__) == expected,
            'Executed helper differs: ' + name)
    return module


def sources(args, report):
    required = dict(HELPER_PINS, **{READER_NAME: args.expected_reader_sha256})
    for name, expected in required.items():
        require(sha(args.source_dir/name) == expected, 'Unreviewed helper/reader: ' + name)
    report['sources'] = dict(helpers=required, readback_source_sha256=sha(__file__),
                             expected_runner_sha256=args.expected_runner_sha256)
    modules = {name[:-3]: import_pinned(args.source_dir, name[:-3], expected) for name, expected in required.items()}
    return (modules['phase2_single_model_readback_20261015'], modules['phase2_single_model_inference_20261015'],
            modules['phase2_foundation_train_20261002'], modules[READER_NAME[:-3]],
            modules['phase2_lejepa_regression_20261003'])


def artifacts(args, report):
    root = args.manifest.parent
    require(sha(args.manifest) == args.expected_manifest_sha256, 'Explicit expected manifest SHA differs')
    require(sha(root/'audit.json') == args.expected_audit_sha256, 'Explicit expected artifact audit SHA differs')
    m, audit = read(args.manifest), read(root/'audit.json')
    run = root/'run'; result, complete = read(run/'result.json'), read(run/'complete.json')
    require(m['schema'] == 'full9000_regression_v1' and m['training_scope'] == SCOPE and
            m['architecture'] == 'cpdino_vitb' and m['recipe']['epochs'] == 20 and
            m['population_n'] == 9000 and m['train_n'] == 8998 and m['heldout_n'] == 0 and
            m['zero_policy'] == 'exclude-nonpositive' and m['independent_accuracy_validation'] is False and
            not FORBIDDEN.intersection(m), 'Manifest is not a full-population terminal20 fit')
    execution = m['required_execution_contract']
    require(execution['runner_source_sha256'] == args.expected_runner_sha256 and
            execution['source_sha256'][READER_NAME] == args.expected_reader_sha256 and
            execution['active_loss_verified'] is True and execution['training_scope'] == SCOPE and
            execution['heldout_n'] == 0 and execution['independent_accuracy_validation'] is False,
            'Runner/reader/actual loss execution identity differs')
    for name, expected in execution['source_sha256'].items():
        require(Path(name).name == name and sha(root/'source'/name) == expected,
                'Manifest frozen implementation changed: ' + name)
    require(execution['source_sha256'][RUNNER_NAME] == args.expected_runner_sha256,
            'Frozen runner is not the explicitly expected implementation')
    require(m['expected_reader_sha256'] == args.expected_reader_sha256 and
            sha(Path(m['data_audit_path'])) == m['data_audit_sha256'], 'Bound reader/data audit differs')
    require(audit['status'] == 'full_population_regression_artifacts_verified_not_accuracy_validated' and
            audit['manifest_sha'] == args.expected_manifest_sha256 and audit['execution_contract'] == execution and
            audit['epoch_count'] == 20 and audit['exact_training_coverage_verified'] is True and
            audit['all_epoch_state_metadata_finite_verified'] is True and audit['terminal_strict_CPU_load_verified'] is True and
            audit['heldout_n'] == 0 and audit['independent_accuracy_validation'] is False and
            audit['cuda_initialized'] is False, 'Required full-population artifact audit is invalid')
    require(result['status'] == 'FULL_POPULATION_REGRESSION_COMPLETE_NOT_ACCURACY_VALIDATED' and
            result['schema'] == m['schema'] and result['training_scope'] == SCOPE and
            result['architecture'] == 'cpdino_vitb' and result['seed'] == m['seed'] and
            result['epochs'] == 20 and result['updates'] == 11260 and result['population_n'] == 9000 and
            result['train_n'] == 8998 and result['train_image_presentations'] == 179960 and
            result['heldout_n'] == 0 and result['independent_accuracy_validation'] is False and
            result['single_model'] is True and result['ranks_synchronized'] is True and
            result['terminal_checkpoint'] == 'epoch_20.pt' and result['manifest_sha'] == args.expected_manifest_sha256 and
            result['details']['execution_contract'] == execution and result['details']['initialization'] == m['initialization'] and
            not FORBIDDEN.intersection(result), 'Completed single-model run contract differs')
    selected = run/'epoch_20.pt'; prediction = run/'readback_reference.npz'
    checkpoint_sha, prediction_sha = sha(selected), sha(prediction)
    expected_complete = dict(status='completed', result_sha=sha(run/'result.json'), checkpoint_sha=checkpoint_sha,
                             manifest_sha=args.expected_manifest_sha256, reference_predictions_sha=prediction_sha)
    require(complete == expected_complete and result['checkpoint_sha'] == checkpoint_sha and
            result['reference_predictions_sha'] == prediction_sha, 'Completed terminal/reference chain differs')
    bindings = dict(result_sha=run/'result.json', complete_sha=run/'complete.json', checkpoint_sha=selected,
                    reference_predictions_sha=prediction, cpu_tests_sha=root/'cpu_tests.json',
                    smoke_sha=root/'smoke.json', current_data_gate_sha=run/'current_data_gate.json')
    for name, path in bindings.items():
        require(audit[name] == sha(path), 'Audited artifact changed: ' + name)
    require(result['history_sha'] == sha(run/'history.json') and result['history'] == read(run/'history.json') and
            len(result['history']) == 20 and result['history'][-1]['epoch'] == 20 and
            result['history'][-1]['weights_sha'] == checkpoint_sha and
            result['history'][-1]['model_tensor_sha'] == result['model_tensor_sha'], 'Terminal history chain differs')
    for name in ('cpu_tests', 'smoke'):
        check = read(root/(name+'.json'))
        require(check['status'] == 'passed' and check['manifest_sha'] == args.expected_manifest_sha256 and
                check['execution_contract'] == execution, 'Training prerequisite execution identity differs')
    reference = m['readback_reference']
    require(reference['heldout'] is False and reference['targets_saved'] is False and
            reference['accuracy_metric_computed'] is False and reference['precision'] == 'bf16' and
            reference['batch_size'] == 4 and reference['cross_batch_diagnostic_tolerance'] == TOLERANCES['gpu_batch1_vs_batch4'],
            'Fixed reference inference contract differs')
    report['artifacts'] = dict(manifest=str(args.manifest), manifest_sha256=args.expected_manifest_sha256,
        audit_sha256=args.expected_audit_sha256, checkpoint=str(selected), checkpoint_sha256=checkpoint_sha,
        reference_predictions_sha256=prediction_sha, result_sha256=sha(run/'result.json'),
        complete_sha256=sha(run/'complete.json'), runner_sha256=args.expected_runner_sha256,
        reader_sha256=args.expected_reader_sha256, epoch=20, model_tensor_sha256=result['model_tensor_sha'])
    return m, result, selected, prediction


@contextmanager
def image_ids_only():
    # A failed attempt is recorded even if a reader were to catch the exception.
    original = np.lib.npyio.NpzFile.__getitem__
    denied = []
    def guarded(obj, key):
        if key != 'ids':
            denied.append(str(key))
            raise ValueError('Image-only readback refuses NPZ array: ' + str(key))
        return original(obj, key)
    with patch.object(np.lib.npyio.NpzFile, '__getitem__', guarded):
        yield
    require(not denied, 'Reader attempted to load a non-ID NPZ array')


def image_checks(args, manifest, prediction_path, reader_module, foundation, inf, report):
    with image_ids_only(), patch.object(reader_module, 'load_original_targets',
                                       side_effect=RuntimeError('Readback forbids original target loading')):
        reader = reader_module.Full9000Reader(args.workspace, manifest['data_audit_path'],
                                              expected_audit_sha256=manifest['data_audit_sha256'])
        require(reader.target_arrays_loaded is False and len(reader) == 9000, 'Reader scope differs')
        groups = [np.flatnonzero(reader.mapping >= 0), np.flatnonzero(reader.mapping < 0)]
        indices = np.asarray([i for group in groups for i in
                              sorted(group.tolist(), key=lambda j:str(reader.ids[j]))[:4]], dtype=np.int64)
        require(indices.shape == (8,), 'Need ID-only first4 from each pixel source')
        records = manifest['readback_reference']['records']
        require(len(records) == 8, 'Manifest reference count differs')
        images = []
        for position, (index, record) in enumerate(zip(indices, records)):
            expected_source = 'native8954_memmap' if position < 4 else 'extra46_mha'
            require(record['index'] == int(index) and record['id'] == str(reader.ids[index]) and
                    record['source'] == expected_source, 'Reference selection/ID/backend differs')
            image = np.array(reader.raw_frame(int(index)), copy=True)
            require(image.dtype == np.uint8 and image.shape == (972,1296) and
                    hashlib.sha256(image.tobytes()).hexdigest() == record['raw_pixel_sha256'],
                    'Current reference raw pixels differ')
            images.append(image)
    with np.load(prediction_path, allow_pickle=False) as saved:
        require(set(saved.files) == {'indices','ids','pred','raw_pixel_sha256'}, 'Reference artifact contains unexpected fields')
        np.testing.assert_array_equal(saved['indices'], indices)
        np.testing.assert_array_equal(saved['ids'], reader.ids[indices])
        np.testing.assert_array_equal(saved['raw_pixel_sha256'], [v['raw_pixel_sha256'] for v in records])
        predictions = saved['pred'].copy()
    require(predictions.shape == (8,3) and predictions.dtype == np.float32 and
            np.isfinite(predictions).all() and (predictions > 0).all(), 'Saved reference predictions invalid')
    # Reuse the actual pinned training __getitem__ transform for BOTH backends.
    # Its target return slot is empty: no numeric labels are supplied or read.
    original = foundation.Frames(np.arange(8), np.empty((8,0), dtype=np.float32), 8, augment=False)
    original.raw = np.stack(images)
    normalized = []
    for position, image in enumerate(images):
        expected, unused, found = original[position]
        require(found == position and unused.numel() == 0 and
                torch.equal(inf.preprocess_native(image), expected), 'Native preprocessing parity failed')
        normalized.append(expected)
    x = torch.stack(normalized)
    padded = torch.nn.functional.pad(x.expand(-1,3,-1,-1), (0,16,0,20), mode='replicate').contiguous()
    require(torch.equal(inf.prepare_batch(x, 'cpu'), padded), 'Channel replication/native padding parity failed')
    require(manifest['data_index'] == reader.provenance['index'], 'Reader population map changed')
    report['inputs'] = dict(count=8, memmap_inputs=4, MHA_inputs=4,
        selection='lexicographic ID first4 per pixel source; outcome independent',
        reference_input_ids=reader.ids[indices].tolist(), reference_raw_pixel_sha256=[v['raw_pixel_sha256'] for v in records],
        every_reference_raw_pixel_hash_verified=True, all46_MHA_current_reader_binding_verified=True,
        preprocessing_bitwise_equal=True, channel_replication_padding_bitwise_equal=True,
        original_target_arrays_loaded=False, accuracy_metrics_computed=False, entire8954_raw_rehashed=False,
        scope='Fixed full-population reproduction cases; not held-out accuracy')
    return images, predictions


def cpu_report_gate(args, report):
    require(args.cpu_report is not None and args.expected_cpu_report_sha256 is not None,
            'GPU readback requires this model\'s completed CPU report and explicit SHA')
    require(sha(args.cpu_report) == args.expected_cpu_report_sha256, 'Expected CPU readback report SHA differs')
    cpu = read(args.cpu_report)
    require(cpu['status'] == 'passed' and cpu['mode'] == 'cpu' and cpu['cuda_initialized'] is False and
            cpu['artifacts'] == report['artifacts'] and cpu['sources'] == report['sources'] and
            cpu['tolerances'] == TOLERANCES and cpu['strict_independent_reload']['passed'] is True and
            cpu['inputs']['preprocessing_bitwise_equal'] is True and
            cpu['independent_accuracy_validation'] is False, 'CPU report belongs to another model/scope/source')
    report['cpu_readback_sha256'] = args.expected_cpu_report_sha256


def fixture_tests(source_dir):
    """Synthetic CPU contracts only; no foundation model or experiment manifest."""
    global np, torch
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), 'Fixtures must start without CUDA')
    legacy = import_pinned(source_dir, 'phase2_single_model_readback_20261015',
                           HELPER_PINS['phase2_single_model_readback_20261015.py'])
    inf = import_pinned(source_dir, 'phase2_single_model_inference_20261015',
                        HELPER_PINS['phase2_single_model_inference_20261015.py'])
    legacy.np, legacy.torch = np, torch
    cases = []
    def reject(name, operation):
        try:
            operation()
        except ValueError:
            cases.append(name)
        else:
            raise AssertionError('Fixture accepted invalid contract: ' + name)
    with patch.object(torch.cuda, '_lazy_init', side_effect=RuntimeError('Fixtures forbid CUDA')), \
         patch.object(inf, 'build_model', side_effect=RuntimeError('Fixtures forbid model construction')), \
         legacy.no_external_access() as offline:
        require(legacy.np is np and legacy.torch is torch, 'Legacy helper globals were not bound')
        x = np.asarray([[1.,2.,3.],[4.,5.,6.]], dtype=np.float32)
        comparison = {}
        legacy.compare(x, x.copy(), TOLERANCES['gpu_batch4_saved_exact'], 'exact', comparison)
        cases.append('legacy runtime globals and exact comparison initialized')
        reject('exact comparison rejects changed output', lambda: legacy.compare(
            x+.001, x, TOLERANCES['gpu_batch4_saved_exact'], 'different', {}))
        legacy.compare(x+.001, x, TOLERANCES['gpu_batch1_vs_batch4'], 'within', comparison)
        reject('fixed cross-batch tolerance rejects excessive drift', lambda: legacy.compare(
            x+.1, x, TOLERANCES['gpu_batch1_vs_batch4'], 'outside', {}))
        edges = {}; legacy.edge_checks(inf, edges)
        require(edges['preprocessing_edge_cases']['constant_finite_and_exact'], 'Legacy edge helper globals differ')
        cases.append('legacy preprocessing edge helper initialized')
        with tempfile.TemporaryDirectory(prefix='full9000-readback-fixture-') as directory:
            directory = Path(directory)
            pixels_meta = directory/'synthetic_ids.npz'
            np.savez(pixels_meta, ids=np.asarray(['synthetic-a','synthetic-b']),
                     Y=np.asarray([object()]), Y_train=np.asarray([object()]))
            with image_ids_only(), np.load(pixels_meta, allow_pickle=False) as loaded:
                np.testing.assert_array_equal(loaded['ids'], ['synthetic-a','synthetic-b'])
            cases.append('image-only guard allows IDs without deserializing targets')
            for forbidden in ('Y','Y_train'):
                def swallowed_denial(key=forbidden):
                    with image_ids_only(), np.load(pixels_meta, allow_pickle=False) as loaded:
                        try:
                            loaded[key]
                        except ValueError:
                            pass  # The context must still fail after this caught violation.
                reject('image-only guard detects caught denial for '+forbidden, swallowed_denial)
            # Check the guard was restored, using a harmless IDs access.
            with np.load(pixels_meta, allow_pickle=False) as loaded:
                require(loaded['ids'].shape == (2,), 'NPZ guard did not restore')
            report = dict(artifacts={'checkpoint_sha256':'1'*64, 'manifest_sha256':'2'*64},
                          sources={'runner_sha256':'3'*64, 'readback_sha256':sha(__file__)})
            cpu = dict(status='passed', mode='cpu', cuda_initialized=False,
                       artifacts=copy.deepcopy(report['artifacts']), sources=copy.deepcopy(report['sources']),
                       tolerances=TOLERANCES, strict_independent_reload={'passed':True},
                       inputs={'preprocessing_bitwise_equal':True}, independent_accuracy_validation=False)
            path = directory/'synthetic_cpu_report.json'
            def check_cpu(value):
                path.write_text(json.dumps(value, allow_nan=False))
                args = argparse.Namespace(cpu_report=path, expected_cpu_report_sha256=sha(path))
                cpu_report_gate(args, report)
            check_cpu(cpu); cases.append('matching synthetic CPU report accepted')
            for key, field in (('artifacts','checkpoint_sha256'), ('sources','runner_sha256'),
                               ('sources','readback_sha256')):
                changed = copy.deepcopy(cpu); changed[key][field] = 'f'*64
                reject('CPU report rejects changed '+key+'.'+field, lambda value=changed: check_cpu(value))
            changed = copy.deepcopy(cpu); changed['mode'] = 'gpu'
            reject('GPU result cannot substitute for CPU readback', lambda: check_cpu(changed))
        require(not offline['blocked_attempts'] and not offline['permitted_local_loads'], 'Fixture accessed model weights/network')
    require(not torch.cuda.is_initialized(), 'Fixtures initialized CUDA')
    return dict(status='fixture_tests_passed', foundation_model_constructed=False,
                cuda_initialized=False, experiment_manifest_created=False, real_data_or_checkpoint_used=False,
                synthetic_temporary_inputs_only=True, helper_globals_verified=True,
                source_sha256=sha(__file__), helper_source_sha256={name:HELPER_PINS[name] for name in
                    ('phase2_single_model_readback_20261015.py','phase2_single_model_inference_20261015.py')},
                tolerances=TOLERANCES, test_count=len(cases), tests=cases,
                scope='Readback helper/guard/comparison/report contracts; no real terminal checkpoint was verified')


def run(args, report):
    global np, torch
    if args.mode == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    torch.set_num_threads(2 if args.mode == 'cpu' else 4)
    legacy, inf, foundation, reader_module, prior = sources(args, report)
    # Reuse pure guard/comparison/edge helpers only; never legacy fold orchestration.
    legacy.np, legacy.torch = np, torch
    manifest, result, checkpoint, prediction = artifacts(args, report)
    report['runtime'] = dict(torch=str(torch.__version__), cuda_build=torch.version.cuda,
                             numpy=np.__version__, torch_cpu_threads=torch.get_num_threads())
    with ExitStack() as guard:
        if args.mode == 'cpu':
            require(not torch.cuda.is_initialized(), 'CPU readback started after CUDA initialization')
            guard.enter_context(patch.object(torch.cuda, '_lazy_init',
                                            side_effect=RuntimeError('CPU readback forbids CUDA initialization')))
        with legacy.no_external_access() as offline:
            probe = inf.build_model()
        require(all(p.device.type == 'cpu' for p in probe.parameters()), 'Offline constructor left CPU')
        del probe
        report['offline_constructor'] = dict(passed=True, **offline)
        with legacy.no_external_access(checkpoint) as loading:
            model, mu, sigma, metadata = inf.load_checkpoint(checkpoint, args.manifest, device='cpu',
                expected_checkpoint_sha256=report['artifacts']['checkpoint_sha256'])
        require(len(loading['permitted_local_loads']) == 1 and not loading['blocked_attempts'],
                'Independent load must read only this terminal checkpoint')
        require(prior.state_sha(model.state_dict()) == result['model_tensor_sha'], 'Strict-loaded model tensor hash differs')
        np.testing.assert_array_equal(mu.numpy(), np.asarray(manifest['log_mean'], dtype=np.float32))
        np.testing.assert_array_equal(sigma.numpy(), np.asarray(manifest['log_std'], dtype=np.float32))
        require(metadata['epoch'] == 20, 'Loaded checkpoint is not terminal20')
        report['strict_independent_reload'] = dict(passed=True, checkpoint_reads=1, external_pretrained_weights_loaded=False,
                                                  terminal_epoch=20, log_stats_exact=True, all_model_tensors_equal=True)
        images, saved = image_checks(args, manifest, prediction, reader_module, foundation, inf, report)
        legacy.edge_checks(inf, report)
        if args.mode == 'cpu':
            require(not torch.cuda.is_initialized(), 'CPU readback initialized CUDA')
            report.update(cuda_initialized=False, model_forward_performed=False,
                          native_preprocessing_verified=True, GPU_forward_verified=False)
        else:
            cpu_report_gate(args, report)
            require(torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(), 'One BF16-capable GPU required')
            require(str(torch.__version__) == result['details']['torch_version'] and
                    torch.version.cuda == result['details']['cuda_version'] and
                    torch.cuda.get_device_name() == result['details']['gpu'],
                    'Exact saved-batch readback requires the training Torch/CUDA/GPU backend')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            model.to('cuda'); mu = mu.to('cuda'); sigma = sigma.to('cuda')
            actual = inf.predict_images(model, images, mu, sigma, device='cuda', precision='bf16', batch_size=4)
            legacy.compare(actual, saved, TOLERANCES['gpu_batch4_saved_exact'], 'gpu_batch4_saved_exact', report)
            require(np.array_equal(actual, saved), 'Batch4 predictions must be exactly equal; no tolerance relaxation')
            one = inf.predict_images(model, images, mu, sigma, device='cuda', precision='bf16', batch_size=1)
            legacy.compare(one, actual, TOLERANCES['gpu_batch1_vs_batch4'], 'gpu_batch1_vs_batch4', report)
            report['gpu'] = dict(name=torch.cuda.get_device_name(), precision='bfloat16',
                batch4_saved_exact=True, batch1_vs_batch4_fixed_tolerance_passed=True,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                scope='Eight fixed full-population cases only; no deployment benchmark')
            report.update(cuda_initialized=torch.cuda.is_initialized(), model_forward_performed=True,
                          GPU_forward_verified=True)
    # Recheck immutable evidence after the potentially slow model operations.
    require(sha(args.manifest) == args.expected_manifest_sha256 and
            sha(args.manifest.parent/'audit.json') == args.expected_audit_sha256 and
            sha(checkpoint) == report['artifacts']['checkpoint_sha256'] and
            sha(prediction) == report['artifacts']['reference_predictions_sha256'], 'Evidence changed during readback')
    report['status'] = 'passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['cpu','gpu','fixture-tests'])
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--source-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--expected-manifest-sha256')
    parser.add_argument('--expected-audit-sha256')
    parser.add_argument('--expected-runner-sha256')
    parser.add_argument('--expected-reader-sha256')
    parser.add_argument('--cpu-report', type=Path)
    parser.add_argument('--expected-cpu-report-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode == 'fixture-tests':
        require(all(value is None for key,value in vars(args).items() if key not in ('mode','source_dir')),
                'Synthetic fixtures accept only --source-dir, no experiment options')
        print(json.dumps(fixture_tests(args.source_dir.resolve()), indent=2, allow_nan=False)); return
    for key in ('workspace','manifest','expected_manifest_sha256','expected_audit_sha256',
                'expected_runner_sha256','expected_reader_sha256','output'):
        require(getattr(args,key) is not None, 'Explicit argument required: --'+key.replace('_','-'))
    for key in ('expected_manifest_sha256','expected_audit_sha256','expected_runner_sha256',
                'expected_reader_sha256','expected_cpu_report_sha256'):
        value = getattr(args,key)
        require(value is None or re.fullmatch('[0-9a-f]{64}', value) is not None, 'Invalid expected SHA256: '+key)
    require((args.mode == 'gpu') == (args.cpu_report is not None and args.expected_cpu_report_sha256 is not None),
            'Only GPU mode requires both CPU report arguments')
    if args.mode == 'cpu':
        require(args.cpu_report is None and args.expected_cpu_report_sha256 is None, 'CPU mode accepts no prior CPU report')
    args.workspace = args.workspace.resolve()
    for key in ('source_dir','manifest','output','cpu_report'):
        value = getattr(args,key)
        if value is not None:
            setattr(args,key,(args.workspace/value).resolve())
    os.chdir(args.workspace)
    require(not args.output.exists(), 'Readback output already exists; never overwrite evidence')
    report = dict(status='failed', mode=args.mode, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='This full-population terminal20 checkpoint and its predeclared reference8; no prior model proof reused',
        tolerances=TOLERANCES, tolerance_policy='Predeclared; no CLI override or automatic relaxation',
        heldout_n=0, independent_accuracy_validation=False, original_target_arrays_loaded=False,
        accuracy_metrics_computed=False, single_model=True, official_submission_changed=False,
        t4_certified=False, raw_case_predictions_exported=False)
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
