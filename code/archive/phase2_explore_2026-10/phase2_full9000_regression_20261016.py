"""Conditional full-population CPDINO fit; this source selects no experiment.

Only an explicit, SHA-bound selected recipe and ``--zero-policy
exclude-nonpositive`` can prepare an isolated run. Original Y is preserved;
8998 positive rows train, two nonpositive rows do not. There is no held-out
set, validation score, checkpoint selection, ensemble, TTA or implicit resume.
The fixed terminal epoch20 is an artifact, not evidence of unseen accuracy.

``model-free-tests`` uses synthetic CPU tensors only. All other commands need
the same explicit inputs. Preparing or running this file requires a separate
experiment decision; merely implementing it does not make that decision.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

import phase2_crossfold_regression_20261015 as r
import phase2_crossfold_loss_ablation_20261016 as loss_module
import phase2_full9000_data_20261016 as data
import phase2_single_model_inference_20261015 as inference

b, prior = r.b, r.prior
ARCH = 'cpdino_vitb'
SCHEMA = 'full9000_regression_v1'
SELECTION_SCHEMA = 'full9000_regression_selected_recipe_v1'
SCOPE = 'full9000_original_Y_positive_rows'
RESULT_STATUS = 'FULL_POPULATION_REGRESSION_COMPLETE_NOT_ACCURACY_VALIDATED'
AUDIT_STATUS = 'full_population_regression_artifacts_verified_not_accuracy_validated'
FORBIDDEN_SCORE_FIELDS = {'fold', 'split_path', 'split_sha256', 'validation_indices', 'validation_ids_sha',
                          'validation_n', 'val_n', 'score', 'retrospective_best', 'train_probe_score'}
MAX_DEADLINE = '2026-10-19T03:59:00Z'
EXTERNAL_SHA = '3ed4c06a3963ab13ff377d4e2957174aaaf637434eb031ae9931fcbfaf9a217f'
PINNED = dict(loss_module.ORIGINAL_SOURCE_SHA256, **{
    'phase2_crossfold_loss_ablation_20261016.py': '0e2c9ab6522f8d361d0d3ae7209bcae99c3ec63b267b009264e90f546e7b94f5',
    'phase2_single_model_inference_20261015.py': '83c36203eb09fc2546346a25c6bafbf2ba041cd6b3c3a7e33e582e0cb60c86dc',
})
SSL_ORIGINAL_PINS = {
    'phase2_crossfold_ssl_20261015.py': '21d1afe8905fba16cc69e10bbc0ba5e8f2e7839345cadad69ee0151dab546f07',
    'phase2_lejepa_20261003.py': '34ce7a35a6b4fa72fa6deb07ba5b4523bb31c21616b0b98c9ea88be6b6ddc88f',
    'phase2_foundation_preflight_20261002.py': '196500b67d01ad9423ea4d7d8c1a1e4b2d886df2328d36bfb1e1c94058ca4b1e',
}
EXECUTION = dict(world_size=2, micro_batch_per_gpu=4, global_batch=16,
                 accumulation_rounds=2, partial_groups='sample weighted; zero-weight dummy as needed',
                 actual_final_batch=6, actual_final_microbatch_counts=[4, 2, 0, 0],
                 actual_final_microbatch_order='round-major: round0 rank0/rank1, round1 rank0/rank1',
                 train_loader_workers=2, train_loader_prefetch=1,
                 deadline_update_reserve_seconds=90, external_absolute_termination_required=True,
                 loader_rng='global Torch generator; no validation/probe loader or RNG consumption emulation')
LIMITS = ['No held-out accuracy estimate exists for this fit',
          'Original 9000 Y retained; only 8998 fully positive rows supervise regression',
          'Terminal epoch20 only; no model selection from training metrics',
          'Reference8 outputs are reproduction evidence, never validation MAPE',
          'Single model; no ensemble, TTA, calibration or exact resume',
          'Native972x1296 research inputs; no official submission or T4 certification']


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return data.sha256(path)


def read(path):
    return json.loads(Path(path).read_text())


def index_sha(indices):
    return r.array_sha(np.asarray(indices, dtype='<i8'))


def exclusive_json(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def atomic_json(path, value):
    # Only mutable progress/history within this fresh run may be replaced.
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    exclusive_json(temporary, value)
    temporary.replace(path)


def exclusive_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('xb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)  # Atomic publication; an existing artifact is never overwritten.
    temporary.unlink()


def validate_selection(value):
    fields = {'schema', 'architecture', 'initialization', 'seed', 'epochs', 'schedule_epochs',
              'encoder_lr', 'head_lr', 'gradient_clip', 'weight_decay',
              'standardized_log_l1_coefficient', 'relative_error_coefficient', 'selection_evidence'}
    require(isinstance(value, dict) and set(value) == fields, 'Selected recipe fields differ from explicit schema')
    require(value['schema'] == SELECTION_SCHEMA and value['architecture'] == ARCH, 'Unsupported selected recipe')
    require(value['initialization'] in ('SSL0_EXTERNAL_PRETRAINED', 'SSL5_FULL9000'), 'Select SSL0 or full9000 SSL5 explicitly')
    for key in ('seed', 'epochs', 'schedule_epochs'):
        require(type(value[key]) is int, 'Recipe integer required: ' + key)
    require(0 <= value['seed'] < 2**32 - 1 and value['epochs'] == 20 and value['schedule_epochs'] >= 2,
            'Seed range, fixed20 epochs or schedule contract differs')
    for key in ('encoder_lr', 'head_lr', 'gradient_clip', 'weight_decay',
                'standardized_log_l1_coefficient', 'relative_error_coefficient'):
        require(type(value[key]) in (int, float) and math.isfinite(value[key]), 'Finite scalar required: ' + key)
    require(value['encoder_lr'] == 1e-5 and value['weight_decay'] == 1e-4 and
            value['head_lr'] > 0 and value['gradient_clip'] > 0, 'Unsupported optimizer recipe')
    require(0 <= value['standardized_log_l1_coefficient'] <= .5 and value['relative_error_coefficient'] == 2.5,
            'Only declared log-L1 [0,.5] with fixed physical-relative2.5 is supported')
    evidence = value['selection_evidence']
    require(isinstance(evidence, list) and len(evidence) > 0, 'Selection requires explicit evidence artifact references')
    for item in evidence:
        require(isinstance(item, dict) and set(item) == {'path', 'sha256'} and
                isinstance(item['path'], str) and bool(item['path']) and
                isinstance(item['sha256'], str) and re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None,
                'Each selection evidence reference requires exactly path and lowercase SHA256')
    require(len({item['path'] for item in evidence}) == len(evidence), 'Duplicate selection evidence path')
    return dict(value)


def selected_recipe(args):
    require(sha(args.selected_recipe) == args.expected_recipe_sha256, 'Selected recipe SHA mismatch')
    value = validate_selection(read(args.selected_recipe))
    for item in value['selection_evidence']:
        require(sha(args.workspace/Path(item['path'])) == item['sha256'], 'Selection evidence artifact changed')
    require((args.ssl_root is not None) == (value['initialization'] == 'SSL5_FULL9000'),
            'ssl-root is required exactly when full9000 SSL5 is selected')
    require((args.expected_ssl_source_sha256 is not None) == (args.ssl_root is not None),
            'SSL5 requires explicit expected SSL runner source SHA')
    return value


def recipe_from_selection(value):
    recipe = {key: value[key] for key in ('seed', 'epochs', 'schedule_epochs', 'encoder_lr', 'head_lr',
                                         'gradient_clip', 'weight_decay')}
    recipe.update(loss_module.recipe_loss(value['standardized_log_l1_coefficient']),
                  micro_batch=4, effective_batch=16, warmup_epochs=1, minimum_lr_ratio=.1,
                  precision='bfloat16', checkpointing=False,
                  head='mean pool -> LayerNorm -> Linear128 -> GELU -> Linear3',
                  view='whole native972x1296; replicate-pad bottom20/right16 to992x1312',
                  normalization='CLAHE+percentile01; Cellpose01',
                  augmentation='independent deterministic H/V flips; seed, zero-based epoch, final9000 index',
                  loader_rng=EXECUTION['loader_rng'], checkpoint_policy='fixed terminal epoch20; no best checkpoint',
                  labels='original Y; explicitly exclude any nonpositive row; positive-train log mean/std',
                  diagnostics='online training metrics only; no independent accuracy validation')
    return recipe


def source_files():
    modules = (r, b, r.prior, r.preflight, loss_module, data, inference)
    return [Path(__file__).resolve(), *[Path(module.__file__).resolve() for module in modules]]


def check_sources(args):
    values = {path.name: sha(path) for path in source_files()}
    for name, expected in PINNED.items():
        require(values[name] == expected, 'Pinned implementation changed: ' + name)
    require(values[Path(data.__file__).name] == args.expected_reader_sha256, 'Reader source SHA mismatch')
    return values


def activate_loss(args, selection):
    check_sources(args)
    active = partial(loss_module.loss_formula,
                     standardized_log_l1_coefficient=selection['standardized_log_l1_coefficient'])
    b.loss_fn = active  # Process-local function binding; frozen source files remain unchanged.
    return active


def execution_contract(args, selection, active):
    require(b.loss_fn is active and isinstance(active, partial) and active.func is loss_module.loss_formula
            and not active.args and active.keywords == {
                'standardized_log_l1_coefficient': selection['standardized_log_l1_coefficient']},
            'Actual loss callable/coefficient differs from selected recipe')
    return dict(runner_source_sha256=sha(__file__), source_sha256=check_sources(args),
                selected_recipe_sha256=args.expected_recipe_sha256,
                loss_name=loss_module.LOSS_NAME,
                standardized_log_l1_coefficient=float(selection['standardized_log_l1_coefficient']),
                relative_error_coefficient=2.5, active_loss_verified=True,
                training_scope=SCOPE, heldout_n=0, independent_accuracy_validation=False)


def positive_training_rows(ids, targets, zero_policy):
    require(zero_policy == 'exclude-nonpositive', 'An explicit supported zero policy is required')
    require(targets.shape == (len(ids), 3) and targets.dtype == np.float32 and
            np.isfinite(targets).all() and (targets >= 0).all(), 'Invalid original target values')
    train = np.flatnonzero((targets > 0).all(1)).astype(np.int64)
    excluded = np.flatnonzero(~(targets > 0).all(1)).astype(np.int64)
    require(len(train) > 0, 'No positive supervision rows')
    logs = np.log(targets[train].astype(np.float64))
    mu, sigma = logs.mean(0), logs.std(0)
    require(np.isfinite(mu).all() and np.isfinite(sigma).all() and (sigma > 0).all(), 'Invalid train log statistics')
    return train, excluded, mu, sigma


def open_data(args):
    reader = data.Full9000Reader(args.workspace, args.data_audit, expected_audit_sha256=args.expected_data_audit_sha256)
    labels = data.load_original_targets(args.workspace, args.data_audit,
                                       expected_audit_sha256=args.expected_data_audit_sha256)
    np.testing.assert_array_equal(reader.ids, labels['ids'])
    train, excluded, mu, sigma = positive_training_rows(reader.ids, labels['targets'], args.zero_policy)
    require(len(reader) == 9000 and len(train) == 8998 and len(excluded) == 2 and
            set(reader.ids[excluded]) == data.ZERO_IDS, 'Unexpected full-population/zero inventory')
    require(labels['provenance']['target_array'] == 'Y' and labels['provenance']['Y_train_loaded'] is False,
            'Only original Y may supervise this fit')
    return reader, labels, train, excluded, mu, sigma


def reference_indices(ids, mapping):
    # Uses source kind and IDs only, never labels, scores or model predictions.
    groups = [np.flatnonzero(mapping >= 0), np.flatnonzero(mapping < 0)]
    require(all(len(group) >= 4 for group in groups), 'Need four references from each pixel source')
    return np.asarray([i for group in groups for i in sorted(group.tolist(), key=lambda j: str(ids[j]))[:4]], dtype=np.int64)


def reference_contract(reader):
    indices = reference_indices(reader.ids, reader.mapping)
    rows = [dict(index=int(i), id=str(reader.ids[i]),
                 source='native8954_memmap' if reader.mapping[i] >= 0 else 'extra46_mha',
                 raw_pixel_sha256=r.array_sha(np.ascontiguousarray(reader.raw_frame(int(i))))) for i in indices]
    return dict(selection='lexicographic ID first4 per pixel source; memmap4 then MHA4; no label/score use',
                records=rows, heldout=False, targets_saved=False, accuracy_metric_computed=False,
                precision='bf16', batch_size=4, same_batch_comparison='exact array equality; same pinned runtime/backend',
                cross_batch_diagnostic_tolerance=dict(rtol=.002, atol=1e-6))


def ssl_evidence(args, ids):
    root = args.ssl_root
    manifest_path, run = root / 'manifest.json', root / 'run'
    m, result, complete = read(manifest_path), read(run/'result.json'), read(run/'complete.json')
    audited = read(root/'audit.json')
    manifest_sha = sha(manifest_path)
    expected = dict(schema='full9000_ssl_v1', training_scope='full9000_label_free', population_n=9000,
                    population_ids_sha=r.ids_sha(ids), indices_sha=index_sha(np.arange(9000)), heldout_n=0,
                    independent_accuracy_validation=False, target_arrays_loaded=False, seed=1003,
                    epochs=5, steps_per_epoch=282, total_steps=1410, last_batch_images=8)
    for key, value in expected.items():
        require(m.get(key) == value, 'Full-population SSL manifest mismatch: ' + key)
    require(m['data_audit_sha256'] == args.expected_data_audit_sha256, 'SSL/regression data audit differs')
    for path, expected_sha in m['hashes'].items():
        require(sha(path) == expected_sha, 'Full-population SSL bound input changed: ' + path)
    source = root/'source'/'phase2_full9000_ssl_20261016.py'
    require(sha(source) == args.expected_ssl_source_sha256, 'Full-population SSL runner source mismatch')
    require(m['external_checkpoint_sha'] == EXTERNAL_SHA, 'SSL did not start from original external weights')
    require(m['canonical_encoder_state'] is True and m['reader_source_sha256'] == args.expected_reader_sha256 and
            m['original_source_sha256'] == SSL_ORIGINAL_PINS and m['runner_source_sha256'] == args.expected_ssl_source_sha256,
            'SSL canonical/source contract differs')
    require(result['status'] == 'SSL_FULL_POPULATION_COMPLETE_NOT_ACCURACY_VALIDATED' and
            result['manifest_sha'] == manifest_sha and result['epochs'] == 5 and
            result['steps'] == result['total_steps'] == 1410 and result['population_n'] == 9000 and
            result['train_image_presentations'] == 45000 and result['heldout_n'] == 0 and
            result['target_arrays_loaded'] is False and result['independent_accuracy_validation'] is False,
            'SSL completion/scope mismatch')
    require(complete == dict(status='completed', result_sha=sha(run/'result.json'),
                             checkpoint_sha=result['checkpoint_sha'], manifest_sha=manifest_sha),
            'SSL complete/result chain mismatch')
    require(sha(run/'last.pt') == result['checkpoint_sha'] == sha(run/'epoch05.pt'), 'SSL terminal checkpoint chain mismatch')
    require(audited['status'] == 'full_population_ssl_artifacts_verified_no_heldout_score' and
            audited['manifest_sha'] == manifest_sha and audited['result_sha'] == sha(run/'result.json') and
            audited['complete_sha'] == sha(run/'complete.json') and audited['checkpoint_sha'] == result['checkpoint_sha'] and
            audited['cpu_tests_sha'] == sha(root/'cpu_tests.json') and audited['smoke_sha'] == sha(root/'smoke.json') and
            audited['epochs_sha'] == sha(run/'epochs.json') and audited['history_sha'] == sha(run/'history.json') and
            audited['runner_source_sha256'] == args.expected_ssl_source_sha256 and
            audited['reader_source_sha256'] == args.expected_reader_sha256 and
            audited['original_source_sha256'] == SSL_ORIGINAL_PINS and audited['external_checkpoint_sha'] == EXTERNAL_SHA and
            audited['population_ids_sha'] == r.ids_sha(ids) and audited['population_n'] == 9000 and
            audited['heldout_n'] == 0 and audited['target_arrays_loaded'] is False and
            audited['independent_accuracy_validation'] is False and audited['exact_population_coverage_each_epoch'] is True and
            audited['weights_only_cpu_load'] is True and audited['cuda_initialized'] is False,
            'Full-population SSL artifact audit is missing, changed or incompatible')
    require(len(result['epochs_detail']) == 5, 'SSL five-epoch evidence missing')
    for epoch, item in enumerate(result['epochs_detail'], 1):
        order = np.random.default_rng(1003 + epoch - 1).permutation(np.arange(9000, dtype=np.int64))
        require(item['epoch'] == epoch and item['unique_train_images'] == item['train_images_seen'] == 9000 and
                item['heldout_n'] == 0 and item['target_arrays_loaded'] is False and
                item['order_sha'] == index_sha(order) and item['ordered_ids_sha'] == r.ids_sha(ids[order]) and
                item['checkpoint_sha'] == sha(run/f'epoch{epoch:02d}.pt'), 'SSL epoch coverage/artifact mismatch')
    checkpoint = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
    require(checkpoint['epoch'] == 5 and checkpoint['step'] == 1410 and checkpoint['manifest_sha'] == manifest_sha,
            'SSL checkpoint metadata mismatch')
    # SSL's canonical JSON tensor digest differs from historical prior.state_sha.
    ssl_hash = hashlib.sha256()
    for name, value in sorted(checkpoint['encoder'].items()):
        value = value.detach().cpu().contiguous()
        ssl_hash.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
        ssl_hash.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    require(ssl_hash.hexdigest() == audited['final_encoder_tensor_sha'], 'SSL audit final tensor binding differs')
    return dict(ssl_root=str(root), ssl_manifest_sha=manifest_sha, ssl_result_sha=sha(run/'result.json'),
                ssl_complete_sha=sha(run/'complete.json'), ssl_checkpoint_sha=result['checkpoint_sha'],
                ssl_encoder_tensor_sha=prior.state_sha(checkpoint['encoder']), ssl_epochs=5,
                ssl_source_sha256=args.expected_ssl_source_sha256, ssl_audit_sha=sha(root/'audit.json'),
                ssl_audit_encoder_json_tensor_sha=ssl_hash.hexdigest())


def binding(args):
    return dict(isolated_root=str(args.root), workspace=str(args.workspace),
                selected_recipe_path=str(args.selected_recipe), selected_recipe_sha256=args.expected_recipe_sha256,
                data_audit_path=str(args.data_audit), data_audit_sha256=args.expected_data_audit_sha256,
                expected_reader_sha256=args.expected_reader_sha256, zero_policy=args.zero_policy,
                ssl_root=str(args.ssl_root) if args.ssl_root else None,
                expected_ssl_source_sha256=args.expected_ssl_source_sha256, deadline_utc=args.deadline_utc)


def prepare(args, selection, active):
    r.ensure_before_deadline(args.deadline_utc)
    require(not args.root.exists(), 'Isolated root must not already exist; no overwrite or old run reuse')
    reader, labels, train, excluded, mu, sigma = open_data(args)
    current_raw = data.verify_current_raw_sha(args.workspace)
    init = dict(initialization=selection['initialization'], ssl_epochs=0,
                external_checkpoint_path=str(b.WEIGHTS[ARCH].resolve()), external_checkpoint_sha=sha(b.WEIGHTS[ARCH]))
    require(init['external_checkpoint_sha'] == EXTERNAL_SHA, 'External checkpoint changed')
    if args.ssl_root:
        init.update(ssl_evidence(args, reader.ids))
    net, _ = r.build_cpu(selection['seed'])
    init['external_encoder_tensor_sha'] = prior.state_sha(net[0].original.state_dict())
    head_sha = prior.state_sha(net[1:].state_dict())
    del net
    contract = execution_contract(args, selection, active)
    runtime_sources = inference.verify_runtime_sources()
    sources = source_files()
    args.root.mkdir(parents=True, exist_ok=False)
    require(shutil.disk_usage(args.root).free > 20*2**30, 'Need 20GiB free for epoch model snapshots and evidence')
    (args.root/'source').mkdir()
    for source in sources:
        with source.open('rb') as src, (args.root/'source'/source.name).open('xb') as dst:
            shutil.copyfileobj(src, dst)
    exclusive_json(args.root/'prepare_current_raw.json', current_raw)
    selection_files = [(args.workspace/Path(item['path'])).resolve() for item in selection['selection_evidence']]
    bound_files = [args.selected_recipe, *selection_files, args.data_audit, b.WEIGHTS[ARCH].resolve(),
                   args.root/'prepare_current_raw.json', *sources, *[args.root/'source'/p.name for p in sources]]
    manifest = dict(schema=SCHEMA, training_scope=SCOPE, architecture=ARCH, created_utc=r.utc_now().isoformat(),
                    **binding(args), seed=selection['seed'], population_n=9000, train_n=8998, heldout_n=0,
                    independent_accuracy_validation=False, target_array='Y', Y_train_loaded=False,
                    population_ids_sha=r.ids_sha(reader.ids), train_ids_sha=r.ids_sha(reader.ids[train]),
                    train_indices=train.tolist(), train_order_base_sha=index_sha(train),
                    excluded_indices=excluded.tolist(), excluded_ids=reader.ids[excluded].tolist(),
                    excluded_original_targets=labels['targets'][excluded].tolist(),
                    original_targets_sha=r.array_sha(labels['targets']),
                    train_targets_sha=r.array_sha(labels['targets'][train]), label_provenance=labels['provenance'],
                    log_mean=mu.tolist(), log_std=sigma.tolist(), recipe=recipe_from_selection(selection),
                    selected_recipe=selection, initialization=init, expected_head_sha=head_sha,
                    execution=EXECUTION, required_execution_contract=contract,
                    runtime_sources=runtime_sources,
                    steps_per_epoch=563, total_steps=11260, readback_reference=reference_contract(reader),
                    data_index=reader.provenance['index'], prepare_current_raw_sha256=sha(args.root/'prepare_current_raw.json'),
                    hashes={str(path):sha(path) for path in bound_files},
                    limits=LIMITS, no_exact_resume=True, training_selected_by_this_tool=False)
    exclusive_json(args.root/'manifest.json', manifest)
    print(json.dumps(dict(status='prepared_not_trained', root=str(args.root), manifest_sha=sha(args.root/'manifest.json'))), flush=True)


def verify(args, selection, active):
    m = read(args.root/'manifest.json')
    for key, expected in binding(args).items():
        require(m.get(key) == expected, 'Invocation/manifest mismatch: ' + key)
    require(m['schema'] == SCHEMA and m['training_scope'] == SCOPE and m['architecture'] == ARCH and
            m['seed'] == selection['seed'] and not FORBIDDEN_SCORE_FIELDS.intersection(m) and
            m['selected_recipe'] == selection and m['recipe'] == recipe_from_selection(selection) and
            m['required_execution_contract'] == execution_contract(args, selection, active) and m['execution'] == EXECUTION,
            'Manifest recipe/source/execution contract mismatch')
    require(m['runtime_sources'] == inference.verify_runtime_sources(), 'Installed architecture runtime differs')
    require(m['heldout_n'] == 0 and m['independent_accuracy_validation'] is False and
            m['population_n'] == 9000 and m['train_n'] == 8998 and m['steps_per_epoch'] == 563 and
            m['total_steps'] == 11260 and m['target_array'] == 'Y' and m['Y_train_loaded'] is False,
            'Manifest population/supervision scope mismatch')
    for path, expected in m['hashes'].items():
        require(sha(path) == expected, 'Frozen input/source changed: ' + path)
    for source in source_files():
        require(sha(source) == m['hashes'][str(args.root/'source'/source.name)], 'Active and frozen source differ')
    gate = read(args.root/'prepare_current_raw.json')
    require(gate['current_raw_sha256_verified'] is True and gate['sha256'] == data.RAW_SHA and
            sha(args.root/'prepare_current_raw.json') == m['prepare_current_raw_sha256'], 'Preparation raw proof missing')
    reader, labels, train, excluded, mu, sigma = open_data(args)
    require(m['train_indices'] == train.tolist() and m['excluded_indices'] == excluded.tolist() and
            m['excluded_ids'] == reader.ids[excluded].tolist() and
            m['excluded_original_targets'] == labels['targets'][excluded].tolist() and
            m['population_ids_sha'] == r.ids_sha(reader.ids) and m['train_ids_sha'] == r.ids_sha(reader.ids[train]) and
            m['train_order_base_sha'] == index_sha(train) and m['original_targets_sha'] == r.array_sha(labels['targets']) and
            m['train_targets_sha'] == r.array_sha(labels['targets'][train]) and m['data_index'] == reader.provenance['index'] and
            m['label_provenance'] == labels['provenance'],
            'Current target/index binding mismatch')
    np.testing.assert_array_equal(mu, m['log_mean']); np.testing.assert_array_equal(sigma, m['log_std'])
    require(m['readback_reference'] == reference_contract(reader), 'Fixed reference inputs changed')
    require(m['initialization']['initialization'] == selection['initialization'] and
            m['initialization']['external_checkpoint_sha'] == sha(b.WEIGHTS[ARCH]) == EXTERNAL_SHA,
            'Initialization provenance changed')
    if args.ssl_root:
        for key, value in ssl_evidence(args, reader.ids).items():
            require(m['initialization'][key] == value, 'SSL evidence changed: ' + key)
    else:
        require(m['initialization']['ssl_epochs'] == 0, 'SSL0 must not claim additional SSL')
    return m, reader, labels['targets'], train


class Frames(Dataset):
    def __init__(self, reader, targets, indices, seed, epoch=0, augment=False):
        self.reader, self.targets, self.indices = reader, targets, np.asarray(indices, dtype=np.int64)
        self.seed, self.epoch, self.augment = seed, epoch, augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, row):
        index = int(self.indices[row])
        x = inference.preprocess_native(self.reader.raw_frame(index))
        if self.augment:
            horizontal, vertical = r.flip_decisions(self.seed, self.epoch, index)
            if horizontal:
                x = torch.flip(x, (-1,))
            if vertical:
                x = torch.flip(x, (-2,))
        return x, torch.tensor(self.targets[index], dtype=torch.float32), index


def training_loader(reader, targets, order, seed, epoch):
    return DataLoader(Frames(reader, targets, order, seed, epoch, True), batch_size=16,
                      shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=1, drop_last=False)


def build_initialized_cpu(m):
    # Both branches consume exactly the original external builder/head RNG path.
    net, details = r.build_cpu(m['seed'])
    init = m['initialization']
    require(prior.state_sha(net[1:].state_dict()) == m['expected_head_sha'], 'Initial head mismatch')
    require(prior.state_sha(net[0].original.state_dict()) == init['external_encoder_tensor_sha'], 'External encoder mismatch')
    if init['ssl_epochs'] == 5:
        path = Path(init['ssl_root'])/'run'/'last.pt'
        require(sha(path) == init['ssl_checkpoint_sha'], 'SSL checkpoint changed before load')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        net[0].original.load_state_dict(checkpoint['encoder'], strict=True)
        require(prior.state_sha(net[0].original.state_dict()) == init['ssl_encoder_tensor_sha'], 'Loaded SSL encoder differs')
    require(prior.state_sha(net[1:].state_dict()) == m['expected_head_sha'], 'Encoder load changed the head')
    require(all(p.requires_grad for p in net.parameters()), 'Full encoder/head must train')
    require(not any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in net.modules()),
            'Unexpected batch-dependent normalization')
    return net, details


def setup(args, selection, active, m):
    execution = execution_contract(args, selection, active)
    net, details = build_initialized_cpu(m)
    encoder_sha = prior.state_sha(net[0].original.state_dict())
    net = net.cuda()
    q = m['recipe']
    opt = torch.optim.AdamW([{'params':net[0].parameters(), 'lr':q['encoder_lr']},
                            {'params':list(net[1:].parameters()), 'lr':q['head_lr']}],
                           weight_decay=q['weight_decay'], fused=True)
    mu, sigma = [torch.tensor(m[key], device='cuda', dtype=torch.float32) for key in ('log_mean', 'log_std')]
    details.update(execution_contract=execution, initialization=m['initialization'],
                   initial_encoder_tensor_sha=encoder_sha, head_initial_tensor_sha=m['expected_head_sha'],
                   projection_or_ssl_optimizer_loaded=False, distributed_execution=EXECUTION,
                   mode='full', seed=m['seed'], gpu=torch.cuda.get_device_name(),
                   torch_version=torch.__version__, cuda_version=torch.version.cuda,
                   bitwise_determinism_guaranteed=False, heldout_n=0, independent_accuracy_validation=False)
    return net, opt, mu, sigma, details


def model_free_tests():
    require(not torch.cuda.is_initialized(), 'CPU fixtures must start without CUDA')
    original = loss_module.model_free_tests()
    with patch.object(torch.cuda, '_lazy_init', side_effect=RuntimeError('CPU fixture forbids CUDA')), \
         patch.object(b, 'build', side_effect=RuntimeError('Fixture forbids model construction')):
        ids = np.asarray(['old-d', 'mha-d', 'old-a', 'mha-a', 'old-c', 'mha-c', 'old-b', 'mha-b', 'zero'])
        targets = np.arange(27, dtype=np.float32).reshape(9, 3) / 10 + 1
        targets[-1, 2] = 0
        snapshot = targets.copy()
        train, excluded, mu, sigma = positive_training_rows(ids, targets, 'exclude-nonpositive')
        np.testing.assert_array_equal(train, np.arange(8)); np.testing.assert_array_equal(excluded, [8])
        np.testing.assert_array_equal(targets, snapshot)
        np.testing.assert_array_equal(mu, np.log(snapshot[:8].astype(np.float64)).mean(0))
        np.testing.assert_array_equal(sigma, np.log(snapshot[:8].astype(np.float64)).std(0))
        mapping = np.asarray([0, -1, 1, -1, 2, -1, 3, -1, 4])
        np.testing.assert_array_equal(reference_indices(ids, mapping), [2, 6, 4, 0, 3, 7, 5, 1])
        rejected = 0
        for policy in (None, '', 'mask', 'log1p', 'clamp'):
            try:
                positive_training_rows(ids, targets, policy)
            except ValueError:
                rejected += 1
            else:
                raise AssertionError('Implicit/unsupported zero policy accepted')
        base = dict(schema=SELECTION_SCHEMA, architecture=ARCH, initialization='SSL0_EXTERNAL_PRETRAINED',
                    seed=42, epochs=20, schedule_epochs=20, encoder_lr=1e-5, head_lr=.001,
                    gradient_clip=1., weight_decay=1e-4, standardized_log_l1_coefficient=.5, relative_error_coefficient=2.5,
                    selection_evidence=[dict(path='synthetic/no-file-access.json', sha256='0'*64)])
        require(validate_selection(base) == base, 'Explicit recipe fixture rejected')
        for key, value in [('epochs', 19), ('seed', True), ('initialization', 'SSL5_FOLD4'),
                           ('standardized_log_l1_coefficient', .5001), ('relative_error_coefficient', 2.),
                           ('head_lr', float('nan')), ('architecture', 'other'), ('selection_evidence', []),
                           ('selection_evidence', [dict(path='unknown.json', sha256='bad')])]:
            try:
                validate_selection(dict(base, **{key:value}))
            except ValueError:
                rejected += 1
            else:
                raise AssertionError('Invalid selected recipe accepted')
        cases = []
        for alpha in (0., .25, .5):
            fn = partial(loss_module.loss_formula, standardized_log_l1_coefficient=alpha)
            for n in (6, 8, 9, 16):
                x = torch.arange(n*5, dtype=torch.float64).reshape(n, 5)/100
                t = torch.exp(torch.tensor([2., -.8, -.5]) + torch.arange(n*3).reshape(n, 3)/100.)
                mean, std = torch.tensor([2., -.8, -.5]), torch.tensor([.1, .2, .3])
                ref = torch.nn.Linear(5, 3, dtype=torch.float64)
                fn(ref(x), t, mean, std).backward()
                accumulated = [torch.zeros_like(p) for p in ref.parameters()]
                counts, coverage = [], []
                for rank in (0, 1):
                    local = torch.nn.Linear(5, 3, dtype=torch.float64); local.load_state_dict(ref.state_dict())
                    for j in range(2):
                        start = (2*j+rank)*4; count = max(0, min(4, n-start)); counts.append(count)
                        coverage.extend(range(start, start+count))
                        xx, yy = (x[start:start+count], t[start:start+count]) if count else (x[:1], t[:1])
                        (fn(local(xx), yy, mean, std)*2*count/n).backward()
                    for actual, parameter in zip(accumulated, local.parameters()):
                        actual += parameter.grad/2
                for actual, parameter in zip(accumulated, ref.parameters()):
                    torch.testing.assert_close(actual, parameter.grad, rtol=1e-6, atol=1e-7)
                require(sorted(coverage) == list(range(n)), 'Partial-batch coverage mismatch')
                if n == 6:
                    require(counts == [4, 0, 2, 0], 'Actual6 rank-major microbatch layout mismatch')
                cases.append(dict(alpha=alpha, n=n, rank_major_counts=counts))
        require(math.ceil(8998/16) == 563 and 8998 % 16 == 6, 'Full-population update count mismatch')
        for schedule in (2, 20, 30):
            require(r.lr_ratio(0, 563, schedule) == 1/563 and
                    abs(r.lr_ratio(563*schedule-1, 563, schedule)-.1) < 1e-12, 'Schedule endpoints differ')
    require(not torch.cuda.is_initialized(), 'Fixture initialized CUDA')
    return dict(status='passed', foundation_model_constructed=False, toy_linear_modules_used=True,
                cuda_initialized=False, production_data_used=False,
                original_loss_tests=original, positive_train_stats_and_original_Y_preserved=True,
                reference_selection_uses_only_ID_and_pixel_source=True, rejected_cases=rejected,
                actual6_and_other_partial_ddp_cases=cases, partial_ddp_tolerance=dict(rtol=1e-6, atol=1e-7))


def cpu_tests(args, selection, active):
    require(not (args.root/'cpu_tests.json').exists(), 'CPU evidence already exists')
    m, reader, targets, train = verify(args, selection, active)
    fixture = model_free_tests()
    with patch.object(torch.cuda, '_lazy_init', side_effect=RuntimeError('CPU tests forbid CUDA')):
        net, _ = build_initialized_cpu(m)
        require(all(p.device.type == 'cpu' for p in net.parameters()), 'CPU build left CPU')
        require(len(r.block_hashes(net)) == 12, 'Expected12 encoder blocks')
        # The head must depend on the chosen seed, while SSL replacement leaves it unchanged.
        head = prior.state_sha(net[1:].state_dict()); del net
        again, _ = build_initialized_cpu(m)
        require(prior.state_sha(again[1:].state_dict()) == head, 'Same seed head differs'); del again
        other, _ = r.build_cpu(m['seed'] + 1)
        require(prior.state_sha(other[1:].state_dict()) != head, 'Head seed is not effective'); del other
        shared = [record for record in m['readback_reference']['records'] if record['source'] == 'native8954_memmap']
        old_indices = [int(reader.mapping[record['index']]) for record in shared]
        original = b.Frames(old_indices, np.ones((8954, 3), dtype=np.float32), 8954)
        for row, record in enumerate(shared):
            torch.testing.assert_close(inference.preprocess_native(reader.raw_frame(record['index'])), original[row][0], rtol=0, atol=0)
        a = np.asarray([r.flip_decisions(m['seed'], 3, int(i)) for i in train[:128]])
        require(not np.array_equal(a, np.asarray([r.flip_decisions(m['seed']+1, 3, int(i)) for i in train[:128]])),
                'Augmentation seed has no effect')
    exclusive_json(args.root/'cpu_tests.json', dict(status='passed', manifest_sha=sha(args.root/'manifest.json'),
                   execution_contract=execution_contract(args, selection, active), model_free_tests=fixture,
                   actual_head_seed_verified=True, initialization_verified=True, preprocessing_shared_bitwise=True,
                   heldout_n=0, independent_accuracy_validation=False, cuda_initialized=torch.cuda.is_initialized()))


def require_stage(args, m, names):
    for name in names:
        value = read(args.root/(name+'.json'))
        require(value['status'] == 'passed' and value['manifest_sha'] == sha(args.root/'manifest.json') and
                value['execution_contract'] == m['required_execution_contract'], 'Required execution evidence differs: ' + name)


def live_data_gate(args, reader, rank, path):
    # Both ranks just opened a reader, which revalidated all46 files/pixels/header.
    evidence = [None]
    if rank == 0:
        evidence[0] = dict(current_raw=data.verify_current_raw_sha(args.workspace),
                           extra46_current_verified=True, data_audit_sha256=args.expected_data_audit_sha256,
                           reader_source_sha256=args.expected_reader_sha256, population_ids_sha=r.ids_sha(reader.ids),
                           checked_utc=r.utc_now().isoformat())
        exclusive_json(path, evidence[0])
    dist.broadcast_object_list(evidence, src=0)
    require(evidence[0]['current_raw']['current_raw_sha256_verified'], 'Current raw launch gate failed')
    return evidence[0]


def update_deadline_hit(m, rank):
    # A future launcher must also enforce the absolute wall-clock cutoff.
    # Reserve time for an in-flight GPU operation; do not start a new update
    # in the last90 seconds, and recheck immediately before optimizer.step.
    cutoff = r.parse_deadline(m['deadline_utc']) - r.datetime.timedelta(seconds=90)
    return r.deadline_hit(dict(deadline_utc=cutoff.isoformat()), rank)


def smoke(args, selection, active):
    m, reader, targets, train = verify(args, selection, active)
    require_stage(args, m, ['cpu_tests']); r.ensure_before_deadline(m['deadline_utc'])
    require(not (args.root/'smoke.json').exists(), 'Smoke evidence already exists')
    rank = r.init_dist()
    live_data_gate(args, reader, rank, args.root/'smoke_current_data.json')
    net, opt, mu, sigma, details = setup(args, selection, active, m)
    ddp = DDP(net, device_ids=[rank], broadcast_buffers=False, find_unused_parameters=True)
    # Exercise both storage backends in both the actual6 and dummy9 smoke cases.
    shared = int(train[reader.mapping[train] >= 0][0])
    extra = int(train[reader.mapping[train] < 0][0])
    smoke_indices = np.asarray([shared, extra] + [int(i) for i in
        np.random.default_rng(m['seed']).permutation(train) if i not in (shared, extra)][:7], dtype=np.int64)
    ds = Frames(reader, targets, smoke_indices, m['seed'])
    rows = [ds[i] for i in range(9)]; x = torch.stack([v[0] for v in rows]); t = torch.stack([v[1] for v in rows])
    evidence = []
    for n in (6, 9):
        require(not update_deadline_hit(m, rank), 'Deadline before GPU smoke')
        ddp.eval(); opt.zero_grad(set_to_none=True)
        r.distributed_backward(ddp, x[:n], t[:n], mu, sigma, rank)
        check = [None]
        if rank == 0:
            ref, _, rmu, rsigma, _ = setup(args, selection, active, m); ref.eval(); ref.zero_grad(set_to_none=True)
            for start in range(0, n, 4):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z = ref(b.inputs(x[start:min(start+4,n)], ARCH))
                    loss = active(z, t[start:min(start+4,n)].cuda(), rmu, rsigma)
                (loss*min(4,n-start)/n).backward()
            error, norm, maximum = 0., 0., 0.
            for (name, p), (other, q) in zip(net.named_parameters(), ref.named_parameters()):
                require(name == other and (p.grad is None) == (q.grad is None), 'Gradient parameter mismatch')
                if p.grad is not None:
                    delta = (p.grad-q.grad).float(); error += delta.square().sum().item()
                    norm += q.grad.float().square().sum().item(); maximum = max(maximum, delta.abs().max().item())
            relative = math.sqrt(error/max(norm,1e-30)); del ref
            check[0] = dict(n=n, relative_gradient_l2=relative, max_abs_gradient_error=maximum, passed=relative < .002)
        dist.broadcast_object_list(check, src=0); require(check[0]['passed'], 'GPU partial-batch reference gradient differs')
        evidence.append(check[0])
    before = r.block_hashes(net); require(len(before) == 12, 'Encoder block count differs')
    reached = {r.block_key(name) for name, p in net[0].original.named_parameters()
               if r.block_key(name) and p.grad is not None and p.grad.abs().max().item() > 0}
    require(set(before) == reached, 'All encoder blocks must receive gradients')
    require(any('patch_embed' in name and p.grad is not None and p.grad.abs().max().item() > 0
                for name, p in net[0].original.named_parameters()), 'Patch embed gradient missing')
    head_before = prior.state_sha(net[1:].state_dict())
    b.finite_grad(net); torch.nn.utils.clip_grad_norm_(net.parameters(), m['recipe']['gradient_clip'], error_if_nonfinite=True)
    require(not update_deadline_hit(m, rank), 'Deadline before smoke update'); opt.step()
    after = r.block_hashes(net)
    require(all(before[key] != after[key] for key in before) and prior.state_sha(net[1:].state_dict()) != head_before,
            'Encoder blocks/head did not actually update')
    ddp.train(); opt.zero_grad(set_to_none=True)
    require(not update_deadline_hit(m, rank), 'Deadline before train-mode smoke')
    r.distributed_backward(ddp, x[:6], t[:6], mu, sigma, rank)
    b.finite_grad(net); torch.nn.utils.clip_grad_norm_(net.parameters(), m['recipe']['gradient_clip'], error_if_nonfinite=True)
    require(not update_deadline_hit(m, rank), 'Deadline before train-mode smoke update'); opt.step()
    hashes = [None, None]; dist.all_gather_object(hashes, prior.state_sha(net.state_dict()))
    require(hashes[0] == hashes[1], 'DDP states differ')
    if rank == 0:
        exclusive_json(args.root/'smoke.json', dict(status='passed', manifest_sha=sha(args.root/'manifest.json'),
                       execution_contract=details['execution_contract'], details=details,
                       partial_ddp_reference=evidence, actual_final6_verified=True, all_12_encoder_blocks_updated=True,
                       smoke_indices=smoke_indices.tolist(), both_pixel_sources_in_each_partial_batch=True,
                       patch_gradient_verified=True, head_updated=True, ranks_synchronized=True,
                       current_data_gate_sha=sha(args.root/'smoke_current_data.json')))
    dist.barrier(); dist.destroy_process_group()


def checkpoint_record(net, m, manifest_sha, epoch, updates):
    return dict(model=net.state_dict(), epoch=epoch, updates=updates, architecture=ARCH,
                manifest_sha=manifest_sha, log_mean=m['log_mean'], log_std=m['log_std'],
                training_scope=SCOPE, heldout_n=0, independent_accuracy_validation=False,
                execution_contract=m['required_execution_contract'], note='Model only; terminal20; no resume')


def train(args, selection, active):
    m, reader, targets, train_indices = verify(args, selection, active)
    require_stage(args, m, ['cpu_tests', 'smoke']); r.ensure_before_deadline(m['deadline_utc'])
    run = args.root/'run'; require(not run.exists(), 'No overwrite or implicit resume')
    rank = r.init_dist()
    if rank == 0:
        run.mkdir(exist_ok=False)
    dist.barrier()
    live_data_gate(args, reader, rank, run/'current_data_gate.json')
    net, opt, mu, sigma, details = setup(args, selection, active, m)
    ddp = DDP(net, device_ids=[rank], broadcast_buffers=False, find_unused_parameters=True)
    manifest_sha = sha(args.root/'manifest.json'); history = []; step = 0; started = time.monotonic()
    initial = prior.state_sha(net[0].original.state_dict())
    for epoch in range(20):
        order = np.random.default_rng(m['seed']+epoch).permutation(train_indices)
        loader = training_loader(reader, targets, order, m['seed'], epoch)
        require(len(loader) == 563, 'Unexpected epoch batches')
        ddp.train(); torch.cuda.reset_peak_memory_stats(); tick = time.monotonic(); seen = []
        totals = torch.zeros(7, device=mu.device, dtype=torch.float64)
        norm_sum = torch.zeros((), device=mu.device); norm_max = torch.zeros_like(norm_sum); clipped = torch.zeros_like(norm_sum)
        stopped = False
        for x, t, ix in loader:
            if update_deadline_hit(m, rank):
                stopped = True; break
            ratio = r.lr_ratio(step, 563, m['recipe']['schedule_epochs'])
            opt.param_groups[0]['lr'] = m['recipe']['encoder_lr']*ratio
            opt.param_groups[1]['lr'] = m['recipe']['head_lr']*ratio
            opt.zero_grad(set_to_none=True)
            totals += r.distributed_backward(ddp, x, t, mu, sigma, rank)
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), m['recipe']['gradient_clip'], error_if_nonfinite=True)
            if update_deadline_hit(m, rank):
                stopped = True; break
            opt.step(); step += 1; seen.extend(ix.tolist())
            norm_sum += norm.detach(); norm_max = torch.maximum(norm_max, norm.detach())
            clipped += (norm.detach() > m['recipe']['gradient_clip']).float()
            if rank == 0 and (step == 1 or step % 25 == 0):
                progress = dict(status='training', epoch=epoch+1, updates=step, train_n=8998, seen=len(seen),
                                heldout_n=0, independent_accuracy_validation=False,
                                seconds_this_epoch=time.monotonic()-tick, gradient_norm=norm.item())
                atomic_json(run/'progress.json', progress); print(json.dumps(progress), flush=True)
        if stopped:
            if rank == 0:
                value = dict(status='deadline_stopped_incomplete', completed_epochs=epoch, updates=step,
                             partial_epoch_seen=len(seen), manifest_sha=manifest_sha,
                             last_complete_checkpoint=f'epoch_{epoch:02d}.pt' if epoch else None,
                             stopped_utc=r.utc_now().isoformat(), no_resume=True)
                exclusive_json(run/'stopped.json', value); atomic_json(run/'progress.json', value)
            dist.barrier(); dist.destroy_process_group(); return
        np.testing.assert_array_equal(seen, order)
        require(step == (epoch+1)*563 and len(set(seen)) == 8998, 'Exact per-epoch coverage/update mismatch')
        rank_orders = [None, None]; dist.all_gather_object(rank_orders, index_sha(seen))
        require(rank_orders[0] == rank_orders[1], 'Ranks did not use the same full global batch order')
        dist.all_reduce(totals)
        require(prior.state_sha(net[0].original.state_dict()) != initial, 'Encoder did not change')
        if rank == 0:
            values = (totals/8998).cpu().tolist()
            require(np.isfinite(values).all(), 'Nonfinite online training metrics')
            weights = run/f'epoch_{epoch+1:02d}.pt'
            exclusive_checkpoint(weights, checkpoint_record(net, m, manifest_sha, epoch+1, step))
            item = dict(epoch=epoch+1, updates=step, unique_train_images=8998, train_images_seen=8998,
                        order_sha=index_sha(order), ordered_ids_sha=r.ids_sha(reader.ids[order]),
                        online_train_loss=values[0], online_train_mape=dict(zip(('CD','CV','HEX'), values[1:4])),
                        online_standardized_log_l1=dict(zip(('CD','CV','HEX'), values[4:7])),
                        gradient_norm_mean=(norm_sum/563).item(), gradient_norm_max=norm_max.item(),
                        gradient_clip_fraction=(clipped/563).item(), gradient_clip_threshold=m['recipe']['gradient_clip'],
                        encoder_lr=opt.param_groups[0]['lr'], head_lr=opt.param_groups[1]['lr'],
                        seconds=time.monotonic()-tick, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        weights=weights.name, weights_sha=sha(weights), model_tensor_sha=prior.state_sha(net.state_dict()),
                        heldout_n=0, independent_accuracy_validation=False)
            history.append(item); atomic_json(run/'history.json', history)
            atomic_json(run/'progress.json', dict(status='epoch_complete', **item))
            print(json.dumps(dict(epoch_complete=epoch+1, **item)), flush=True)
        dist.barrier()
    hashes = [None, None]; dist.all_gather_object(hashes, prior.state_sha(net.state_dict()))
    require(hashes[0] == hashes[1], 'Terminal DDP states differ')
    if rank == 0:
        # No labels enter this reference prediction artifact or calculation.
        records = m['readback_reference']['records']; indices = np.asarray([v['index'] for v in records], dtype=np.int64)
        require(r.utc_now() < r.parse_deadline(m['deadline_utc']), 'Deadline before terminal reference readback')
        predictions = inference.predict_images(net, [reader.raw_frame(int(i)) for i in indices], mu, sigma,
                                               device='cuda:0', precision='bf16', batch_size=4)
        reference_path = run/'readback_reference.npz'
        with reference_path.open('xb') as stream:
            np.savez_compressed(stream, indices=indices, ids=reader.ids[indices], pred=predictions,
                                raw_pixel_sha256=np.asarray([v['raw_pixel_sha256'] for v in records]))
        result = dict(status=RESULT_STATUS, schema=SCHEMA, architecture=ARCH, training_scope=SCOPE,
                      seed=m['seed'], epochs=20, updates=step, population_n=9000, train_n=8998,
                      train_image_presentations=179960, heldout_n=0, independent_accuracy_validation=False,
                      single_model=True, terminal_checkpoint='epoch_20.pt', checkpoint_sha=sha(run/'epoch_20.pt'),
                      model_tensor_sha=hashes[0], reference_predictions_sha=sha(reference_path),
                      reference_prediction_role='fixed full-population inputs; reproduction only; no target or MAPE',
                      manifest_sha=manifest_sha, history=history, history_sha=sha(run/'history.json'), details=details,
                      current_data_gate_sha=sha(run/'current_data_gate.json'), ranks_synchronized=True,
                      cpu_tests_sha=sha(args.root/'cpu_tests.json'), smoke_sha=sha(args.root/'smoke.json'),
                      seconds=time.monotonic()-started, limits=LIMITS, finished_utc=r.utc_now().isoformat())
        exclusive_json(run/'result.json', result)
        exclusive_json(run/'complete.json', dict(status='completed', result_sha=sha(run/'result.json'),
                       checkpoint_sha=result['checkpoint_sha'], manifest_sha=manifest_sha,
                       reference_predictions_sha=result['reference_predictions_sha']))
        atomic_json(run/'progress.json', dict(status='completed', epoch=20, updates=step, heldout_n=0,
                                             independent_accuracy_validation=False))
    dist.barrier(); dist.destroy_process_group()


def audit(args, selection, active):
    require(not torch.cuda.is_initialized(), 'Artifact audit is CPU only')
    require(not (args.root/'audit.json').exists(), 'Audit already exists')
    m, reader, _, train_indices = verify(args, selection, active)
    require_stage(args, m, ['cpu_tests', 'smoke'])
    run = args.root/'run'; result, complete = read(run/'result.json'), read(run/'complete.json')
    manifest_sha = sha(args.root/'manifest.json')
    require(result['status'] == RESULT_STATUS and result['manifest_sha'] == manifest_sha and
            result['schema'] == SCHEMA and result['architecture'] == ARCH and result['training_scope'] == SCOPE and
            result['seed'] == m['seed'] and result['population_n'] == 9000 and result['train_n'] == 8998 and
            result['terminal_checkpoint'] == 'epoch_20.pt' and not FORBIDDEN_SCORE_FIELDS.intersection(result) and
            result['details']['execution_contract'] == m['required_execution_contract'] and
            result['details']['initialization'] == m['initialization'] and
            result['epochs'] == 20 and result['updates'] == 11260 and result['train_image_presentations'] == 179960 and
            result['heldout_n'] == 0 and result['independent_accuracy_validation'] is False and result['single_model'] is True and
            result['ranks_synchronized'] is True, 'Completed run contract differs')
    require(complete == dict(status='completed', result_sha=sha(run/'result.json'), manifest_sha=manifest_sha,
                             checkpoint_sha=sha(run/'epoch_20.pt'), reference_predictions_sha=sha(run/'readback_reference.npz')),
            'Completion/hash chain differs')
    require(result['checkpoint_sha'] == complete['checkpoint_sha'] and
            result['reference_predictions_sha'] == complete['reference_predictions_sha'] and
            result['history_sha'] == sha(run/'history.json') and result['history'] == read(run/'history.json') and
            result['cpu_tests_sha'] == sha(args.root/'cpu_tests.json') and result['smoke_sha'] == sha(args.root/'smoke.json'),
            'Result evidence binding differs')
    gate = read(run/'current_data_gate.json')
    require(result['current_data_gate_sha'] == sha(run/'current_data_gate.json') and
            gate['current_raw']['sha256'] == data.RAW_SHA and gate['current_raw']['current_raw_sha256_verified'] is True and
            gate['extra46_current_verified'] is True and gate['data_audit_sha256'] == args.expected_data_audit_sha256,
            'Actual training current-data proof missing')
    require(len(result['history']) == 20, 'Incomplete epoch history')
    with patch.object(torch.cuda, '_lazy_init', side_effect=RuntimeError('Audit forbids CUDA')):
        model = inference.build_model()
        for epoch, item in enumerate(result['history'], 1):
            order = np.random.default_rng(m['seed']+epoch-1).permutation(train_indices)
            require(item['epoch'] == epoch and item['updates'] == epoch*563 and
                    item['unique_train_images'] == item['train_images_seen'] == 8998 and
                    item['order_sha'] == index_sha(order) and item['ordered_ids_sha'] == r.ids_sha(reader.ids[order]) and
                    item['weights'] == f'epoch_{epoch:02d}.pt' and item['heldout_n'] == 0 and
                    item['independent_accuracy_validation'] is False and not FORBIDDEN_SCORE_FIELDS.intersection(item),
                    'Epoch coverage/metadata differs')
            ratio = r.lr_ratio(epoch*563-1, 563, m['recipe']['schedule_epochs'])
            require(item['encoder_lr'] == m['recipe']['encoder_lr']*ratio and
                    item['head_lr'] == m['recipe']['head_lr']*ratio and
                    item['gradient_clip_threshold'] == m['recipe']['gradient_clip'] and
                    0 <= item['gradient_clip_fraction'] <= 1, 'Epoch optimizer schedule/clip differs')
            values = [item['online_train_loss'], item['gradient_norm_mean'], item['gradient_norm_max'],
                      *item['online_train_mape'].values(), *item['online_standardized_log_l1'].values()]
            require(np.isfinite(values).all() and (np.asarray(values) >= 0).all(), 'Invalid online training metrics')
            path = run/item['weights']; require(sha(path) == item['weights_sha'], 'Epoch weights changed')
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            require(checkpoint['architecture'] == ARCH and checkpoint['epoch'] == epoch and
                    checkpoint['updates'] == epoch*563 and checkpoint['manifest_sha'] == manifest_sha and
                    checkpoint['execution_contract'] == m['required_execution_contract'] and
                    checkpoint['training_scope'] == SCOPE and checkpoint['heldout_n'] == 0 and
                    checkpoint['independent_accuracy_validation'] is False, 'Epoch checkpoint metadata differs')
            np.testing.assert_array_equal(checkpoint['log_mean'], m['log_mean'])
            np.testing.assert_array_equal(checkpoint['log_std'], m['log_std'])
            state = checkpoint['model']; template = model.state_dict()
            require(state.keys() == template.keys() and all(v.dtype == template[k].dtype and
                    v.shape == template[k].shape and torch.isfinite(v).all().item() for k,v in state.items()),
                    'Checkpoint state key/shape/dtype/finite mismatch')
            model.load_state_dict(state, strict=True)
            require(prior.state_sha(state) == item['model_tensor_sha'], 'Epoch tensor SHA differs')
            require('optimizer' not in checkpoint, 'Epoch snapshots must be model only')
            del checkpoint, state
        require(prior.state_sha(model.state_dict()) == result['model_tensor_sha'], 'Terminal state differs')
    records = m['readback_reference']['records']; indices = np.asarray([v['index'] for v in records], dtype=np.int64)
    with np.load(run/'readback_reference.npz', allow_pickle=False) as reference:
        require(set(reference.files) == {'indices','ids','pred','raw_pixel_sha256'}, 'Reference must contain no targets/metrics')
        np.testing.assert_array_equal(reference['indices'], indices)
        np.testing.assert_array_equal(reference['ids'], reader.ids[indices])
        np.testing.assert_array_equal(reference['raw_pixel_sha256'], [v['raw_pixel_sha256'] for v in records])
        require(reference['pred'].shape == (8,3) and reference['pred'].dtype == np.float32 and
                np.isfinite(reference['pred']).all() and (reference['pred'] > 0).all(), 'Invalid reference predictions')
    require(not torch.cuda.is_initialized(), 'Audit initialized CUDA')
    report = dict(status=AUDIT_STATUS, manifest_sha=manifest_sha, execution_contract=m['required_execution_contract'],
                  result_sha=sha(run/'result.json'), complete_sha=sha(run/'complete.json'),
                  checkpoint_sha=sha(run/'epoch_20.pt'), reference_predictions_sha=sha(run/'readback_reference.npz'),
                  cpu_tests_sha=sha(args.root/'cpu_tests.json'), smoke_sha=sha(args.root/'smoke.json'),
                  current_data_gate_sha=sha(run/'current_data_gate.json'), epoch_count=20,
                  exact_training_coverage_verified=True, all_epoch_state_metadata_finite_verified=True,
                  terminal_strict_CPU_load_verified=True, terminal_forward_readback_verified=False,
                  dedicated_CPU_GPU_readback_required=True, heldout_n=0, independent_accuracy_validation=False,
                  cuda_initialized=False, limits=LIMITS)
    exclusive_json(args.root/'audit.json', report); print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['model-free-tests','prepare','cpu_tests','smoke','train','audit'])
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--selected-recipe', type=Path)
    parser.add_argument('--expected-recipe-sha256')
    parser.add_argument('--data-audit', type=Path)
    parser.add_argument('--expected-data-audit-sha256')
    parser.add_argument('--expected-reader-sha256')
    parser.add_argument('--zero-policy', choices=['exclude-nonpositive'])
    parser.add_argument('--ssl-root', type=Path)
    parser.add_argument('--expected-ssl-source-sha256')
    parser.add_argument('--deadline-utc')
    args = parser.parse_args()
    if args.command == 'model-free-tests':
        require(all(value is None for key,value in vars(args).items() if key != 'command'),
                'Synthetic fixtures accept no experiment/data options')
        print(json.dumps(model_free_tests(), indent=2, allow_nan=False)); return
    required = ('workspace','root','selected_recipe','expected_recipe_sha256','data_audit',
                'expected_data_audit_sha256','expected_reader_sha256','zero_policy','deadline_utc')
    for key in required:
        require(getattr(args,key) is not None, 'Explicit required argument: --' + key.replace('_','-'))
    args.workspace = args.workspace.resolve()
    for key in ('root','selected_recipe','data_audit','ssl_root'):
        value = getattr(args,key)
        if value is not None:
            setattr(args,key,(args.workspace/value).resolve() if not value.is_absolute() else value.resolve())
    require(args.root != args.workspace and args.workspace in args.root.parents, 'Use a fresh isolated root inside workspace')
    require(r.parse_deadline(args.deadline_utc) <= r.parse_deadline(MAX_DEADLINE), 'Deadline exceeds authorized cutoff')
    os.chdir(args.workspace)
    selection = selected_recipe(args); active = activate_loss(args, selection)
    globals()[args.command](args, selection, active)


if __name__ == '__main__':
    main()
