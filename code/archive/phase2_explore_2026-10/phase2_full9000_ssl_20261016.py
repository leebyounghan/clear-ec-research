#!/usr/bin/env python3
"""Conditional full-population SSL preparation; this file does not select a fit.

Fresh external CPDINO, 9,000 native images, no target arrays or heldout split.
The pinned original modules own model construction, loss, cache backward, RNG,
and checkpoint serialization. This module owns the explicit no-heldout data and
completion contract. Run each command in a separate process. No resume option.
fixture-tests constructs no model and reads no research dataset or checkpoint.
"""
import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import sys
import tempfile
import time

import numpy as np

PINNED = {
    'phase2_crossfold_ssl_20261015.py': '21d1afe8905fba16cc69e10bbc0ba5e8f2e7839345cadad69ee0151dab546f07',
    'phase2_lejepa_20261003.py': '34ce7a35a6b4fa72fa6deb07ba5b4523bb31c21616b0b98c9ea88be6b6ddc88f',
    'phase2_foundation_preflight_20261002.py': '196500b67d01ad9423ea4d7d8c1a1e4b2d886df2328d36bfb1e1c94058ca4b1e',
}
READER_NAME = 'phase2_full9000_data_20261016.py'
WEIGHT = Path('experiments/phase2_foundation_20261002/weights/cellpose/cpdino-vitb')
EXTERNAL_SHA = '3ed4c06a3963ab13ff377d4e2957174aaaf637434eb031ae9931fcbfaf9a217f'
RAW = Path('datasets/clear_ec_fullframe_20260904/img_full_raw.dat')
RAW_SHA = '93e67ac6df5887a1a5396c85e236b8398297dd3e728cdb6513238c1628c1c88a'
RAW_BYTES = 8954 * 972 * 1296
SCOPE = 'full9000_label_free'
SCHEMA = 'full9000_ssl_v1'
RESULT_STATUS = 'SSL_FULL_POPULATION_COMPLETE_NOT_ACCURACY_VALIDATED'
N, EPOCHS, SEED, BATCH, STEPS = 9000, 5, 1003, 32, 282
MAX_DEADLINE = '2026-10-19T03:59:00Z'
FORBIDDEN_SPLIT_FIELDS = {'fold', 'split_path', 'split_sha256', 'validation_ids_sha',
                          'validation_images_used', 'validation_n', 'val_n'}
LIMITATIONS = [
    'No independent heldout accuracy measurement exists for this full-population fit.',
    'All 9000 images are included without consulting regression targets or choosing a zero-target policy.',
    'Patient independence and near-duplicate leakage remain unverified.',
    'Full-frame two-view adaptation, not original multi-crop LeJEPA.',
    'A final selected regression model requires separate strict reload and forward readback.',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exclusive_json(path, value):
    """Only immutable evidence uses this writer; never replace an existing file."""
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x') as stream:
        stream.write(payload)


def progress_json(path, value):
    """Replace only this run's mutable progress/history, after exclusive run creation."""
    temporary = Path(path).with_suffix('.tmp')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def ids_sha(ids):
    return hashlib.sha256('\n'.join(np.asarray(ids).astype(str)).encode()).hexdigest()


def index_sha(indices):
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def validate_population(ids):
    ids = np.asarray(ids)
    require(ids.ndim == 1 and len(ids) == N and ids.dtype.kind in 'US',
            'Population must contain exactly 9000 string IDs in metadata order')
    require(len(np.unique(ids)) == N and all(str(i).strip() for i in ids),
            'Population IDs must be unique and nonempty')
    return np.arange(N, dtype=np.int64)


def validate_scope(m):
    require(not FORBIDDEN_SPLIT_FIELDS.intersection(m), 'Fold/validation fields cannot describe a full fit')
    expected = dict(schema=SCHEMA, training_scope=SCOPE, population_n=N, heldout_n=0,
                    independent_accuracy_validation=False, target_arrays_loaded=False,
                    seed=SEED, epochs=EPOCHS, steps_per_epoch=STEPS,
                    total_steps=EPOCHS * STEPS, last_batch_images=8)
    for key, value in expected.items():
        require(type(m.get(key)) is type(value) and m[key] == value, f'Invalid population contract: {key}')


def deadline_contract(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    maximum = datetime.fromisoformat(MAX_DEADLINE.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0
            and parsed <= maximum, 'Deadline must be UTC and no later than the authorized cutoff')


def source_paths(args):
    directory = Path(__file__).resolve().parent
    expected = {**PINNED, READER_NAME: args.expected_reader_sha256}
    paths = {name: directory / name for name in expected}
    for name, path in paths.items():
        require(sha(path) == expected[name], f'Unreviewed source: {path}')
    paths[Path(__file__).name] = Path(__file__).resolve()
    return paths


def import_checked(name, expected):
    directory = Path(__file__).resolve().parent
    require(sha(directory / (name + '.py')) == expected, f'Import pin differs: {name}')
    # Pinned siblings must precede incidental project copies on sys.path.
    sys.path.insert(0, str(directory))
    module = importlib.import_module(name)
    require(Path(module.__file__).resolve() == directory / (name + '.py')
            and sha(module.__file__) == expected, f'Executed module differs: {name}')
    return module


def runtime(args):
    source_paths(args)  # Check all sibling bytes before importing any reused code.
    base = import_checked('phase2_crossfold_ssl_20261015', PINNED['phase2_crossfold_ssl_20261015.py'])
    for name, expected in PINNED.items():
        module = sys.modules[name[:-3]]
        require(sha(module.__file__) == expected, f'Reused runtime differs: {name}')
    return base, base.pilot, base.torch


def reader_module(args):
    source_paths(args)
    return import_checked(READER_NAME[:-3], args.expected_reader_sha256)


@contextmanager
def ids_only_npz():
    """An SSL reader may inspect NPZ IDs; even accidental label access fails."""
    cls = np.lib.npyio.NpzFile
    original = cls.__getitem__
    def only_ids(self, key):
        require(key == 'ids', f'SSL NPZ access outside image IDs refused: {key}')
        return original(self, key)
    cls.__getitem__ = only_ids
    try:
        yield
    finally:
        cls.__getitem__ = original


def make_reader(args):
    module = reader_module(args)
    with ids_only_npz():
        reader = module.Full9000Reader(args.workspace, args.data_audit,
                                      expected_audit_sha256=args.expected_data_audit_sha256)
    validate_population(np.asarray(reader.ids))
    require(len(reader) == N, 'Reader length differs')
    return reader


def get_frame(reader, index):
    require(type(index) is int and 0 <= index < N, 'Invalid population image index')
    with ids_only_npz():
        frame = np.asarray(reader.raw_frame(index))
    require(frame.dtype == np.uint8 and frame.shape == (972, 1296),
            'SSL requires original uint8 native972x1296 frames')
    return frame


def recipe(base):
    value = copy.deepcopy(base.pilot.RECIPE)
    value.update(seed=SEED, epochs=EPOCHS, status='FRESH_FULL9000_LABEL_FREE_SSL',
                 order_seed_rule='seed + zero_based_epoch',
                 augmentation_seed_rule='SeedSequence([seed,zero_based_epoch,image_index,view])',
                 lr_policy='linear global20step warmup, then fixed LR through all epochs',
                 initialization='external CPDINO only; fresh optimizer/projector/RNG')
    return value


def freeze_sources(args):
    directory = args.root / 'source'
    directory.mkdir()
    frozen = {}
    for name, path in source_paths(args).items():
        destination = directory / name
        with path.open('rb') as src, destination.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        require(sha(destination) == sha(path), 'Source copy differs')
        frozen[str(destination)] = sha(destination)
    return frozen


def raw_report_contract(report, workspace):
    require(report.get('current_raw_sha256_verified') is True
            and report.get('sha256') == RAW_SHA and report.get('bytes') == RAW_BYTES
            and (workspace / report['path']).resolve() == (workspace / RAW).resolve(),
            'A successful explicit current raw SHA report is required')
    timestamp = datetime.fromisoformat(report['checked_utc'].replace('Z', '+00:00'))
    require(timestamp.tzinfo is not None and timestamp.timestamp() <= time.time() + 5,
            'Raw verification report timestamp is invalid')


def data_evidence(args, *, rehash_raw):
    require(sha(args.data_audit) == args.expected_data_audit_sha256, 'Data audit pin differs')
    require(sha(args.current_raw_verification) == args.expected_current_raw_verification_sha256,
            'Current raw verification report pin differs')
    audit, report = read(args.data_audit), read(args.current_raw_verification)
    raw_report_contract(report, args.workspace)
    require(audit['schema'] == 'clear_ec_full9000_data_v1'
            and audit['status'] == 'preparation_checks_passed'
            and audit['source_sha256'] == args.expected_reader_sha256,
            'Data audit and executed reader source differ')
    require(audit['index']['count'] == N and audit['index']['old_count'] == 8954
            and audit['index']['extra_count'] == 46 and audit['index']['raw_shape'] == [972, 1296]
            and audit['index']['raw_dtype'] == 'uint8' and len(audit['extras']) == 46,
            'Data population/backend contract differs')
    reader = make_reader(args)
    require(reader.target_arrays_loaded is False and reader.provenance['target_arrays_loaded'] is False,
            'Reader reported supervised target access')
    ids = np.asarray(reader.ids).astype(str)
    indices = validate_population(ids)
    require(reader.mapping.shape == (N,) and int((reader.mapping >= 0).sum()) == 8954
            and int((reader.mapping == -1).sum()) == 46, 'Native/MHA mapping differs')
    require([r['index'] for r in audit['extras']] == np.flatnonzero(reader.mapping == -1).tolist(),
            'Extra image inventory differs')
    for record in audit['extras']:
        require(record['id'] == ids[record['index']] and record['central512_equal'] is True
                and record['original_label_matches_source_csv'] is True
                and (args.workspace / record['path']).stat().st_size == record['bytes'],
                'Extra image audit incomplete')
        # Reader already checks actual bytes, decoded native pixels, and metadata.
        require(hashlib.sha256(get_frame(reader, record['index']).tobytes()).hexdigest()
                == record['pixel_sha256'], 'Current extra native pixels differ')
    require(sha(args.workspace / WEIGHT) == EXTERNAL_SHA, 'External CPDINO initialization changed')
    hashes = {str(args.data_audit): args.expected_data_audit_sha256,
              str(args.current_raw_verification): args.expected_current_raw_verification_sha256,
              str(args.workspace / WEIGHT): EXTERNAL_SHA}
    for record in [*audit['inputs'].values(), *audit['extras']]:
        path = args.workspace / record['path']
        require(sha(path) == record['sha256'], f'Data inventory hash differs: {path}')
        hashes[str(path)] = record['sha256']
    current = None
    if rehash_raw:
        # This is a real prelaunch content hash, not trust in an earlier report.
        current = reader_module(args).verify_current_raw_sha(args.workspace)
        raw_report_contract(current, args.workspace)
    return ids, indices, audit, hashes, current


def prepare(args):
    deadline_contract(args.deadline_utc)
    require(not args.root.exists(), 'Experiment root already exists; use a new isolated root')
    base, _, torch = runtime(args)
    require(not torch.cuda.is_initialized(), 'Prepare must not initialize CUDA')
    base.check_deadline(base.deadline_timestamp(args.deadline_utc), reserve=120)
    ids, indices, audit, hashes, current = data_evidence(args, rehash_raw=True)
    base.check_deadline(base.deadline_timestamp(args.deadline_utc), reserve=120)
    args.root.mkdir(parents=True)
    hashes.update(freeze_sources(args))
    m = dict(schema=SCHEMA, training_scope=SCOPE, created_utc=now(),
             population_n=N, heldout_n=0, independent_accuracy_validation=False,
             target_arrays_loaded=False, seed=SEED, epochs=EPOCHS,
             steps_per_epoch=STEPS, total_steps=EPOCHS * STEPS, last_batch_images=8,
             population_ids_sha=ids_sha(ids), indices_sha=index_sha(indices),
             data_ordered_ids_sha256=audit['index']['ordered_ids_sha256'],
             old_index_mapping_sha256=audit['index']['old_index_mapping_sha256'],
             image_inventory_sha256=audit['image_inventory_sha256'],
             data_audit_path=str(args.data_audit), data_audit_sha256=args.expected_data_audit_sha256,
             current_raw_verification_path=str(args.current_raw_verification),
             current_raw_verification_sha256=args.expected_current_raw_verification_sha256,
             current_raw_verified_at_prepare=current,
             raw_path=str(args.workspace / RAW), raw_sha256=RAW_SHA, raw_bytes=RAW_BYTES,
             native_shape=[972, 1296], native_dtype='uint8', native_cache_images=8954, extra_mha_images=46,
             architecture='CellposeDINO-B (DINOv3 ViT-B, kernel16/stride8)',
             initialization='external CPDINO checkpoint; fresh full-population SSL',
             external_checkpoint_path=str(args.workspace / WEIGHT), external_checkpoint_sha=EXTERNAL_SHA,
             canonical_encoder_state=True, recipe=recipe(base), deadline_utc=args.deadline_utc,
             reader_source_sha256=args.expected_reader_sha256, original_source_sha256=PINNED,
             runner_source_sha256=sha(__file__), hashes=hashes,
             limitations=LIMITATIONS, official_submission_changed=False)
    validate_scope(m)
    exclusive_json(args.root / 'manifest.json', m)
    print(json.dumps(dict(status='prepared', manifest_sha=sha(args.root / 'manifest.json'), **m)), flush=True)


def verify(args, *, launch=False):
    deadline_contract(args.deadline_utc)
    base, _, torch = runtime(args)
    m = read(args.root / 'manifest.json')
    validate_scope(m)
    require(m['recipe'] == recipe(base) and m['deadline_utc'] == args.deadline_utc
            and m['data_audit_path'] == str(args.data_audit)
            and m['data_audit_sha256'] == args.expected_data_audit_sha256
            and m['current_raw_verification_path'] == str(args.current_raw_verification)
            and m['current_raw_verification_sha256'] == args.expected_current_raw_verification_sha256
            and m['reader_source_sha256'] == args.expected_reader_sha256
            and m['original_source_sha256'] == PINNED and m['runner_source_sha256'] == sha(__file__),
            'CLI/runtime differs from frozen manifest')
    for path, expected in m['hashes'].items():
        require(sha(path) == expected, f'Frozen source/input changed: {path}')
    for name, path in source_paths(args).items():
        require(sha(path) == m['hashes'][str(args.root / 'source' / name)],
                f'Executed source differs from frozen copy: {name}')
    ids, indices, audit, hashes, current = data_evidence(args, rehash_raw=launch)
    require(m['population_ids_sha'] == ids_sha(ids) and m['indices_sha'] == index_sha(indices)
            and m['data_ordered_ids_sha256'] == audit['index']['ordered_ids_sha256']
            and m['old_index_mapping_sha256'] == audit['index']['old_index_mapping_sha256']
            and m['image_inventory_sha256'] == audit['image_inventory_sha256']
            and m['raw_sha256'] == RAW_SHA and m['raw_bytes'] == RAW_BYTES
            and m['external_checkpoint_sha'] == EXTERNAL_SHA
            and m['canonical_encoder_state'] is True, 'Population/input identity changed')
    require(all(m['hashes'].get(path) == value for path, value in hashes.items()),
            'An actual data input is absent from frozen hashes')
    if launch:
        args.launch_current_raw = current
        base.check_deadline(base.deadline_timestamp(args.deadline_utc), reserve=120)
    return m, ids, indices


class Views:
    """Duck-typed torch Dataset; worker-local image reader, no supervised API."""
    def __init__(self, args, indices, epoch):
        self.args, self.indices, self.epoch = args, np.asarray(indices, dtype=np.int64), epoch
        self.reader, self.base, self.torch = None, None, None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, row):
        import cv2
        cv2.setNumThreads(1)
        if self.reader is None:
            self.base, _, self.torch = runtime(self.args)
            self.reader = make_reader(self.args)
        ix = int(self.indices[row])
        # Identical to the pinned fold runner; central512 never enters the model.
        a = cv2.createCLAHE(2., (1296 // 32, 972 // 32)).apply(get_frame(self.reader, ix))
        lo, hi = np.percentile(a, (1, 99))
        a = np.clip((a.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
        return self.torch.stack([self.base.augmented_view(a, SEED, self.epoch, ix, v)
                                 for v in range(2)]), ix


def loader(args, indices, epoch, batch=BATCH):
    _, _, torch = runtime(args)
    return torch.utils.data.DataLoader(Views(args, indices, epoch), batch_size=batch,
                                      shuffle=False, num_workers=2, pin_memory=True,
                                      prefetch_factor=1, drop_last=False)


def tensor_sha(state, torch):
    h = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        h.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode())
        h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def execution_details(args, details, encoder, torch):
    require(details['weight_sha256'] == EXTERNAL_SHA and details['fresh_external_initialization'] is True,
            'Model did not start freshly from the pinned external CPDINO')
    return {**details, 'training_scope': SCOPE, 'target_arrays_loaded': False,
            'runner_source_sha256': sha(__file__), 'reader_source_sha256': args.expected_reader_sha256,
            'original_source_sha256': PINNED, 'population_n': N, 'heldout_n': 0,
            'external_checkpoint_sha': EXTERNAL_SHA,
            'external_encoder_tensor_sha': tensor_sha(encoder.original.state_dict(), torch),
            'launch_current_raw': args.launch_current_raw}


def check_values(values, count):
    require(values['actual_images'] == count, 'Actual batch size differs')
    for key in ('loss', 'invariance', 'sigreg', 'feature_std', 'projector_std',
                'effective_rank', 'replay_max_abs_error', 'gradient_norm'):
        require(math.isfinite(values[key]), f'Nonfinite SSL telemetry: {key}')
    require(values['feature_std'] > 0 and values['projector_std'] > 0 and values['gradient_norm'] > 0,
            'Collapsed features or zero gradient')


def prerequisite(args, name, manifest_sha):
    value = read(args.root / name)
    require(value['status'] == 'passed' and value['manifest_sha'] == manifest_sha
            and value['training_scope'] == SCOPE and value['target_arrays_loaded'] is False
            and value['runner_source_sha256'] == sha(__file__), f'Unverified prerequisite: {name}')
    return value


def train(args):
    m, ids, indices = verify(args, launch=True)
    base, pilot, torch = runtime(args)
    require(torch.cuda.device_count() == 1, 'Exactly one visible GPU is required')
    manifest_sha = sha(args.root / 'manifest.json')
    prerequisite(args, 'cpu_tests.json', manifest_sha)
    smoke = prerequisite(args, 'smoke.json', manifest_sha)
    require(smoke['production_batch_images'] == 32 and smoke['partial_batch_images'] == 8
            and smoke['training_requires_fresh_setup'] is True
            and smoke['smoke_checkpoint_saved'] is False, 'GPU smoke scope differs')
    deadline = base.deadline_timestamp(args.deadline_utc)
    base.check_deadline(deadline, reserve=120)
    run = args.root / 'run'
    run.mkdir()  # Exclusive: an interrupted run is never implicitly resumed.
    history, epochs, global_step, durable = [], [], 0, None
    started, reference_std = time.monotonic(), None
    base.arm_deadline(deadline)
    try:
        enc, proj, reg, opt, raw_details = base.setup(SEED)
        details = execution_details(args, raw_details, enc, torch)
        require(not opt.state, 'Fresh training must have empty optimizer state')
        progress_json(run / 'progress.json', dict(status='starting', manifest_sha=manifest_sha, details=details))
        for epoch in range(1, EPOCHS + 1):
            order = np.random.default_rng(SEED + epoch - 1).permutation(indices)
            dl = loader(args, order, epoch - 1)
            seen, epoch_start = [], time.monotonic()
            require(len(dl) == STEPS, 'Unexpected full-population epoch length')
            for step, (x, ix) in enumerate(dl, 1):
                base.check_deadline(deadline, reserve=90)
                expected = order[(step - 1) * BATCH:step * BATCH]
                np.testing.assert_array_equal(ix.numpy(), expected)
                ratio = min(1., (global_step + 1) / pilot.RECIPE['warmup_steps'])
                opt.param_groups[0]['lr'] = pilot.RECIPE['encoder_lr'] * ratio
                opt.param_groups[1]['lr'] = pilot.RECIPE['projector_lr'] * ratio
                tick = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                values = pilot.update(enc, proj, reg, opt, x, step == 1)
                torch.cuda.synchronize()
                check_values(values, len(expected))
                global_step += 1
                seen.extend(ix.tolist())
                if reference_std is None:
                    reference_std = values['feature_std']
                require(values['feature_std'] >= max(1e-6, reference_std * 1e-3)
                        and values['projector_std'] >= 1e-6, 'Collapse tripwire; no automatic retry')
                item = dict(status='ssl_training', epoch=epoch, epochs=EPOCHS,
                            step_this_epoch=step, steps_this_epoch=STEPS, global_step=global_step,
                            total_steps=EPOCHS * STEPS, seen_this_epoch=len(seen), population_n=N,
                            seconds_step=time.monotonic() - tick, elapsed_seconds=time.monotonic() - started,
                            peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                            encoder_lr=opt.param_groups[0]['lr'], projector_lr=opt.param_groups[1]['lr'], **values)
                history.append(item)
                progress_json(run / 'progress.json', item)
                if step == 1 or step % 5 == 0:
                    print(json.dumps(item), flush=True)
                if step % 50 == 0 or step == STEPS:
                    base.save_checkpoint(run, enc, proj, opt, epoch, global_step,
                                         manifest_sha, step, order, len(seen))
                    durable = dict(epoch=epoch, step=global_step, step_this_epoch=step)
                    progress_json(run / 'history.json', history)
            np.testing.assert_array_equal(np.asarray(seen, dtype=np.int64), order)
            require(len(set(seen)) == N and global_step == epoch * STEPS, 'Incomplete epoch coverage')
            checkpoint = run / f'epoch{epoch:02d}.pt'
            os.link(run / 'last.pt', checkpoint)
            record = dict(epoch=epoch, unique_train_images=N, train_images_seen=N,
                          population_n=N, heldout_n=0, target_arrays_loaded=False,
                          order_sha=index_sha(order), ordered_ids_sha=ids_sha(ids[order]),
                          checkpoint=str(checkpoint), checkpoint_sha=sha(checkpoint),
                          seconds=time.monotonic() - epoch_start, last=history[-1])
            epochs.append(record)
            progress_json(run / 'epochs.json', epochs)
            print(json.dumps(dict(epoch_complete=epoch, **record)), flush=True)
        require(global_step == EPOCHS * STEPS, 'Incomplete planned SSL steps')
        result = dict(status=RESULT_STATUS, schema=SCHEMA, training_scope=SCOPE,
                      epochs=EPOCHS, total_steps=global_step, steps=global_step,
                      population_n=N, unique_train_images=N, train_image_presentations=EPOCHS * N,
                      augmented_view_presentations=EPOCHS * N * 2,
                      heldout_n=0, independent_accuracy_validation=False, target_arrays_loaded=False,
                      seed=SEED, population_ids_sha=m['population_ids_sha'], indices_sha=m['indices_sha'],
                      epochs_detail=epochs, seconds=time.monotonic() - started, details=details,
                      manifest_sha=manifest_sha, checkpoint_sha=sha(run / 'last.pt'),
                      data_audit_sha256=m['data_audit_sha256'],
                      current_raw_verification_sha256=m['current_raw_verification_sha256'],
                      limitations=LIMITATIONS, finished_utc=now())
        exclusive_json(run / 'result.json', result)
        exclusive_json(run / 'complete.json', dict(status='completed', result_sha=sha(run / 'result.json'),
                                                  checkpoint_sha=result['checkpoint_sha'], manifest_sha=manifest_sha))
        progress_json(run / 'progress.json', dict(status='completed', epochs=EPOCHS, global_step=global_step))
        print(json.dumps(result), flush=True)
    except base.DeadlineReached as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        stopped = dict(status='deadline_stopped', deadline_utc=args.deadline_utc, stopped_utc=now(),
                       manifest_sha=manifest_sha, completed_epochs=len(epochs), observed_global_step=global_step,
                       durable_checkpoint=durable, reason=str(exc),
                       checkpoint_sha=sha(run / 'last.pt') if (run / 'last.pt').exists() else None,
                       note='Last durable checkpoint retained; interrupted optimizer state discarded; no resume')
        exclusive_json(run / 'deadline_stop.json', stopped)
        progress_json(run / 'progress.json', stopped)
        print(json.dumps(stopped), flush=True)
        return 3
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return 0


def fixture_tests():
    """Contract-only synthetic tests: no torch, model, real dataset, or weight."""
    cases = []
    def reject(name, fn):
        try:
            fn()
        except (ValueError, FileExistsError):
            cases.append(name)
        else:
            raise AssertionError('Invalid fixture accepted: ' + name)
    ids = np.asarray([f'synthetic-image-{i:04d}' for i in range(N)])
    indices = validate_population(ids)
    require(len(indices) == N and math.ceil(N / BATCH) == STEPS and N % BATCH == 8, 'Population arithmetic')
    cases.append('9000 images; 282 updates; last8; 1410 total updates')
    for epoch in range(EPOCHS):
        order = np.random.default_rng(SEED + epoch).permutation(indices)
        again = np.random.default_rng(SEED + epoch).permutation(indices)
        np.testing.assert_array_equal(order, again)
        require(len(np.unique(order)) == N and len(order[-8:]) == 8, 'Coverage fixture')
        if epoch:
            require(not np.array_equal(order, previous), 'Epoch order seed ignored')
        previous = order
    cases.append('five exact deterministic population permutations')
    deadline_contract(MAX_DEADLINE)
    for value in ('2026-10-19T04:00:00Z', '2026-10-19T03:59:00', '2026-10-18T23:59:00-04:00'):
        reject('invalid/unapproved deadline ' + value, lambda value=value: deadline_contract(value))
    reject('missing population row', lambda: validate_population(ids[:-1]))
    duplicate = ids.copy(); duplicate[-1] = duplicate[0]
    reject('duplicate population ID', lambda: validate_population(duplicate))
    reject('nonvector population IDs', lambda: validate_population(ids[:, None]))
    reject('numeric population IDs', lambda: validate_population(indices))
    valid = dict(schema=SCHEMA, training_scope=SCOPE, population_n=N, heldout_n=0,
                 independent_accuracy_validation=False, target_arrays_loaded=False,
                 seed=SEED, epochs=EPOCHS, steps_per_epoch=STEPS, total_steps=EPOCHS * STEPS,
                 last_batch_images=8)
    validate_scope(valid)
    for key, value in [('population_n', 8998), ('heldout_n', 1791), ('seed', 42), ('epochs', 4),
                       ('total_steps', 1409), ('last_batch_images', 10),
                       ('target_arrays_loaded', True), ('independent_accuracy_validation', True),
                       ('heldout_n', False), ('fold', 0), ('validation_ids_sha', 'fake')]:
        bad = {**valid, key: value}
        reject('reject contract ' + key + '=' + str(value), lambda bad=bad: validate_scope(bad))
    with tempfile.TemporaryDirectory(prefix='full9000-ssl-contract-') as directory:
        root = Path(directory)
        np.savez(root / 'synthetic.npz', ids=ids, Y=np.array([object()]), Y_train=np.array([object()]))
        with ids_only_npz():
            with np.load(root / 'synthetic.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], ids)
                reject('label Y access denied before deserialization', lambda: z['Y'])
                reject('label Y_train access denied before deserialization', lambda: z['Y_train'])
        cases.append('label-free ID access and NPZ guard restore')
        path = root / 'evidence.json'
        exclusive_json(path, valid)
        original = sha(path)
        reject('immutable evidence overwrite', lambda: exclusive_json(path, {'wrong': True}))
        require(sha(path) == original, 'Exclusive output changed')
        (root / 'run').mkdir()
        reject('existing run resume refused', lambda: (root / 'run').mkdir())
        raw = dict(path=str(RAW), sha256=RAW_SHA, bytes=RAW_BYTES,
                   checked_utc=now(), current_raw_sha256_verified=True)
        raw_report_contract(raw, root)
        for key, value in [('sha256', '0' * 64), ('bytes', RAW_BYTES - 1),
                           ('current_raw_sha256_verified', False), ('path', 'other.dat')]:
            bad = {**raw, key: value}
            reject('raw evidence ' + key, lambda bad=bad: raw_report_contract(bad, root))
        class Frame:
            def __init__(self, value): self.value = value
            def raw_frame(self, index): return self.value
        require(get_frame(Frame(np.zeros((972, 1296), dtype=np.uint8)), 0).shape == (972, 1296),
                'Native frame fixture')
        reject('central512 model input', lambda: get_frame(Frame(np.zeros((512, 512), dtype=np.uint8)), 0))
        reject('float native image input', lambda: get_frame(Frame(np.zeros((972, 1296), dtype=np.float32)), 0))
        reject('boolean index', lambda: get_frame(Frame(None), True))
    return dict(status='passed', scope='synthetic comparison/data contract only; not scientific model evidence',
                model_constructed=False, torch_imported='torch' in sys.modules,
                research_dataset_used=False, external_checkpoint_used=False, test_count=len(cases), tests=cases)


def cpu_tests(args):
    require(not (args.root / 'cpu_tests.json').exists(), 'Prior CPU evidence already exists')
    m, ids, _ = verify(args)
    base, pilot, torch = runtime(args)
    require(not torch.cuda.is_initialized(), 'CPU tests must not initialize CUDA')
    torch.set_num_threads(2)
    old_lazy = torch.cuda._lazy_init
    def deny_cuda(*_args, **_kwargs):
        raise AssertionError('CUDA initialization forbidden during CPU tests')
    torch.cuda._lazy_init = deny_cuda
    try:
        fixtures = fixture_tests()
        # Original tiny CPU gradient/RNG/deadline tests; no foundation model.
        base.tests(argparse.Namespace(root=None))
        base.seed_all(19)
        nn = torch.nn
        encoder = nn.Sequential(nn.Linear(7, 11), nn.Dropout(.2), nn.Linear(11, 768))
        projector = nn.Sequential(nn.Linear(768, 128), nn.BatchNorm1d(128))
        other_encoder, other_projector = copy.deepcopy(encoder), copy.deepcopy(projector)
        x, sigreg = torch.randn(8 * 2, 7), pilot.SIGReg()
        initial = pilot.rng_state()
        loss, _, _, _ = pilot.objective(torch.cat([encoder(p) for p in x.split(4)]), projector, sigreg)
        loss.backward()
        pilot.restore_rng(initial)
        values = pilot.cache_backward(other_encoder, other_projector, sigreg, x, 4, lambda e, p: e(p), True)
        require(values['actual_images'] == 8 and abs(loss.item() - values['loss']) < 1e-6,
                'Eight-image partial batch objective mismatch')
        error = 0.
        for p, q in zip(list(encoder.parameters()) + list(projector.parameters()),
                        list(other_encoder.parameters()) + list(other_projector.parameters())):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=3e-5)
            error = max(error, float((p.grad - q.grad).abs().max()))
        # Real data verification covers all46 MHA and the entire ordered ID map.
        reader = make_reader(args)
        chosen = [int(np.flatnonzero(reader.mapping >= 0)[0]), int(np.flatnonzero(reader.mapping < 0)[0])]
        for ix in chosen:
            get_frame(reader, ix)
        require(reader.target_arrays_loaded is False and not torch.cuda.is_initialized(), 'CPU scope breach')
        result = dict(status='passed', manifest_sha=sha(args.root / 'manifest.json'), created_utc=now(),
                      training_scope=SCOPE, target_arrays_loaded=False, runner_source_sha256=sha(__file__),
                      fixture_tests=fixtures, original_cpu_gradient_rng_deadline_suite_passed=True,
                      partial8_gradient_max_abs_error=error, population_ids_sha=ids_sha(ids),
                      native_backend_probe_indices=chosen, no_foundation_model_constructed=True,
                      cuda_initialized=False, cpu_threads=torch.get_num_threads(),
                      scope='Toy CPU gradient/RNG tests and real image integrity; no accuracy claim')
        exclusive_json(args.root / 'cpu_tests.json', result)
        print(json.dumps(result), flush=True)
    finally:
        torch.cuda._lazy_init = old_lazy


def smoke(args):
    require(not (args.root / 'smoke.json').exists(), 'Prior smoke evidence already exists')
    m, _, indices = verify(args, launch=True)
    base, pilot, torch = runtime(args)
    require(torch.cuda.device_count() == 1, 'Exactly one visible GPU is required')
    manifest_sha = sha(args.root / 'manifest.json')
    prerequisite(args, 'cpu_tests.json', manifest_sha)
    reader = make_reader(args)
    permutation = np.random.default_rng(SEED).permutation(indices)
    # Force both backends into the feasibility batch; this is not the training order.
    extra = int(np.flatnonzero(reader.mapping < 0)[0])
    shared = int(np.flatnonzero(reader.mapping >= 0)[0])
    selected = np.asarray([shared, extra] + [int(i) for i in permutation if i not in (shared, extra)][:30])
    partial = permutation[-8:]
    deadline = base.deadline_timestamp(args.deadline_utc)
    base.check_deadline(deadline, reserve=120)
    base.arm_deadline(deadline)
    try:
        enc, proj, reg, opt, raw_details = base.setup(SEED)
        details = execution_details(args, raw_details, enc, torch)
        require(not opt.state, 'Smoke must start fresh')
        tracked = {name: p for name, p in enc.named_parameters() if p.ndim >= 2}
        before = {name: p.detach().clone() for name, p in tracked.items()}
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        x, ix = next(iter(loader(args, selected, 0)))
        np.testing.assert_array_equal(ix.numpy(), selected)
        values = pilot.update(enc, proj, reg, opt, x, True)
        check_values(values, 32)
        changed = [name for name, p in tracked.items() if not torch.equal(before[name], p.detach())]
        blocks = lambda names: {re.search(r'blocks\.\d+', name).group() for name in names
                                if re.search(r'blocks\.\d+', name)}
        require(blocks(changed) == blocks(tracked) and any('patch_embed' in name for name in changed),
                'Missing encoder block or patch projection update')
        del before, x
        x, ix = next(iter(loader(args, partial, 0)))
        np.testing.assert_array_equal(ix.numpy(), partial)
        partial_values = pilot.update(enc, proj, reg, opt, x, True)
        check_values(partial_values, 8)
        require(all(int(state['step'].item()) == 2 for state in opt.state.values()), 'Smoke optimizer step mismatch')
        for module in proj.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                require(int(module.num_batches_tracked) == 2 and torch.isfinite(module.running_mean).all().item()
                        and torch.isfinite(module.running_var).all().item(), 'Projector full-batch BN evidence differs')
        torch.cuda.synchronize()
        result = dict(status='passed', manifest_sha=manifest_sha, created_utc=now(), training_scope=SCOPE,
                      runner_source_sha256=sha(__file__), target_arrays_loaded=False, values=values,
                      partial_values=partial_values, production_batch_images=32, partial_batch_images=8,
                      selected_indices=selected.tolist(), partial_indices=partial.tolist(),
                      native_cache_and_extra_mha_in_production_batch=True,
                      encoder_blocks_updated=sorted(blocks(changed)), optimizer_step_after_updates=2,
                      optimizer_state_parameter_ids=sorted(opt.state_dict()['state']),
                      training_requires_fresh_setup=True, smoke_checkpoint_saved=False,
                      seconds=time.monotonic() - started, peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                      details=details)
        exclusive_json(args.root / 'smoke.json', result)
        print(json.dumps(result), flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def checkpoint_contract(state, epoch, manifest_sha, order, external, expected_optimizer_ids, torch):
    require(state['epoch'] == epoch and state['step'] == epoch * STEPS
            and state['step_this_epoch'] == STEPS and state['seen_this_epoch'] == N
            and state['manifest_sha'] == manifest_sha, 'Checkpoint progress/manifest binding differs')
    require(state['order'].dtype == torch.int64 and state['order'].shape == (N,), 'Checkpoint order dtype/shape')
    np.testing.assert_array_equal(state['order'].numpy(), order)
    require(state['encoder'].keys() == external.keys(), 'Canonical encoder keys differ from external architecture')
    for name, tensor in state['encoder'].items():
        require(tensor.shape == external[name].shape and tensor.dtype == external[name].dtype,
                f'Encoder shape/dtype differs: {name}')
    shapes = {'0.weight': (2048, 768), '0.bias': (2048,), '1.weight': (2048,), '1.bias': (2048,),
              '1.running_mean': (2048,), '1.running_var': (2048,), '1.num_batches_tracked': (),
              '3.weight': (2048, 2048), '3.bias': (2048,), '4.weight': (2048,), '4.bias': (2048,),
              '4.running_mean': (2048,), '4.running_var': (2048,), '4.num_batches_tracked': (),
              '6.weight': (128, 2048), '6.bias': (128,)}
    require(state['projector'].keys() == shapes.keys(), 'Projector state keys differ')
    for name, tensor in state['projector'].items():
        dtype = torch.int64 if name.endswith('num_batches_tracked') else torch.float32
        require(tuple(tensor.shape) == shapes[name] and tensor.dtype == dtype, 'Projector shape/dtype differs')
        if name.endswith('num_batches_tracked'):
            require(int(tensor) == epoch * STEPS, 'Projector BN step count differs')
    for section in ('encoder', 'projector'):
        for name, tensor in state[section].items():
            require(tensor.device.type == 'cpu', 'Audit loaded GPU state')
            if torch.is_floating_point(tensor):
                require(torch.isfinite(tensor).all().item(), f'Nonfinite {section}:{name}')
    optimizer = state['optimizer']
    require(len(optimizer['param_groups']) == 2 and optimizer['state'], 'Optimizer state absent')
    group_ids = [p for group in optimizer['param_groups'] for p in group['params']]
    # PyTorch legitimately omits optimizer state for an unused encoder parameter.
    # Match the actual two-update smoke coverage, rather than inventing state for it.
    require(len(set(group_ids)) == len(group_ids)
            and set(optimizer['state']).issubset(group_ids)
            and sorted(optimizer['state']) == expected_optimizer_ids
            and set(optimizer['param_groups'][1]['params']).issubset(optimizer['state']),
            'Optimizer state coverage differs')
    for group, lr in zip(optimizer['param_groups'], (1e-5, 1e-3)):
        require(group['lr'] == lr and group['weight_decay'] == 1e-4, 'Final optimizer recipe differs')
    for entry in optimizer['state'].values():
        require(int(entry['step'].item()) == epoch * STEPS, 'Optimizer step differs')
        for key in ('exp_avg', 'exp_avg_sq'):
            require(key in entry and entry[key].device.type == 'cpu'
                    and torch.isfinite(entry[key]).all().item(), 'Optimizer moments missing/nonfinite')
        require(entry['exp_avg'].shape == entry['exp_avg_sq'].shape
                and entry['exp_avg'].dtype == entry['exp_avg_sq'].dtype, 'Optimizer moment shape/dtype differs')
    require(state['rng_cpu'].dtype == torch.uint8 and state['rng_cpu'].ndim == 1
            and len(state['rng_cuda']) == 1 and state['rng_cuda'][0].dtype == torch.uint8
            and state['rng_cuda'][0].ndim == 1 and state['rng_cuda'][0].numel() > 0
            and isinstance(state['rng_python'], tuple) and state['rng_numpy']['kind'] == 'MT19937'
            and len(state['rng_numpy']['keys']) == 624, 'Full RNG state is absent or malformed')
    torch.Generator(device='cpu').set_state(state['rng_cpu'])
    random.Random().setstate(state['rng_python'])
    n = state['rng_numpy']
    require(all(type(value) is int and 0 <= value <= 2**32 - 1 for value in n['keys']), 'Invalid NumPy RNG keys')
    np.random.RandomState().set_state((n['kind'], np.asarray(n['keys'], dtype=np.uint32),
                                      n['position'], n['has_gauss'], n['cached_gaussian']))


def audit(args):
    require(not (args.root / 'audit.json').exists(), 'Prior audit evidence already exists')
    m, ids, indices = verify(args)
    base, _, torch = runtime(args)
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), 'Audit must use CPU only')
    manifest_sha = sha(args.root / 'manifest.json')
    cpu = prerequisite(args, 'cpu_tests.json', manifest_sha)
    smoke_value = prerequisite(args, 'smoke.json', manifest_sha)
    run = args.root / 'run'
    result, complete = read(run / 'result.json'), read(run / 'complete.json')
    require(complete == dict(status='completed', result_sha=sha(run / 'result.json'),
                             checkpoint_sha=sha(run / 'last.pt'), manifest_sha=manifest_sha),
            'Final complete/result/checkpoint chain differs')
    require(result['status'] == RESULT_STATUS and result['schema'] == SCHEMA
            and result['training_scope'] == SCOPE and result['epochs'] == EPOCHS
            and result['steps'] == result['total_steps'] == EPOCHS * STEPS
            and result['population_n'] == result['unique_train_images'] == N
            and result['train_image_presentations'] == EPOCHS * N
            and result['augmented_view_presentations'] == EPOCHS * N * 2
            and result['heldout_n'] == 0 and result['independent_accuracy_validation'] is False
            and result['target_arrays_loaded'] is False and result['seed'] == SEED
            and result['population_ids_sha'] == m['population_ids_sha']
            and result['indices_sha'] == m['indices_sha']
            and result['manifest_sha'] == manifest_sha and result['checkpoint_sha'] == sha(run / 'last.pt')
            and result['data_audit_sha256'] == m['data_audit_sha256']
            and result['current_raw_verification_sha256'] == m['current_raw_verification_sha256']
            and not FORBIDDEN_SPLIT_FIELDS.intersection(result), 'Final result contract differs')
    external_all = torch.load(args.workspace / WEIGHT, map_location='cpu', weights_only=True)
    external = {name[len('encoder.'):]: value for name, value in external_all.items() if name.startswith('encoder.')}
    require(external, 'External canonical encoder state absent')
    external_tensor_sha = tensor_sha(external, torch)
    for details in (result['details'], smoke_value['details']):
        require(details['external_checkpoint_sha'] == EXTERNAL_SHA
                and details['external_encoder_tensor_sha'] == external_tensor_sha
                and details['fresh_external_initialization'] is True
                and details['runner_source_sha256'] == sha(__file__)
                and details['reader_source_sha256'] == args.expected_reader_sha256
                and details['original_source_sha256'] == PINNED
                and details['training_scope'] == SCOPE and details['target_arrays_loaded'] is False,
                'Actual fresh initialization/runtime identity differs')
        raw_report_contract(details['launch_current_raw'], args.workspace)
    del external_all
    epochs, history = read(run / 'epochs.json'), read(run / 'history.json')
    require(len(epochs) == EPOCHS and result['epochs_detail'] == epochs
            and len(history) == EPOCHS * STEPS, 'Epoch/history evidence differs')
    records = []
    for epoch, record in enumerate(epochs, 1):
        order = np.random.default_rng(SEED + epoch - 1).permutation(indices)
        checkpoint = run / f'epoch{epoch:02d}.pt'
        require(record['epoch'] == epoch and Path(record['checkpoint']).resolve() == checkpoint.resolve()
                and record['checkpoint_sha'] == sha(checkpoint)
                and record['unique_train_images'] == record['train_images_seen'] == record['population_n'] == N
                and record['heldout_n'] == 0 and record['target_arrays_loaded'] is False
                and record['order_sha'] == index_sha(order) and record['ordered_ids_sha'] == ids_sha(ids[order]),
                'Epoch checkpoint/coverage identity differs')
        rows = history[(epoch - 1) * STEPS:epoch * STEPS]
        for step, row in enumerate(rows, 1):
            count = BATCH if step < STEPS else 8
            require(row['epoch'] == epoch and row['step_this_epoch'] == step
                    and row['global_step'] == (epoch - 1) * STEPS + step
                    and row['seen_this_epoch'] == min(step * BATCH, N)
                    and row['population_n'] == N, 'History progression differs')
            check_values(row, count)
            ratio = min(1., row['global_step'] / 20)
            require(row['encoder_lr'] == 1e-5 * ratio and row['projector_lr'] == 1e-3 * ratio,
                    'History learning-rate schedule differs')
        require(record['last'] == rows[-1], 'Epoch last history row differs')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        checkpoint_contract(state, epoch, manifest_sha, order, external,
                            smoke_value['optimizer_state_parameter_ids'], torch)
        records.append(dict(epoch=epoch, checkpoint_sha=sha(checkpoint), step=state['step'],
                            optimizer_steps=epoch * STEPS, unique_train_images=N, last_batch_images=8,
                            encoder_tensor_sha=tensor_sha(state['encoder'], torch)))
        del state
    require(sha(run / 'last.pt') == records[-1]['checkpoint_sha'] and not torch.cuda.is_initialized(),
            'Terminal checkpoint or CPU-only audit differs')
    value = dict(status='full_population_ssl_artifacts_verified_no_heldout_score', schema=SCHEMA,
                 checked_utc=now(), training_scope=SCOPE, manifest_sha=manifest_sha,
                 result_sha=sha(run / 'result.json'), complete_sha=sha(run / 'complete.json'),
                 checkpoint_sha=sha(run / 'last.pt'), cpu_tests_sha=sha(args.root / 'cpu_tests.json'),
                 smoke_sha=sha(args.root / 'smoke.json'), epochs_sha=sha(run / 'epochs.json'),
                 history_sha=sha(run / 'history.json'), runner_source_sha256=sha(__file__),
                 reader_source_sha256=args.expected_reader_sha256, original_source_sha256=PINNED,
                 external_checkpoint_sha=EXTERNAL_SHA, external_encoder_tensor_sha=external_tensor_sha,
                 final_encoder_tensor_sha=records[-1]['encoder_tensor_sha'], epoch_checkpoints=records,
                 population_n=N, population_ids_sha=m['population_ids_sha'], heldout_n=0,
                 target_arrays_loaded=False, independent_accuracy_validation=False,
                 exact_population_coverage_each_epoch=True, weights_only_cpu_load=True,
                 cuda_initialized=False, cpu_threads=torch.get_num_threads(),
                 limitations=LIMITATIONS + ['Checkpoint artifact audit; does not instantiate or forward a restored model.'])
    exclusive_json(args.root / 'audit.json', value)
    print(json.dumps(value), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('fixture-tests', 'prepare', 'tests', 'smoke', 'train', 'audit'))
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--root', type=Path)
    parser.add_argument('--data-audit', type=Path)
    parser.add_argument('--expected-data-audit-sha256')
    parser.add_argument('--current-raw-verification', type=Path)
    parser.add_argument('--expected-current-raw-verification-sha256')
    parser.add_argument('--expected-reader-sha256')
    parser.add_argument('--deadline-utc', default=MAX_DEADLINE)
    args = parser.parse_args()
    if args.command == 'fixture-tests':
        print(json.dumps(fixture_tests()), flush=True)
        return 0
    for name in ('root', 'data_audit', 'expected_data_audit_sha256', 'current_raw_verification',
                 'expected_current_raw_verification_sha256', 'expected_reader_sha256'):
        if getattr(args, name) is None:
            parser.error('--' + name.replace('_', '-') + ' is required for data-bound commands')
    for name in ('expected_data_audit_sha256', 'expected_current_raw_verification_sha256', 'expected_reader_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', getattr(args, name)):
            parser.error(name + ' must be a reviewed lowercase SHA256')
    args.workspace = args.workspace.resolve()
    deadline_contract(args.deadline_utc)
    for name in ('root', 'data_audit', 'current_raw_verification'):
        setattr(args, name, (args.workspace / getattr(args, name)).resolve())
    os.chdir(args.workspace)  # Pinned original builder uses workspace-relative external checkpoint paths.
    if args.command in ('prepare', 'tests', 'audit'):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['OMP_NUM_THREADS'] = '2'
        os.environ['MKL_NUM_THREADS'] = '2'
    function = cpu_tests if args.command == 'tests' else globals()[args.command]
    result = function(args)
    return result if isinstance(result, int) else 0


if __name__ == '__main__':
    raise SystemExit(main())
