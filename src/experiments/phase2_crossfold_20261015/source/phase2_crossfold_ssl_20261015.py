#!/usr/bin/env python3
"""Fresh fold-clean CellposeDINO LeJEPA SSL; one visible GPU, no labels.

The frozen 20261003 module owns SIGReg, objective, gradient caching, padded
GPU forward and optimizer-update math. This runner parameterizes membership,
seed and duration, starts from the external encoder, and never reads old SSL
checkpoints. Run prepare, tests, smoke and train in separate processes.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import phase2_lejepa_20261003 as pilot
import phase2_foundation_preflight_20261002 as foundation

META = Path('datasets/clear_ec_official/cache_native_meta_512.npz')
FULL_META = Path('datasets/clear_ec_fullframe_20260904/fullframe_meta.npz')
RAW = Path('datasets/clear_ec_fullframe_20260904/img_full_raw.dat')
WEIGHT = pilot.WEIGHT
EXPECTED_META = '956f448a2de4d65b4e091d78f56bdde09aab9554e3ab027fc7fe58f8b69625e6'
EXPECTED_RAW = '93e67ac6df5887a1a5396c85e236b8398297dd3e728cdb6513238c1628c1c88a'
EXPECTED_REUSED_SOURCES = {
    'phase2_lejepa_20261003.py':'34ce7a35a6b4fa72fa6deb07ba5b4523bb31c21616b0b98c9ea88be6b6ddc88f',
    'phase2_foundation_preflight_20261002.py':'196500b67d01ad9423ea4d7d8c1a1e4b2d886df2328d36bfb1e1c94058ca4b1e',
}
SOURCES = (Path(__file__), Path(pilot.__file__), Path(foundation.__file__))
digest, save = foundation.digest, foundation.save


def read(path):
    return json.loads(Path(path).read_text())


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


class DeadlineReached(RuntimeError):
    pass


def deadline_timestamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError('--deadline-utc must include UTC Z or +00:00')
    return parsed.timestamp()


def check_deadline(deadline, reserve=0., now=None):
    if (time.time() if now is None else now) + reserve >= deadline:
        raise DeadlineReached('Training deadline reached; no further optimizer updates')


def arm_deadline(deadline):
    check_deadline(deadline)
    def expired(_signum, _frame):
        raise DeadlineReached('Absolute UTC deadline reached')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.time()))


def recipe(args):
    r = copy.deepcopy(pilot.RECIPE)
    r.update(seed=args.seed, epochs=args.epochs, status='FRESH_FOLD_CLEAN_SSL',
             order_seed_rule='seed + zero_based_epoch',
             augmentation_seed_rule='SeedSequence([seed,zero_based_epoch,image_index,view])',
             lr_policy='linear global20step warmup, then fixed LR through all epochs',
             initialization='external CPDINO only; fresh optimizer/projector/RNG')
    return r


def id_sha(ids, indices):
    return hashlib.sha256('\n'.join(ids[indices]).encode()).hexdigest()


def validate_membership(ids, train, val, primary=None):
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('Image IDs must be a unique one-dimensional vector')
    for name, index in (('train', train), ('validation', val)):
        if (index.ndim != 1 or not np.issubdtype(index.dtype, np.integer)
                or len(index) < 2 or index.min() < 0 or index.max() >= len(ids)
                or len(np.unique(index)) != len(index)):
            raise ValueError(f'Invalid {name} indices')
    if np.intersect1d(train, val).size:
        raise ValueError('Train and full validation overlap')
    if set(ids[train]) & set(ids[val]):
        raise ValueError('Train and validation IDs overlap')
    if len(train) % 32 == 1:
        raise ValueError('SIGReg requires >=2 images; unsupported singleton final batch')
    if primary is not None:
        if (primary.ndim != 1 or not np.issubdtype(primary.dtype, np.integer)
                or len(np.unique(primary)) != len(primary)
                or not np.isin(primary, val).all()):
            raise ValueError('Primary validation must be a unique subset of full validation')


def split(path, meta=META, full_meta=FULL_META):
    # NPZ is lazy: only ids is accessed. Target arrays are never loaded.
    with np.load(meta, allow_pickle=True) as z:
        ids = z['ids'].astype(str)
    with np.load(full_meta, allow_pickle=True) as z:
        np.testing.assert_array_equal(ids, z['ids'].astype(str))
    with np.load(path, allow_pickle=False) as z:
        train, val = z['train'].copy(), z['validation'].copy()
        primary = z['primary_validation'].copy() if 'primary_validation' in z.files else None
        validate_membership(ids, train, val, primary)
        for name, indices in (('train_ids', train), ('validation_ids', val)):
            if name in z.files:
                np.testing.assert_array_equal(z[name].astype(str), ids[indices])
    return ids, train.astype(np.int64), val.astype(np.int64), primary


def freeze_sources(root):
    result = {}
    directory = root / 'source'
    directory.mkdir(parents=True, exist_ok=True)
    for original in SOURCES:
        if original.name in EXPECTED_REUSED_SOURCES and digest(original) != EXPECTED_REUSED_SOURCES[original.name]:
            raise ValueError(f'Original audited implementation changed: {original}')
        frozen = directory / original.name
        if frozen.exists():
            if digest(frozen) != digest(original):
                raise FileExistsError(f'Frozen source differs: {frozen}')
        else:
            with original.open('rb') as src, frozen.open('xb') as dst:
                shutil.copyfileobj(src, dst)
        result[str(frozen)] = digest(frozen)
    return result


def preflight_evidence(args, ids, train, val):
    audit = read(args.data_preflight)
    if (audit['status'] != 'shared_checks_passed' or audit['errors']
            or audit['failed_shared_checks'] or not audit['pixels']['verified']
            or not audit['checks'] or not all(c['passed'] for c in audit['checks'])):
        raise ValueError('Cross-fold data preflight has unverified or failed shared checks')
    if (audit['pixels']['whole_cache_sha256'] != EXPECTED_RAW
            or audit['pixels']['n'] != len(ids) or audit['pixels']['mismatch_rows'] != 0):
        raise ValueError('Preflight current-pixel identity differs')
    matches = [entry for path, entry in audit['splits'].items()
               if Path(path).resolve() == args.split.resolve()]
    if len(matches) != 1:
        raise ValueError('Chosen split missing or ambiguous in data preflight')
    entry = matches[0]
    if (not entry['eligible_positive_target_candidate']
            or entry['split_sha256'] != digest(args.split)
            or not entry['checks'] or not all(c['passed'] for c in entry['checks'])
            or entry['train']['n'] != len(train) or entry['validation']['n'] != len(val)
            or str(args.fold) not in entry['provenance']['matched_manifest_folds']):
        raise ValueError('Chosen fold has no matching eligible preflight evidence')
    for path in (META, FULL_META):
        matches = [record for name, record in audit['inputs'].items()
                   if Path(name).resolve() == path.resolve()]
        if len(matches) != 1 or matches[0]['sha256'] != digest(path):
            raise ValueError(f'Preflight metadata identity differs: {path}')
    return entry


def prepare(args):
    check_deadline(deadline_timestamp(args.deadline_utc))
    if (args.root / 'manifest.json').exists() or (args.root / 'run').exists():
        raise FileExistsError('Never overwrite or implicitly resume an experiment')
    ids, train, val, primary = split(args.split)
    preflight_evidence(args, ids, train, val)
    if digest(META) != EXPECTED_META or digest(WEIGHT) != pilot.EXPECTED_WEIGHT:
        raise ValueError('Audited metadata or external initialization changed')
    expected_bytes = len(ids) * 972 * 1296
    if RAW.stat().st_size != expected_bytes or digest(RAW) != EXPECTED_RAW:
        raise ValueError('Raw image cache differs from audited bytes/hash')
    hashes = freeze_sources(args.root)
    for path in (META, FULL_META, WEIGHT, args.split, args.data_preflight):
        hashes[str(path)] = digest(path)
    m = dict(created_utc=utc_now(), fold=args.fold, seed=args.seed,
             architecture='CellposeDINO-B (DINOv3 ViT-B, kernel16/stride8)',
             initialization='external CPDINO checkpoint; fresh fold-specific SSL',
             recipe=recipe(args), deadline_utc=args.deadline_utc,
             split_path=str(args.split), split_sha256=digest(args.split),
             data_preflight_path=str(args.data_preflight),
             data_preflight_sha256=digest(args.data_preflight),
             train_n=len(train), val_n=len(val), validation_n=len(val),
             primary_validation_n=len(val),
             historical_primary_validation_n=len(primary) if primary is not None else len(val),
             total_images=len(ids), train_ids_sha=id_sha(ids, train),
             validation_ids_sha=id_sha(ids, val),
             data_path=str(RAW), metadata_path=str(META), full_metadata_path=str(FULL_META),
             raw_sha256=EXPECTED_RAW, raw_bytes=expected_bytes,
             raw_hash_verified_utc=utc_now(), raw_shape=[len(ids), 972, 1296],
             raw_hash_provenance='Full current raw-file SHA256 compared with 20261002 audited cache hash',
             hashes=hashes, total_steps=math.ceil(len(train)/32)*args.epochs,
             validation_images_used=0, target_arrays_loaded=False,
             source_reference_commit='c293d291ca87cd4fddee9d3fffe4e914c7272052',
             limitations=['SSL loss is not an accuracy result',
                          'Patient independence and near-duplicate leakage remain unverified',
                          'Full-frame two-view adaptation, not original multi-crop LeJEPA',
                          'T4 feasibility remains unverified'],
             official_submission_changed=False)
    save(args.root / 'manifest.json', m)
    print(json.dumps(dict(status='prepared', **m)), flush=True)


def verify(args):
    m = read(args.root / 'manifest.json')
    if (m['recipe'] != recipe(args) or m['fold'] != args.fold
            or m['split_path'] != str(args.split) or m['deadline_utc'] != args.deadline_utc
            or m['data_preflight_path'] != str(args.data_preflight)):
        raise ValueError('CLI differs from frozen manifest')
    for path, expected in m['hashes'].items():
        if digest(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')
    for original in SOURCES:
        if digest(original) != m['hashes'][str(args.root/'source'/original.name)]:
            raise ValueError(f'Executed module differs from frozen source: {original}')
    ids, train, val, _ = split(args.split)
    preflight_evidence(args, ids, train, val)
    if (RAW.stat().st_size != m['raw_bytes'] or id_sha(ids, train) != m['train_ids_sha']
            or id_sha(ids, val) != m['validation_ids_sha']):
        raise ValueError('Data identity changed')
    return m, ids, train, val


def augmented_view(a, seed, epoch, index, view):
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, index, view]))
    b = a
    if rng.random() < .5:
        b = b[:, ::-1]
    if rng.random() < .5:
        b = b[::-1, :]
    b = np.clip(b*rng.uniform(.9, 1.1)+rng.uniform(-.03, .03), 0, 1)
    return torch.from_numpy(np.ascontiguousarray(b[None]))


class Views(Dataset):
    def __init__(self, indices, n, epoch, seed):
        self.indices, self.n, self.epoch, self.seed = indices, n, epoch, seed
        self.raw = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, row):
        import cv2
        cv2.setNumThreads(1)
        if self.raw is None:
            self.raw = np.memmap(RAW, mode='r', dtype=np.uint8, shape=(self.n, 972, 1296))
        ix = int(self.indices[row])
        a = cv2.createCLAHE(2., (1296//32, 972//32)).apply(np.asarray(self.raw[ix]))
        lo, hi = np.percentile(a, (1, 99))
        a = np.clip((a.astype(np.float32)-lo)/(hi-lo+1e-6), 0, 1)
        return torch.stack([augmented_view(a, self.seed, self.epoch, ix, v) for v in range(2)]), ix


def loader(indices, n, epoch, seed, batch=32):
    return DataLoader(Views(indices, n, epoch, seed), batch_size=batch, shuffle=False,
                      num_workers=2, pin_memory=True, prefetch_factor=1, drop_last=False)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup(seed):
    # Same setup order as the original, including construction of the discarded
    # regression head before the train-only projector; seed=1003 is compatible.
    torch.set_num_threads(2)
    seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model, _, details = foundation.build('cpdino_vitb', False)
    encoder = model[0].cuda().train()
    del model
    if details['patch_stride'] != [8, 8] or details['patch_kernel'] != [16, 16]:
        raise ValueError('Unexpected CellposeDINO geometry')
    projector = nn.Sequential(nn.Linear(768, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
                              nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
                              nn.Linear(2048, 128)).cuda().train()
    sigreg = pilot.SIGReg().cuda()
    opt = torch.optim.AdamW([{'params':encoder.parameters(), 'lr':pilot.RECIPE['encoder_lr']},
                            {'params':projector.parameters(), 'lr':pilot.RECIPE['projector_lr']}],
                           weight_decay=pilot.RECIPE['weight_decay'], fused=True)
    details.update(gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                   seed=seed, fresh_external_initialization=True,
                   encoder_parameters=sum(p.numel() for p in encoder.parameters()),
                   train_only_projector_parameters=sum(p.numel() for p in projector.parameters()))
    return encoder, projector, sigreg, opt, details


def all_rng_state():
    numpy_state = np.random.get_state()
    return dict(rng_cpu=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                rng_python=random.getstate(),
                rng_numpy=dict(kind=numpy_state[0], keys=numpy_state[1].tolist(),
                               position=numpy_state[2], has_gauss=numpy_state[3],
                               cached_gaussian=numpy_state[4]))


def restore_all_rng(state):
    torch.set_rng_state(state['rng_cpu'])
    if state['rng_cuda']:
        torch.cuda.set_rng_state_all(state['rng_cuda'])
    random.setstate(state['rng_python'])
    n = state['rng_numpy']
    np.random.set_state((n['kind'], np.array(n['keys'], dtype=np.uint32),
                         n['position'], n['has_gauss'], n['cached_gaussian']))


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal_tree(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


def tests(args):
    if args.root is not None and (args.root/'cpu_tests.json').exists():
        raise FileExistsError('Never overwrite prior CPU test evidence')
    manifest_sha = None
    if args.root is not None:
        verify(args)
        manifest_sha = digest(args.root/'manifest.json')
    pilot.tests()
    torch.set_num_threads(2)
    # Actual final-batch size for fold4 is 14. Check both 32 images and 14,
    # with a partial encoder microbatch, against retaining all graphs.
    errors = {}
    for images in (32, 14, 3):
        seed_all(19)
        encoder = nn.Sequential(nn.Linear(7, 11), nn.Dropout(.2), nn.Linear(11, 768))
        projector = nn.Sequential(nn.Linear(768, 128), nn.BatchNorm1d(128))
        encoder2, projector2 = copy.deepcopy(encoder), copy.deepcopy(projector)
        x = torch.randn(images*2, 7)
        sigreg = pilot.SIGReg()
        initial = pilot.rng_state()
        loss, _, _, _ = pilot.objective(torch.cat([encoder(p) for p in x.split(4)]), projector, sigreg)
        loss.backward()
        pilot.restore_rng(initial)
        values = pilot.cache_backward(encoder2, projector2, sigreg, x, 4, lambda e, p:e(p), True)
        assert values['actual_images'] == images and abs(loss.item()-values['loss']) < 1e-6
        errors[str(images)] = 0.
        for p, q in zip(list(encoder.parameters())+list(projector.parameters()),
                        list(encoder2.parameters())+list(projector2.parameters())):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=3e-5)
            errors[str(images)] = max(errors[str(images)], (p.grad-q.grad).abs().max().item())
    a = np.linspace(0, 1, 63, dtype=np.float32).reshape(7, 9)
    v = augmented_view(a, 1003, 0, 7, 0)
    assert torch.equal(v, augmented_view(a, 1003, 0, 7, 0))
    for changed in ((1004, 0, 7, 0), (1003, 1, 7, 0), (1003, 0, 7, 1)):
        assert not torch.equal(v, augmented_view(a, *changed))
    # Validate seed initialization and exact original augmentation at seed1003.
    rng = np.random.default_rng(np.random.SeedSequence([1003, 0, 7, 0]))
    b = a
    if rng.random() < .5: b = b[:, ::-1]
    if rng.random() < .5: b = b[::-1, :]
    b = np.clip(b*rng.uniform(.9, 1.1)+rng.uniform(-.03, .03), 0, 1)
    assert torch.equal(v, torch.from_numpy(np.ascontiguousarray(b[None])))
    seed_all(57); first = torch.rand(4)
    seed_all(57); assert torch.equal(first, torch.rand(4))
    seed_all(58); assert not torch.equal(first, torch.rand(4))
    tr, va = np.arange(64), np.arange(64, 80)
    ids = np.array([f'image{i}' for i in range(80)])
    validate_membership(ids, tr, va, va[:10])
    for bad_train, bad_val in ((tr, np.array([0, 64])), (tr.astype(float), va),
                               (np.array([0, 0]), va), (tr, np.array([64, 80]))):
        try: validate_membership(ids, bad_train, bad_val)
        except ValueError: pass
        else: raise AssertionError('Invalid split accepted')
    assert np.array_equal(np.random.default_rng(1003).permutation(tr),
                          np.random.default_rng(1003).permutation(tr))
    assert not np.array_equal(np.random.default_rng(1003).permutation(tr),
                              np.random.default_rng(1004).permutation(tr))
    # Full optimizer + Python/NumPy/Torch RNG serialization is weights_only-safe
    # and gives an identical next stochastic optimizer update.
    seed_all(99)
    net = nn.Sequential(nn.Linear(7, 9), nn.BatchNorm1d(9), nn.Dropout(.2), nn.Linear(9, 3))
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    x = torch.randn(8, 7)
    def toy_update():
        opt.zero_grad(set_to_none=True)
        loss = net(x).square().mean() * (1 + random.random() + float(np.random.rand()))
        loss.backward(); opt.step()
    toy_update()
    f = io.BytesIO()
    torch.save(dict(model=net.state_dict(), optimizer=opt.state_dict(), **all_rng_state()), f)
    f.seek(0); c = torch.load(f, weights_only=True)
    toy_update()
    expected_net, expected_opt = copy.deepcopy(net.state_dict()), copy.deepcopy(opt.state_dict())
    net.load_state_dict(c['model']); opt.load_state_dict(c['optimizer']); restore_all_rng(c)
    toy_update(); equal_tree(net.state_dict(), expected_net); equal_tree(opt.state_dict(), expected_opt)
    d = deadline_timestamp('2026-10-19T03:59:00Z')
    check_deadline(d, now=d-1)
    for now, reserve in ((d, 0), (d-10, 11)):
        try: check_deadline(d, reserve=reserve, now=now)
        except DeadlineReached: pass
        else: raise AssertionError('Expired training allowed')
    previous_alarm = signal.getsignal(signal.SIGALRM)
    try:
        arm_deadline(time.time()+.02)
        try: time.sleep(.2)
        except DeadlineReached: pass
        else: raise AssertionError('Absolute deadline signal did not interrupt work')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
    result = dict(status='passed', manifest_sha=manifest_sha, created_utc=utc_now(),
                  gradient_cache_max_abs_errors=errors, full_optimizer_rng_next_update_exact=True,
                  weights_only_checkpoint_load=True,
                  tests=['original SIGReg/cache suite', '32/14/3-image exact gradients',
                         'seed-dependent augmentation and initialization', 'seed1003 original-view parity',
                         'split indices/IDs disjoint, full validation excluded',
                         'partial primary validation accepted', 'deadline boundary and interrupt enforcement'])
    if args.root is not None:
        save(args.root/'cpu_tests.json', result)
    print(json.dumps(result), flush=True)


def smoke(args):
    _, ids, train, val = verify(args)
    if (args.root/'smoke.json').exists():
        raise FileExistsError('Never replace prior smoke evidence')
    cpu = read(args.root/'cpu_tests.json')
    if cpu['status'] != 'passed' or cpu['manifest_sha'] != digest(args.root/'manifest.json'):
        raise ValueError('CPU tests must pass for the same manifest')
    deadline = deadline_timestamp(args.deadline_utc)
    check_deadline(deadline, reserve=120)
    arm_deadline(deadline)
    enc, proj, reg, opt, details = setup(args.seed)
    order = np.random.default_rng(args.seed).permutation(train)
    # Full real batch tests projector BN/statistics and memory at production size.
    x, ix = next(iter(loader(order[:32], len(ids), 0, args.seed)))
    assert not np.intersect1d(ix.numpy(), val).size
    tracked = {n:p for n,p in enc.named_parameters() if p.ndim >= 2}
    before = {n:p.detach().clone() for n,p in tracked.items()}
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    value = pilot.update(enc, proj, reg, opt, x, True)
    changed = [n for n,p in tracked.items() if not torch.equal(before[n], p.detach())]
    blocks = lambda names:{re.search(r'blocks\.\d+', n).group() for n in names if re.search(r'blocks\.\d+', n)}
    if blocks(changed) != blocks(tracked) or not any('patch_embed' in n for n in changed):
        raise ValueError('Encoder block or patch-projection gradient missing')
    assert all(int(s['step'].item()) == 1 for s in opt.state.values())
    torch.cuda.synchronize()
    result = dict(status='passed', manifest_sha=digest(args.root/'manifest.json'), values=value,
                  encoder_blocks_updated=sorted(blocks(changed)), optimizer_step_after_update=1,
                  production_batch_images=len(ix), seed=args.seed,
                  training_requires_fresh_setup=True, smoke_checkpoint_saved=False,
                  validation_images_used=0, target_arrays_loaded=False,
                  seconds=time.monotonic()-started, peak_gib=torch.cuda.max_memory_allocated()/2**30,
                  details=details)
    save(args.root/'smoke.json', result)
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(result), flush=True)


def save_checkpoint(run, enc, proj, opt, epoch, step, manifest_sha, epoch_step, order, seen):
    temporary = run/'last.pt.tmp'
    torch.save(dict(encoder=enc.original.state_dict(), projector=proj.state_dict(),
                    optimizer=opt.state_dict(), epoch=epoch, step=step,
                    step_this_epoch=epoch_step, manifest_sha=manifest_sha,
                    order=torch.from_numpy(order.copy()), seen_this_epoch=seen,
                    **all_rng_state()), temporary)
    temporary.replace(run/'last.pt')


def train(args):
    m, ids, train_indices, val = verify(args)
    run = args.root/'run'
    if run.exists():
        raise FileExistsError('No overwrite or implicit resume; retain prior run for audit')
    manifest_sha = digest(args.root/'manifest.json')
    for name in ('cpu_tests.json', 'smoke.json'):
        check = read(args.root/name)
        if check['status'] != 'passed' or check['manifest_sha'] != manifest_sha:
            raise ValueError(f'Unverified prerequisite: {name}')
    deadline = deadline_timestamp(args.deadline_utc)
    check_deadline(deadline, reserve=120)
    run.mkdir()
    history, epochs, global_step = [], [], 0
    started, reference_std = time.monotonic(), None
    durable = None
    arm_deadline(deadline)
    try:
        enc, proj, reg, opt, details = setup(args.seed)
        if opt.state:
            raise ValueError('Fresh training must start with an empty optimizer state')
        save(run/'progress.json', dict(status='starting', manifest_sha=manifest_sha,
                                       epochs=args.epochs, total_steps=m['total_steps'], details=details))
        for epoch in range(1, args.epochs+1):
            order = np.random.default_rng(args.seed+epoch-1).permutation(train_indices)
            dl = loader(order, len(ids), epoch-1, args.seed)
            seen, epoch_start = [], time.monotonic()
            for step, (x, ix) in enumerate(dl, 1):
                check_deadline(deadline, reserve=90)
                assert not np.intersect1d(ix.numpy(), val).size
                next_step = global_step+1
                ratio = min(1., next_step/pilot.RECIPE['warmup_steps'])
                opt.param_groups[0]['lr'] = pilot.RECIPE['encoder_lr']*ratio
                opt.param_groups[1]['lr'] = pilot.RECIPE['projector_lr']*ratio
                tick = time.monotonic(); torch.cuda.reset_peak_memory_stats()
                values = pilot.update(enc, proj, reg, opt, x, step == 1)
                torch.cuda.synchronize(); global_step += 1; seen.extend(ix.tolist())
                if reference_std is None:
                    reference_std = values['feature_std']
                if values['feature_std'] < max(1e-6, reference_std*1e-3) or values['projector_std'] < 1e-6:
                    raise ValueError('Representation collapse tripwire; no automatic retry')
                item = dict(status='ssl_training', epoch=epoch, epochs=args.epochs,
                            step_this_epoch=step, steps_this_epoch=len(dl), global_step=global_step,
                            total_steps=m['total_steps'], seen_this_epoch=len(seen), train_n=len(train_indices),
                            seconds_step=time.monotonic()-tick, elapsed_seconds=time.monotonic()-started,
                            peak_gib=torch.cuda.max_memory_allocated()/2**30,
                            encoder_lr=opt.param_groups[0]['lr'], projector_lr=opt.param_groups[1]['lr'], **values)
                history.append(item); save(run/'progress.json', item)
                if step == 1 or step % 5 == 0:
                    print(json.dumps(item), flush=True)
                if step % 50 == 0 or step == len(dl):
                    save_checkpoint(run, enc, proj, opt, epoch, global_step, manifest_sha, step, order, len(seen))
                    durable = dict(epoch=epoch, step=global_step, step_this_epoch=step)
                    save(run/'history.json', history)
            np.testing.assert_array_equal(np.array(seen, dtype=np.int64), order)
            assert set(seen) == set(train_indices.tolist())
            assert global_step == epoch*math.ceil(len(train_indices)/32)
            checkpoint = run/f'epoch{epoch:02d}.pt'
            # A hard link preserves each completed epoch when last.pt is atomically replaced.
            os.link(run/'last.pt', checkpoint)
            checkpoint_sha = digest(checkpoint)
            record = dict(epoch=epoch, unique_train_images=len(set(seen)), train_images_seen=len(seen),
                          validation_images_used=0, order_sha=hashlib.sha256(order.tobytes()).hexdigest(),
                          train_ids_sha=id_sha(ids, order), seconds=time.monotonic()-epoch_start,
                          checkpoint=str(checkpoint), checkpoint_sha=checkpoint_sha, last=history[-1])
            epochs.append(record); save(run/'epochs.json', epochs)
            print(json.dumps(dict(epoch_complete=epoch, **record)), flush=True)
        if global_step != m['total_steps']:
            raise ValueError('Incomplete planned SSL steps')
        result = dict(status='SSL_COMPLETE_NOT_ACCURACY_VALIDATED', epochs=args.epochs,
                      total_steps=global_step, steps=global_step, train_n=len(train_indices),
                      train_image_presentations=args.epochs*len(train_indices),
                      unique_train_images=len(train_indices), validation_images_used=0,
                      target_arrays_loaded=False, fold=args.fold, seed=args.seed,
                      train_ids_sha=m['train_ids_sha'], validation_ids_sha=m['validation_ids_sha'],
                      epochs_detail=epochs, seconds=time.monotonic()-started, details=details,
                      manifest_sha=manifest_sha, checkpoint_sha=digest(run/'last.pt'),
                      limitations=m['limitations'], finished_utc=utc_now())
        save(run/'result.json', result)
        save(run/'complete.json', dict(status='completed', result_sha=digest(run/'result.json'),
                                      checkpoint_sha=result['checkpoint_sha'], manifest_sha=manifest_sha))
        save(run/'progress.json', dict(status='completed', epochs=args.epochs, global_step=global_step))
        print(json.dumps(result), flush=True)
    except DeadlineReached as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        stopped = dict(status='deadline_stopped', deadline_utc=args.deadline_utc, stopped_utc=utc_now(),
                       manifest_sha=manifest_sha, completed_epochs=len(epochs), observed_global_step=global_step,
                       durable_checkpoint=durable, reason=str(exc),
                       checkpoint_sha=digest(run/'last.pt') if (run/'last.pt').exists() else None,
                       note='Last durable checkpoint retained; interrupted optimizer state is discarded')
        save(run/'deadline_stop.json', stopped); save(run/'progress.json', stopped)
        print(json.dumps(stopped), flush=True)
        return 3
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'tests', 'smoke', 'train'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--split', type=Path)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--data-preflight', type=Path,
                        default=Path('experiments/phase2_crossfold_20261015/data_preflight.json'))
    parser.add_argument('--seed', type=int, default=1003)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--deadline-utc', default='2026-10-19T03:59:00Z')
    args = parser.parse_args()
    if args.command != 'tests' or args.root is not None:
        if args.root is None or args.split is None or args.fold is None:
            parser.error('--root, --split and --fold are required for data-bound commands')
    if args.seed < 0 or args.epochs < 1:
        parser.error('seed must be nonnegative; epochs must be positive')
    deadline_timestamp(args.deadline_utc)
    if args.command in ('smoke', 'train') and torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible GPU is required for SSL')
    result = globals()[args.command](args)
    return result if isinstance(result, int) else 0


if __name__ == '__main__':
    raise SystemExit(main())
