"""NumPy-only contract audit for a declared log-L1 coefficient ablation.

compare requires an independently reviewed adapter SHA; no candidate Python is
imported. Existing arm audits remain responsible for model-state readback.
fixture-tests uses synthetic JSON/NPZ and byte-only checkpoint stand-ins, never
trains/loads a model, and is not scientific evidence about an actual experiment.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np

ORIGINAL_SOURCES = {
    'phase2_crossfold_regression_20261015.py': '7289b583643c9c74ad300f73e2a0d6debbe38be5d6393b03508437cdc8232ffe',
    'phase2_foundation_train_20261002.py': '20a849cb607c3c5777102e3266c6c1b5d9ba4533f9bfac9f1f6e9d18232dac2f',
    'phase2_foundation_preflight_20261002.py': '196500b67d01ad9423ea4d7d8c1a1e4b2d886df2328d36bfb1e1c94058ca4b1e',
    'phase2_lejepa_regression_20261003.py': 'f66fe903c2360c6e2392b407c55392dc1d7a45a03e624bed0acee8cbc32b33b8',
}
ADAPTER_NAME = 'phase2_crossfold_loss_ablation_20261016.py'
ORIGINAL_OBJECTIVE = '0.5 standardized log L1 + 0.5 physical MAPE / 0.2'
LOSS_NAME = 'standardized_log_l1_plus_physical_relative'
LOSS_KEYS = {'objective', 'loss_name', 'standardized_log_l1_coefficient', 'relative_error_coefficient'}
METRICS = ('CD', 'CV', 'HEX', 'mean')
SCORE_ATOL = 1e-4  # Frozen before candidate observations; percentage points, no relative tolerance.
COMMON = (
    'architecture', 'fold', 'seed', 'split_sha256', 'train_n', 'validation_n',
    'images', 'train_ids_sha', 'validation_ids_sha', 'train_order_base_sha',
    'train_indices', 'validation_indices', 'historical_validation_indices',
    'historical_validation_n', 'excluded_from_split_indices', 'log_mean', 'log_std',
    'raw_sha256', 'raw_bytes', 'expected_head_sha', 'mode', 'execution', 'diagnostics',
    'primary_validation', 'historical_subset_role', 'ssl_manifest_sha',
    'ssl_checkpoint_sha', 'ssl_encoder_tensor_sha', 'ssl_epochs', 'ssl_seed', 'ssl_steps',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def exact_contract(actual, expected):
    # Unlike dict equality this also distinguishes JSON false from numeric zero.
    return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)


def checked(path, expected, inputs):
    path = Path(path)
    actual = inputs.get(str(path))
    if actual is None:
        actual = sha(path)
    require(valid_sha(expected) and actual == expected, 'Changed artifact: ' + str(path))
    inputs[str(path)] = actual
    return actual


def bound(path, manifest, inputs):
    path = Path(path)
    hashes = {value for name, value in manifest['hashes'].items()
              if Path(name).resolve() == path.resolve()}
    require(len(hashes) == 1, 'Input missing or ambiguous in manifest: ' + str(path))
    return checked(path, next(iter(hashes)), inputs)


def ids_sha(ids):
    return hashlib.sha256('\n'.join(np.asarray(ids).astype(str)).encode()).hexdigest()


def array_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def mape64(pred, target):
    require(pred.shape == target.shape and target.ndim == 2 and target.shape[1] == 3
            and len(target) > 0, 'Predictions and targets must be matching nonempty N by 3')
    require(np.isfinite(pred).all() and (pred > 0).all()
            and np.isfinite(target).all() and (target > 0).all(), 'Invalid physical predictions/targets')
    p, y = pred.astype(np.float64), target.astype(np.float64)
    values = np.mean(np.abs(p-y)/y, axis=0)*100
    return dict(zip(METRICS, map(float, [*values, values.mean()])), n=len(y))


def same_score(actual, expected, context):
    require(type(expected.get('n')) is int and actual['n'] == expected['n'], context + ': n differs')
    for key in METRICS:
        require(key in expected and math.isfinite(expected[key])
                and abs(actual[key]-expected[key]) <= SCORE_ATOL, context + ': score differs: ' + key)


def source_contract(root, manifest, inputs, adapter_sha=None):
    for name, expected in ORIGINAL_SOURCES.items():
        require(bound(root/'source'/name, manifest, inputs) == expected, 'Original source pin differs: ' + name)
        matching = [(path, value) for path, value in manifest['hashes'].items() if Path(path).name == name]
        require(matching and all(value == expected for _, value in matching), 'Conflicting original source pin: ' + name)
    if adapter_sha is not None:
        require(bound(root/'source'/ADAPTER_NAME, manifest, inputs) == adapter_sha, 'Adapter differs from independent CLI pin')
        matching = [value for path, value in manifest['hashes'].items() if Path(path).name == ADAPTER_NAME]
        require(matching and all(value == adapter_sha for value in matching), 'Conflicting adapter source pin')


def data_contract(manifest, inputs):
    split = Path(manifest['split_path'])
    require(bound(split, manifest, inputs) == manifest['split_sha256'], 'Split hash declaration differs')
    meta_paths = {Path(path).resolve() for path in manifest['hashes']
                  if Path(path).name == 'cache_native_meta_512.npz'}
    require(len(meta_paths) == 1, 'Missing or ambiguous original metadata')
    meta = next(iter(meta_paths)); bound(meta, manifest, inputs)
    # The exact original metadata bytes are already bound above. Some historical
    # ID arrays use NumPy object dtype; no candidate Python module is imported.
    with np.load(meta, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    require(ids.ndim == 1 and len(ids) == manifest['images'] and len(np.unique(ids)) == len(ids), 'Metadata IDs differ')
    require(y.shape == (len(ids), 3), 'Metadata target shape differs')
    with np.load(split, allow_pickle=False) as z:
        train, val = z['train'].copy(), z['validation'].copy()
        historical = z['primary_validation'].copy() if 'primary_validation' in z else val.copy()
    for name, ix in (('train', train), ('validation', val)):
        require(ix.ndim == 1 and np.issubdtype(ix.dtype, np.integer) and len(ix) > 0
                and len(np.unique(ix)) == len(ix) and ix.min() >= 0 and ix.max() < len(ids), 'Invalid ' + name + ' indices')
        require(len(ix) == manifest[name+'_n'] and ix.tolist() == manifest[name+'_indices'], name + ' membership differs')
        require(ids_sha(ids[ix]) == manifest[name+'_ids_sha'], name + ' ID digest differs')
    require(not np.intersect1d(train, val).size, 'Train/validation overlap')
    require(array_sha(train) == manifest['train_order_base_sha'], 'Train order digest differs')
    require(historical.tolist() == manifest['historical_validation_indices']
            and len(historical) == manifest['historical_validation_n'] and len(historical) > 0
            and len(np.unique(historical)) == len(historical) and np.isin(historical, val).all(), 'Historical subset differs')
    require(np.setdiff1d(np.arange(len(ids)), np.r_[train, val]).tolist() == manifest['excluded_from_split_indices'], 'Excluded membership differs')
    require(np.isfinite(y[np.r_[train, val]]).all() and (y[np.r_[train, val]] > 0).all(), 'Positive target protocol differs')
    logs = np.log(y[train].astype(np.float64))
    require(np.array_equal(logs.mean(0), manifest['log_mean'])
            and np.array_equal(logs.std(0), manifest['log_std']) and (logs.std(0) > 0).all(), 'Train-only log statistics differ')
    return ids, y, train, val, historical


def prediction_file(path, manifest, data):
    ids, y, _, val, _ = data
    with np.load(path, allow_pickle=False) as z:
        fields = {key: z[key].copy() for key in ('indices', 'ids', 'target', 'pred')}
    n = manifest['validation_n']
    require(fields['pred'].shape == fields['target'].shape == (n, 3), 'Prediction matrix shape differs')
    require(fields['indices'].shape == fields['ids'].shape == (n,), 'Prediction identities shape differs')
    require(np.issubdtype(fields['indices'].dtype, np.integer), 'Prediction indices must be integers')
    for key, expected in (('indices', val), ('ids', ids[val]), ('target', y[val])):
        require(np.array_equal(fields[key], expected), 'Predictions differ from original validation ' + key)
    return fields, mape64(fields['pred'], fields['target'])


def ssl_contract(manifest, inputs):
    root = Path(manifest['ssl_root'])
    for name in ('manifest.json', 'run/result.json', 'run/complete.json', 'run/last.pt'):
        bound(root/name, manifest, inputs)
    sm, sr, complete = read(root/'manifest.json'), read(root/'run/result.json'), read(root/'run/complete.json')
    require(sha(root/'manifest.json') == manifest['ssl_manifest_sha'], 'Declared SSL manifest hash differs')
    checked(root/'run/result.json', complete['result_sha'], inputs)
    checked(root/'run/last.pt', manifest['ssl_checkpoint_sha'], inputs)
    require(complete['status'] == 'completed' and sr['manifest_sha'] == manifest['ssl_manifest_sha']
            and sr['checkpoint_sha'] == manifest['ssl_checkpoint_sha'], 'SSL completion chain differs')
    for key in ('fold', 'split_sha256', 'train_ids_sha', 'validation_ids_sha', 'train_n', 'raw_sha256', 'raw_bytes'):
        require(sm[key] == manifest[key], 'SSL provenance differs: '+key)
    require(sm['val_n'] == manifest['validation_n'] and sm['total_images'] == manifest['images']
            and sm['seed'] == manifest['ssl_seed'], 'SSL population/seed differs')
    require(sr['epochs'] == 5 and sr['steps'] == manifest['ssl_steps'] == math.ceil(manifest['train_n']/32)*5
            and sr['validation_images_used'] == 0 and len(sr['epochs_detail']) == 5, 'SSL duration/coverage differs')
    for epoch, item in enumerate(sr['epochs_detail'], 1):
        require(item['epoch'] == epoch and item['unique_train_images'] == item['train_images_seen'] == manifest['train_n']
                and item['validation_images_used'] == 0, 'SSL epoch coverage differs')


def arm(root, label, inputs):
    root = Path(root)
    m, audit = read(root/'manifest.json'), read(root/'audit.json')
    msha = sha(root/'manifest.json'); inputs[str(root/'manifest.json')] = msha
    inputs[str(root/'audit.json')] = sha(root/'audit.json')
    require(m['architecture'] == 'cpdino_vitb' and m['mode'] == 'full', 'Unexpected architecture/mode')
    require(m['recipe']['epochs'] == 20 and m['recipe']['seed'] == m['seed'], 'Expected matched terminal20/seed recipe')
    require(m['recipe']['effective_batch'] == 16 and m['recipe']['micro_batch'] == 4, 'Batch recipe differs')
    require(m['ssl_epochs'] == 5, 'Requires the same verified SSL5 initialization')
    require(m['hashes'], 'Frozen input hashes absent')
    for path, expected in m['hashes'].items():
        checked(Path(path), expected, inputs)
    data = data_contract(m, inputs)
    ssl_contract(m, inputs)
    require(audit['status'] == 'single_model_crossfold_regression_verified' and audit['manifest_sha'] == msha, 'Missing bound arm audit')
    for key in ('fold', 'seed', 'recipe'):
        require(audit[key] == m[key], 'Arm audit differs: ' + key)
    require(audit['single_model'] and audit['all_validation_images_scored']
            and audit['ssl_split_exact_match'] and audit['ssl_train_only'], 'Incomplete individual arm audit')
    cpu, smoke = read(root/'cpu_tests.json'), read(root/'smoke.json')
    for name, report in (('cpu_tests', cpu), ('smoke', smoke)):
        require(report['status'] == 'passed' and report['manifest_sha'] == msha, 'Unverified ' + name)
        inputs[str(root/(name+'.json'))] = sha(root/(name+'.json'))
    require(cpu['partial_ddp_gradient_equivalence'] and cpu['actual_head_seed_verified']
            and cpu['order_seed_verified'] and cpu['metric_identity'], 'Missing CPU recipe evidence')
    require(smoke['ranks_synchronized'] and smoke['all_12_encoder_blocks_updated']
            and smoke['patch_gradient_verified'] and smoke['head_updated']
            and smoke['single_gpu_reference_gradient_check']['passed'], 'Missing actual smoke evidence')
    gradient = smoke['single_gpu_reference_gradient_check']
    require(math.isfinite(gradient['relative_gradient_l2']) and 0 <= gradient['relative_gradient_l2'] < .002
            and gradient['includes_final9_samples_and_zero_weight_dummy'] is True, 'Invalid smoke gradient evidence')
    run = root/'runs'/f'fold{m["fold"]}'/'cpdino_vitb'
    result, complete = read(run/'result.json'), read(run/'complete.json')
    require(complete['status'] == 'completed', 'Arm incomplete')
    checked(run/'result.json', complete['result_sha'], inputs)
    inputs[str(run/'complete.json')] = sha(run/'complete.json')
    require(result['manifest_sha'] == msha and result['architecture'] == m['architecture']
            and result['fold'] == m['fold'] and result['seed'] == m['seed'], 'Result identity differs')
    require(result['single_model'] and result['ranks_synchronized'], 'Result is not one synchronized model')
    checked(run/'last.pt', result['checkpoint_sha'], inputs)
    checked(run/'predictions.npz', result['predictions_sha'], inputs)
    fields, independent_score = prediction_file(run/'predictions.npz', m, data)
    for name, value in (('result', result['score']), ('audit', audit['score'])):
        same_score(independent_score, value, label+' '+name)
    epochs = m['recipe']['epochs']; train, val, historical = data[2:]
    require(len(result['history']) == len(audit['epoch_artifacts']) == epochs, 'Incomplete epoch evidence')
    updates = math.ceil(len(train)/16)
    for epoch, (row, artifact) in enumerate(zip(result['history'], audit['epoch_artifacts']), 1):
        require(row['epoch'] == artifact['epoch'] == epoch and row['updates'] == epoch*updates, 'Epoch/update mismatch')
        expected_order = np.random.default_rng(m['seed']+epoch-1).permutation(train)
        require(row['order_sha'] == array_sha(expected_order), 'Seeded training order differs')
        same_score(row['score'], artifact['score'], label+' epoch audit')
        require(row['score']['n'] == len(val), 'Epoch validation population differs')
    same_score(independent_score, result['history'][-1]['score'], label+' terminal history')
    if (run/'history.json').exists():
        require(read(run/'history.json') == result['history'], 'History sidecar differs')
        inputs[str(run/'history.json')] = sha(run/'history.json')
    terminal = audit['epoch_artifacts'][-1]
    checked(run/f'epoch_{epochs:02d}.pt', terminal['weights_sha'], inputs)
    final_pred = run/f'predictions_epoch_{epochs:02d}.npz'
    checked(final_pred, terminal['predictions_sha'], inputs)
    final_fields, _ = prediction_file(final_pred, m, data)
    require(all(np.array_equal(fields[k], final_fields[k]) for k in fields), 'Terminal and final predictions differ')
    positions = {int(index): row for row, index in enumerate(val)}
    take = [positions[int(index)] for index in historical]
    hs = mape64(fields['pred'][take], fields['target'][take])
    same_score(hs, result['historical_subset_score'], label+' historical result')
    same_score(hs, audit['historical_subset_score'], label+' historical audit')
    for details in (result['details'], smoke['details']):
        require(details['head_initial_tensor_sha'] == m['expected_head_sha']
                and details['encoder_tensor_sha'] == m['ssl_encoder_tensor_sha']
                and details['ssl_checkpoint_sha'] == m['ssl_checkpoint_sha'] and details['ssl_epochs'] == 5,
                'Loaded initialization differs')
    return dict(root=root, manifest=m, audit=audit, cpu=cpu, smoke=smoke, result=result,
                complete=complete, fields=fields, independent_score=independent_score,
                independent_historical_score=hs)


def loss_contract(records, expected_adapter_sha, expected_coefficient, inputs):
    require(valid_sha(expected_adapter_sha), 'An independently reviewed lowercase adapter SHA256 is required')
    require(not isinstance(expected_coefficient, bool) and math.isfinite(expected_coefficient)
            and 0 <= expected_coefficient < .5, 'Expected candidate coefficient must be in [0,0.5)')
    baseline, candidate = records['baseline'], records['candidate']
    left, right = baseline['manifest'], candidate['manifest']
    source_contract(baseline['root'], left, inputs)
    source_contract(candidate['root'], right, inputs, expected_adapter_sha)
    require(left['recipe']['objective'] == ORIGINAL_OBJECTIVE and 'loss_ablation' not in left
            and not (set(left['recipe']) & (LOSS_KEYS-{'objective'})), 'Baseline is not the pinned original mixed-loss recipe')
    recipe = right['recipe']
    require(type(recipe['standardized_log_l1_coefficient']) in (int, float)
            and recipe['loss_name'] == LOSS_NAME and recipe['standardized_log_l1_coefficient'] == expected_coefficient
            and recipe['relative_error_coefficient'] == 2.5
            and recipe['objective'] == f'{expected_coefficient:g} standardized log L1 + 2.5 physical relative error',
            'Candidate declared loss differs')
    expected = dict(adapter_source_sha256=expected_adapter_sha,
                    base_runner_sha256=ORIGINAL_SOURCES['phase2_crossfold_regression_20261015.py'],
                    original_source_sha256=ORIGINAL_SOURCES, loss_name=LOSS_NAME,
                    standardized_log_l1_coefficient=expected_coefficient,
                    relative_error_coefficient=2.5, active_loss_verified=True)
    declaration = right['loss_ablation']
    require(declaration['isolated_root'] == str(candidate['root'].resolve()), 'Candidate isolated root differs')
    require(declaration['adapter_source_sha256'] == expected_adapter_sha
            and declaration['original_source_sha256'] == ORIGINAL_SOURCES
            and declaration['original_objective'] == ORIGINAL_OBJECTIVE
            and exact_contract(declaration['required_execution_contract'], expected), 'Manifest loss execution contract differs')
    for name, evidence in (
            ('result', candidate['result']['details']), ('smoke', candidate['smoke']['details']),
            ('CPU', candidate['cpu']), ('audit', candidate['audit'])):
        require(exact_contract(evidence.get('loss_execution'), expected), 'Missing or differing actual loss execution: ' + name)
    tests = candidate['cpu']['loss_model_free_tests']
    require(tests['status'] == 'passed' and tests['adapter_source_sha256'] == expected_adapter_sha
            and tests['original_source_sha256'] == ORIGINAL_SOURCES and tests['model_constructed'] is False
            and tests['cuda_initialized'] is False, 'Missing or differently sourced loss CPU contract tests')
    require(candidate['audit']['audit_type'] == 'single_model_loss_coefficient_ablation'
            and candidate['audit']['loss_contract_verified'] is True, 'Missing dedicated candidate loss audit')
    run = candidate['root']/'runs'/f'fold{right["fold"]}'/'cpdino_vitb'
    audit_bindings = {'result_sha': run/'result.json', 'complete_sha': run/'complete.json',
                      'checkpoint_sha': run/'last.pt', 'predictions_sha': run/'predictions.npz',
                      'cpu_tests_sha': candidate['root']/'cpu_tests.json', 'smoke_sha': candidate['root']/'smoke.json'}
    for key, path in audit_bindings.items():
        require(candidate['audit'].get(key) == inputs[str(path)], 'Candidate audit does not bind execution artifact: ' + key)
    for key in COMMON:
        require(left[key] == right[key], 'Pair mismatch: ' + key)
    for key in (set(left['recipe']) | set(recipe))-LOSS_KEYS:
        require(key in left['recipe'] and key in recipe and left['recipe'][key] == recipe[key], 'Non-loss recipe differs: ' + key)
    for key in ('split_path', 'ssl_root'):
        require(Path(left[key]).resolve() == Path(right[key]).resolve(), 'Pair path differs: ' + key)
    for a, b in zip(baseline['result']['history'], candidate['result']['history']):
        require(all(a[key] == b[key] for key in ('epoch', 'updates', 'order_sha')), 'Pair epoch/order differs')
    for key in ('indices', 'ids', 'target'):
        require(np.array_equal(baseline['fields'][key], candidate['fields'][key]), 'Pair validation differs: ' + key)
    return dict(
        baseline=dict(loss_name=LOSS_NAME, standardized_log_l1_coefficient=.5,
                      relative_error_coefficient=2.5, interpretation='Exact original objective plus all four pinned sources'),
        candidate=expected)


def compare(baseline, candidate, expected_adapter_sha256, expected_candidate_coefficient=0.0):
    baseline, candidate = Path(baseline), Path(candidate)
    require(baseline.resolve() != candidate.resolve(), 'Two distinct arm roots required')
    inputs = {}
    records = {label: arm(root, label, inputs) for label, root in [('baseline', baseline), ('candidate', candidate)]}
    contracts = loss_contract(records, expected_adapter_sha256, expected_candidate_coefficient, inputs)
    # Existing curve analyzer compares reported scores at 1e-6. Preserve those
    # verified values and separately publish independent float64 arithmetic.
    scores = {label: record['result']['score'] for label, record in records.items()}
    independent = {label: record['independent_score'] for label, record in records.items()}
    delta = lambda values: {key: values['candidate'][key]-values['baseline'][key] for key in METRICS}
    configurations = {}
    for label, record in records.items():
        recipe = record['manifest']['recipe']
        configurations[label] = {key: recipe[key] for key in ('head_lr', 'gradient_clip', 'epochs', 'schedule_epochs', *sorted(LOSS_KEYS)) if key in recipe}
    return dict(status='matched_single_model_pair_verified', comparison_type='standardized_log_l1_coefficient_ablation',
                checked_utc=time.strftime('%FT%TZ', time.gmtime()), fold=records['baseline']['manifest']['fold'],
                seed=records['baseline']['manifest']['seed'], primary='fixed terminal epoch20',
                scores=scores, candidate_minus_baseline=delta(scores), independent_scores_float64=independent,
                independent_candidate_minus_baseline_float64=delta(independent),
                score_verification=dict(dtype='float64', absolute_tolerance_percentage_points=SCORE_ATOL,
                                        relative_tolerance=0., no_tolerance_override=True),
                lower_is_better=True, ranking=sorted(scores, key=lambda k:(scores[k]['mean'], k)),
                independent_float64_ranking=sorted(independent, key=lambda k:(independent[k]['mean'], k)),
                same_ssl_head_initialization_data_order_and_validation=True, single_model_only=True,
                configurations=configurations, canonical_loss_contracts=contracts,
                only_declared_difference='standardized_log_l1_coefficient; objective label follows that scalar',
                expected_adapter_sha256=expected_adapter_sha256,
                retrospective_best={k:r['audit']['retrospective_best'] for k,r in records.items()},
                historical_subset_scores={k:r['result']['historical_subset_score'] for k,r in records.items()},
                training_seconds={k:r['result']['seconds'] for k,r in records.items()},
                source_sha256=sha(__file__), inputs=inputs, candidate_code_imported=False, model_state_loaded=False,
                automatic_followup_selected=False, automatic_launch=False,
                limitations=['This comparison checks artifacts and declared execution evidence; it does not rerun training or model forward.',
                             'Individual arm audits check checkpoint artifacts; any final selected model additionally requires its own strict reload and forward readback.',
                             'Removing log loss also changes gradient scale/clipping frequency; this is part of the loss intervention.',
                             'Online total loss values have different objectives and are not comparable between arms.',
                             'Reused development fold; terminal20 is primary and intermediate minima are retrospective diagnostics.'])


def _fixture(directory, change=None, alpha=0.0):
    """Small synthetic contracts only. Original sources are bytes, never imported."""
    directory = Path(directory); directory.mkdir()
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(value, stream, indent=2); stream.write('\n')
    ids = np.array([f'image{i:02d}' for i in range(20)])
    y = np.arange(60, dtype=np.float32).reshape(20, 3)/100+1
    tr, va = np.arange(17, dtype=np.int64), np.arange(17,20,dtype=np.int64)
    meta = directory/'cache_native_meta_512.npz'; np.savez(meta, ids=ids, Y=y)
    split = directory/'split.npz'; np.savez(split, train=tr, validation=va, primary_validation=va[:2])
    ssl_root=directory/'ssl'; (ssl_root/'run').mkdir(parents=True)
    (ssl_root/'run/last.pt').write_bytes(b'synthetic SSL checkpoint, not a model')
    ssl_manifest=dict(fold=0,split_sha256=sha(split),train_ids_sha=ids_sha(ids[tr]),validation_ids_sha=ids_sha(ids[va]),
                      train_n=len(tr),val_n=len(va),total_images=len(ids),raw_sha256='a'*64,raw_bytes=20*972*1296,seed=1003)
    save(ssl_root/'manifest.json',ssl_manifest)
    ssl_result=dict(manifest_sha=sha(ssl_root/'manifest.json'),checkpoint_sha=sha(ssl_root/'run/last.pt'),epochs=5,steps=5,
                    validation_images_used=0,epochs_detail=[dict(epoch=e,unique_train_images=17,train_images_seen=17,validation_images_used=0) for e in range(1,6)])
    save(ssl_root/'run/result.json',ssl_result)
    save(ssl_root/'run/complete.json',dict(status='completed',result_sha=sha(ssl_root/'run/result.json')))
    frozen = directory/'original'; frozen.mkdir()
    for name, expected in ORIGINAL_SOURCES.items():
        source = Path(__file__).parent/name
        require(sha(source) == expected, 'Fixture requires adjacent original source pin: '+name)
        (frozen/name).write_bytes(source.read_bytes())
    adapter_bytes = b'# Synthetic comparison fixture only; no model or training code.\n'
    adapter_sha = hashlib.sha256(adapter_bytes).hexdigest()
    marker = dict(adapter_source_sha256=adapter_sha, base_runner_sha256=ORIGINAL_SOURCES['phase2_crossfold_regression_20261015.py'],
                  original_source_sha256=ORIGINAL_SOURCES, loss_name=LOSS_NAME,
                  standardized_log_l1_coefficient=alpha, relative_error_coefficient=2.5, active_loss_verified=True)
    recipe = dict(epochs=20,schedule_epochs=20,seed=42,micro_batch=4,effective_batch=16,encoder_lr=1e-5,
                  head_lr=1e-3,gradient_clip=1.,weight_decay=1e-4,warmup_epochs=1,minimum_lr_ratio=.1,
                  objective=ORIGINAL_OBJECTIVE,precision='bfloat16',head='unchanged fixture head',view='unchanged fixture view',normalization='unchanged fixture normalization')
    logs=np.log(y[tr].astype(np.float64)); manifests={}
    for label in ('baseline','candidate'):
        root=directory/label; (root/'source').mkdir(parents=True)
        hashes={str(meta):sha(meta),str(split):sha(split)}
        for name in ('manifest.json','run/result.json','run/complete.json','run/last.pt'):
            hashes[str(ssl_root/name)]=sha(ssl_root/name)
        for name in ORIGINAL_SOURCES:
            dest=root/'source'/name; dest.write_bytes((frozen/name).read_bytes());hashes[str(dest)]=sha(dest)
        m=dict(architecture='cpdino_vitb',fold=0,seed=42,split_path=str(split),split_sha256=sha(split),
               train_n=len(tr),validation_n=len(va),images=len(ids),train_ids_sha=ids_sha(ids[tr]),validation_ids_sha=ids_sha(ids[va]),
               train_order_base_sha=array_sha(tr),train_indices=tr.tolist(),validation_indices=va.tolist(),
               historical_validation_indices=va[:2].tolist(),historical_validation_n=2,excluded_from_split_indices=[],
               log_mean=logs.mean(0).tolist(),log_std=logs.std(0).tolist(),raw_sha256='a'*64,raw_bytes=20*972*1296,
               expected_head_sha='b'*64,mode='full',execution=dict(world_size=2),diagnostics=dict(train_probe_n=17,train_probe_seed=12012),
               primary_validation='all full validation',historical_subset_role='secondary only',ssl_root=str(ssl_root),
               ssl_manifest_sha=sha(ssl_root/'manifest.json'),ssl_checkpoint_sha=sha(ssl_root/'run/last.pt'),ssl_encoder_tensor_sha='d'*64,
               ssl_epochs=5,ssl_seed=1003,ssl_steps=5,recipe=copy.deepcopy(recipe),hashes=hashes)
        if label=='candidate':
            dest=root/'source'/ADAPTER_NAME;dest.write_bytes(adapter_bytes);m['hashes'][str(dest)]=adapter_sha
            m['recipe'].update(loss_name=LOSS_NAME,standardized_log_l1_coefficient=alpha,relative_error_coefficient=2.5,
                               objective=f'{alpha:g} standardized log L1 + 2.5 physical relative error')
            m['loss_ablation']=dict(isolated_root=str(root.resolve()),adapter_source_sha256=adapter_sha,
                                   original_source_sha256=ORIGINAL_SOURCES,original_objective=ORIGINAL_OBJECTIVE,
                                   required_execution_contract=copy.deepcopy(marker))
        if change: change('manifest',label,m)
        save(root/'manifest.json',m);msha=sha(root/'manifest.json');manifests[label]=m
        details=dict(head_initial_tensor_sha=m['expected_head_sha'],encoder_tensor_sha=m['ssl_encoder_tensor_sha'],
                     ssl_checkpoint_sha=m['ssl_checkpoint_sha'],ssl_epochs=5)
        if label=='candidate':details['loss_execution']=copy.deepcopy(marker)
        cpu=dict(status='passed',manifest_sha=msha,partial_ddp_gradient_equivalence=True,actual_head_seed_verified=True,
                 order_seed_verified=True,metric_identity=True)
        if label=='candidate':cpu.update(loss_execution=copy.deepcopy(marker),loss_model_free_tests=dict(status='passed',
             adapter_source_sha256=adapter_sha,original_source_sha256=ORIGINAL_SOURCES,model_constructed=False,cuda_initialized=False))
        smoke=dict(status='passed',manifest_sha=msha,details=copy.deepcopy(details),ranks_synchronized=True,
                   all_12_encoder_blocks_updated=True,patch_gradient_verified=True,head_updated=True,
                   single_gpu_reference_gradient_check=dict(passed=True,relative_gradient_l2=0.,includes_final9_samples_and_zero_weight_dummy=True))
        if change:change('cpu',label,cpu);change('smoke',label,smoke)
        save(root/'cpu_tests.json',cpu);save(root/'smoke.json',smoke)
        run=root/'runs/fold0/cpdino_vitb';run.mkdir(parents=True)
        pred=y[va]*np.float32(1.02 if label=='baseline' else 1.01)
        fields=dict(indices=va.copy(),ids=ids[va].copy(),target=y[va].copy(),pred=pred)
        if change:change('predictions',label,fields)
        np.savez(run/'predictions.npz',**fields);np.savez(run/'predictions_epoch_20.npz',**fields)
        (run/'last.pt').write_bytes(b'synthetic last checkpoint, not a model: '+label.encode())
        (run/'epoch_20.pt').write_bytes(b'synthetic terminal model, not a model: '+label.encode())
        # Deliberately score the original valid matrices: malformed fixtures must
        # be rejected by production shape/identity checks, not fixture construction.
        score=mape64(pred,y[va]);historical=mape64(pred[:2],y[va][:2])
        rows=[dict(epoch=e,updates=e*2,order_sha=array_sha(np.random.default_rng(42+e-1).permutation(tr)),score=score,
                   historical_subset_score=historical) for e in range(1,21)]
        result=dict(architecture='cpdino_vitb',fold=0,seed=42,manifest_sha=msha,single_model=True,ranks_synchronized=True,
                    checkpoint_sha=sha(run/'last.pt'),predictions_sha=sha(run/'predictions.npz'),score=score,history=rows,
                    historical_subset_score=historical,details=copy.deepcopy(details),seconds=1.)
        if change:change('result',label,result)
        save(run/'result.json',result);save(run/'complete.json',dict(status='completed',result_sha=sha(run/'result.json')))
        epoch_artifacts=[dict(epoch=e,score=score,weights_sha=sha(run/'epoch_20.pt') if e==20 else 'e'*64,
                              predictions_sha=sha(run/'predictions_epoch_20.npz') if e==20 else 'f'*64) for e in range(1,21)]
        audit=dict(status='single_model_crossfold_regression_verified',manifest_sha=msha,fold=0,seed=42,recipe=m['recipe'],
                   single_model=True,all_validation_images_scored=True,ssl_split_exact_match=True,ssl_train_only=True,
                   score=score,historical_subset_score=historical,epoch_artifacts=epoch_artifacts,
                   retrospective_best=dict(epoch=20,score=score,not_primary=True))
        if label=='candidate':
            audit.update(loss_execution=copy.deepcopy(marker),audit_type='single_model_loss_coefficient_ablation',loss_contract_verified=True,
                         result_sha=sha(run/'result.json'),complete_sha=sha(run/'complete.json'),checkpoint_sha=sha(run/'last.pt'),
                         predictions_sha=sha(run/'predictions.npz'),cpu_tests_sha=sha(root/'cpu_tests.json'),smoke_sha=sha(root/'smoke.json'))
        if change:change('audit',label,audit)
        save(root/'audit.json',audit)
    return directory/'baseline',directory/'candidate',adapter_sha


def fixture_tests():
    cases=[]
    def candidate(stage, action):
        return lambda actual,label,obj: action(obj) if actual==stage and label=='candidate' else None
    mutations=[
        ('missing_result_loss_execution',candidate('result',lambda x:x['details'].pop('loss_execution'))),
        ('wrong_execution_coefficient',candidate('result',lambda x:x['details']['loss_execution'].update(standardized_log_l1_coefficient=.25))),
        ('missing_CPU_loss_execution',candidate('cpu',lambda x:x.pop('loss_execution'))),
        ('wrong_smoke_execution',candidate('smoke',lambda x:x['details']['loss_execution'].update(active_loss_verified=False))),
        ('wrong_arm_audit_execution',candidate('audit',lambda x:x['loss_execution'].update(relative_error_coefficient=3.))),
        ('unbound_cpu_evidence',candidate('audit',lambda x:x.update(cpu_tests_sha='0'*64))),
        ('head_lr_changed',candidate('manifest',lambda x:x['recipe'].update(head_lr=.0001))),
        ('clip_changed',candidate('manifest',lambda x:x['recipe'].update(gradient_clip=3.))),
        ('seed_changed',candidate('manifest',lambda x:x.update(seed=43))),
        ('schedule_changed',candidate('manifest',lambda x:x['recipe'].update(schedule_epochs=10))),
        ('declared_coefficient_changed',candidate('manifest',lambda x:x['recipe'].update(standardized_log_l1_coefficient=.25))),
        ('boolean_coefficient',candidate('manifest',lambda x:x['recipe'].update(standardized_log_l1_coefficient=False))),
        ('relative_coefficient_changed',candidate('manifest',lambda x:x['recipe'].update(relative_error_coefficient=3.))),
        ('validation_target_changed',candidate('predictions',lambda x:x['target'].__setitem__((0,0),9.))),
        ('validation_ID_changed',candidate('predictions',lambda x:x['ids'].__setitem__(0,'wrongID'))),
        ('float_validation_indices',candidate('predictions',lambda x:x.update(indices=x['indices'].astype(np.float64)))),
        ('broadcast_shape',candidate('predictions',lambda x:x.update(pred=x['pred'][:,:1]))),
        ('training_order_changed',candidate('result',lambda x:x['history'][0].update(order_sha='0'*64))),
        ('result_score_changed',candidate('result',lambda x:x['score'].update(mean=999.))),
    ]
    with tempfile.TemporaryDirectory(prefix='clec_loss_pair_fixture_') as temporary:
        root=Path(temporary)
        for index,alpha in enumerate((0.,.25)):
            b,c,pin=_fixture(root/f'valid{index}',alpha=alpha)
            report=compare(b,c,pin,alpha)
            require(report['status']=='matched_single_model_pair_verified','Valid fixture did not pass')
            require(report['canonical_loss_contracts']['baseline']['standardized_log_l1_coefficient']==.5,'Baseline normalization differs')
            for label,armroot in [('baseline',b),('candidate',c)]:
                recipe=read(armroot/'manifest.json')['recipe']
                require(all(recipe[k]==v for k,v in report['configurations'][label].items()),'Curve analyzer configuration incompatibility')
            cases.append(dict(name=f'valid_coefficient_{alpha:g}',passed=True))
        for index,(name,mutation) in enumerate(mutations):
            b,c,pin=_fixture(root/f'bad{index}',mutation)
            try: compare(b,c,pin)
            except (ValueError,KeyError,AssertionError) as error:
                cases.append(dict(name=name,passed=True,rejected_with=str(error)))
            else:raise AssertionError('Invalid fixture accepted: '+name)
        b,c,pin=_fixture(root/'pins')
        for name,wrong_pin,alpha in [('wrong_independent_adapter_pin','0'*64,0.),('candidate_coefficient_half',pin,.5),('candidate_coefficient_negative',pin,-.1)]:
            try:compare(b,c,wrong_pin,alpha)
            except ValueError as error:cases.append(dict(name=name,passed=True,rejected_with=str(error)))
            else:raise AssertionError('Invalid fixture accepted: '+name)
        # Unchanged metadata target binding must also reject a common corruption.
        def both_targets(stage,label,obj):
            if stage=='predictions':obj['target'][0,0]=9.
        b,c,pin=_fixture(root/'both_bad_targets',both_targets)
        try:compare(b,c,pin)
        except ValueError as error:cases.append(dict(name='both_validation_targets_changed',passed=True,rejected_with=str(error)))
        else:raise AssertionError('Common target corruption accepted')
        # Mutating immutable bytes after the claimed completion must be caught.
        b,c,pin=_fixture(root/'result_hash')
        result=c/'runs/fold0/cpdino_vitb/result.json'
        with result.open('a') as stream:stream.write(' ')
        try:compare(b,c,pin)
        except ValueError as error:cases.append(dict(name='result_hash_changed',passed=True,rejected_with=str(error)))
        else:raise AssertionError('Result hash corruption accepted')
    return dict(status='passed',scope='Synthetic comparison-contract fixtures only; fake byte checkpoint files, no model load, forward, GPU or scientific experiment',
                cases=cases,passed_cases=len(cases),source_sha256=sha(__file__),score_atol=SCORE_ATOL,
                original_source_pins=ORIGINAL_SOURCES,synthetic_adapter_pin_used_only_in_fixtures=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('compare')
    for name in ('baseline','candidate','output'):
        q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--expected-adapter-sha256',required=True)
    q.add_argument('--expected-candidate-coefficient',type=float,default=0.)
    sub.add_parser('fixture-tests')
    args=p.parse_args()
    if args.command=='fixture-tests':
        print(json.dumps(fixture_tests(),indent=2,allow_nan=False));return
    require(not args.output.exists(),'Never overwrite a loss pair audit')
    report=compare(args.baseline,args.candidate,args.expected_adapter_sha256,args.expected_candidate_coefficient)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=report['status'],output=str(args.output),sha256=sha(args.output)),allow_nan=False))


if __name__=='__main__':
    main()
