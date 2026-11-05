"""Audit a matched external-CPDINO SSL0 versus fold-train-only SSL5 pair.

Both arms are independent single regression models. Every regression setting,
head initialization, train order and validation row must match. Only the initial
encoder may differ. Scores are terminal-epoch MAPE; intermediate minima remain
retrospective diagnostics. This NumPy-only audit never changes input artifacts.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

EXTERNAL_SHA = '3ed4c06a3963ab13ff377d4e2957174aaaf637434eb031ae9931fcbfaf9a217f'
COMMON = ('architecture', 'fold', 'seed', 'split_sha256', 'train_n', 'validation_n',
          'images', 'train_ids_sha', 'validation_ids_sha', 'train_order_base_sha',
          'train_indices', 'validation_indices', 'historical_validation_indices',
          'historical_validation_n', 'excluded_from_split_indices', 'log_mean',
          'log_std', 'raw_sha256', 'raw_bytes', 'expected_head_sha', 'mode',
          'execution', 'diagnostics', 'primary_validation')
METRICS = ('CD', 'CV', 'HEX', 'mean', 'n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def ids_sha(ids):
    return hashlib.sha256('\n'.join(np.asarray(ids).astype(str)).encode()).hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def score(pred, target):
    require(pred.shape == target.shape and target.ndim == 2 and target.shape[1] == 3,
            'Prediction/target shapes must match N by 3')
    require(len(pred) > 0 and np.isfinite(pred).all() and np.isfinite(target).all()
            and (target > 0).all(), 'Invalid predictions/targets; no zero exclusions')
    values = (np.abs(pred-target)/target).mean(0)*100
    return dict(CD=float(values[0]), CV=float(values[1]), HEX=float(values[2]),
                mean=float(values.mean()), n=len(pred))


def same_score(actual, expected, context):
    for key in METRICS:
        require(key in expected and np.isfinite(expected[key])
                and abs(actual[key]-expected[key]) < 1e-6,
                f'{context}: metric mismatch {key}')


def checked_file(path, expected, inputs):
    actual = sha(path)
    require(actual == expected, f'Changed artifact: {path}')
    inputs[str(path)] = actual
    return actual


def predictions(path, manifest):
    with np.load(path, allow_pickle=False) as archive:
        result = {key:archive[key].copy() for key in ('indices', 'ids', 'target', 'pred')}
    n = manifest['validation_n']
    require(result['target'].shape == result['pred'].shape == (n, 3), 'Invalid prediction matrix shape')
    require(result['indices'].shape == result['ids'].shape == (n,), 'Invalid validation identity shape')
    require(np.issubdtype(result['indices'].dtype, np.integer), 'Validation indices must be integers')
    require(len(np.unique(result['indices'])) == len(np.unique(result['ids'])) == n,
            'Validation rows/IDs must be unique')
    np.testing.assert_array_equal(result['indices'], manifest['validation_indices'])
    require(ids_sha(result['ids']) == manifest['validation_ids_sha'], 'Validation ID hash differs')
    return result, score(result['pred'], result['target'])


def verify_arm(root, label, inputs):
    root = Path(root)
    manifest_path, audit_path = root/'manifest.json', root/'audit.json'
    manifest, audit = read(manifest_path), read(audit_path)
    manifest_sha = sha(manifest_path)
    inputs[str(manifest_path)] = manifest_sha
    inputs[str(audit_path)] = sha(audit_path)
    require(audit['status'] == 'single_model_crossfold_regression_verified', 'Missing individual regression audit')
    require(audit['manifest_sha'] == manifest_sha, 'Individual audit has a different manifest')
    require(audit['recipe'] == manifest['recipe'], 'Individual audit recipe differs')
    require(audit['fold'] == manifest['fold'] and audit['seed'] == manifest['seed'], 'Audit fold/seed differs')
    require(audit['single_model'] and audit['all_validation_images_scored'], 'Incomplete/summed model audit')
    require(manifest['architecture'] == 'cpdino_vitb' and manifest['mode'] == 'full', 'Unexpected model/tuning mode')
    require(manifest['recipe']['seed'] == manifest['seed'], 'Manifest recipe seed differs')
    require(manifest['hashes'], 'Missing frozen input hashes')
    for path, expected in manifest['hashes'].items():
        checked_file(Path(path), expected, inputs)
    train = np.asarray(manifest['train_indices'], dtype=np.int64)
    validation = np.asarray(manifest['validation_indices'], dtype=np.int64)
    require(len(train) == manifest['train_n'] and len(np.unique(train)) == len(train), 'Malformed train indices')
    require(not np.intersect1d(train, validation).size, 'Regression train/validation overlap')
    require(array_sha(train) == manifest['train_order_base_sha'], 'Train order base hash differs')
    run = root/'runs'/f'fold{manifest["fold"]}'/'cpdino_vitb'
    result_path, complete_path = run/'result.json', run/'complete.json'
    result, complete = read(result_path), read(complete_path)
    require(complete['status'] == 'completed', 'Regression arm did not complete')
    checked_file(result_path, complete['result_sha'], inputs)
    inputs[str(complete_path)] = sha(complete_path)
    require(result['manifest_sha'] == manifest_sha, 'Result manifest differs')
    require(result['architecture'] == manifest['architecture'] and result['fold'] == manifest['fold']
            and result['seed'] == manifest['seed'], 'Result architecture/fold/seed differs')
    require(result['single_model'] and result['ranks_synchronized'], 'Not one synchronized single model')
    if 'result_sha' in audit:
        require(audit['result_sha'] == inputs[str(result_path)], 'Individual audit result hash differs')
    if 'complete_sha' in audit:
        require(audit['complete_sha'] == inputs[str(complete_path)], 'Individual audit completion hash differs')
    checked_file(run/'predictions.npz', result['predictions_sha'], inputs)
    checked_file(run/'last.pt', result['checkpoint_sha'], inputs)
    fields, final_score = predictions(run/'predictions.npz', manifest)
    same_score(final_score, result['score'], f'{label} result')
    same_score(final_score, audit['score'], f'{label} audit')
    epochs = manifest['recipe']['epochs']
    require(len(result['history']) == len(audit['epoch_artifacts']) == epochs, 'Incomplete epoch evidence')
    updates_per_epoch = math.ceil(len(train)/manifest['recipe']['effective_batch'])
    for epoch, (item, artifact) in enumerate(zip(result['history'], audit['epoch_artifacts']), 1):
        require(item['epoch'] == artifact['epoch'] == epoch, 'Incorrect epoch sequence')
        require(item['updates'] == epoch*updates_per_epoch, 'Incorrect update count')
        order = np.random.default_rng(manifest['seed']+epoch-1).permutation(train)
        require(item['order_sha'] == array_sha(order), 'Training order does not match seed/indices')
        same_score(item['score'], artifact['score'], f'{label} epoch{epoch}')
    final_artifact = audit['epoch_artifacts'][-1]
    final_predictions = run/f'predictions_epoch_{epochs:02d}.npz'
    checked_file(final_predictions, final_artifact['predictions_sha'], inputs)
    checked_file(run/f'epoch_{epochs:02d}.pt', final_artifact['weights_sha'], inputs)
    with np.load(final_predictions, allow_pickle=False) as final:
        for key, value in fields.items():
            np.testing.assert_array_equal(value, final[key])
    same_score(final_score, result['history'][-1]['score'], f'{label} terminal history')
    historical = np.asarray(manifest['historical_validation_indices'], dtype=np.int64)
    require(len(historical) == manifest['historical_validation_n']
            and len(np.unique(historical)) == len(historical)
            and np.isin(historical, validation).all(), 'Historical subset not in held-out validation')
    positions = {int(index):row for row,index in enumerate(fields['indices'])}
    take = np.array([positions[int(index)] for index in historical], dtype=np.int64)
    historical_score = score(fields['pred'][take], fields['target'][take])
    same_score(historical_score, result['historical_subset_score'], f'{label} historical result')
    same_score(historical_score, audit['historical_subset_score'], f'{label} historical audit')
    require(result['details']['head_initial_tensor_sha'] == manifest['expected_head_sha'], 'Loaded head differs')
    return dict(root=str(root), manifest=manifest, result=result, audit=audit,
                fields=fields, score=final_score, historical_score=historical_score)


def verify_initialization(records, inputs):
    zero, five = records['ssl0'], records['ssl5']
    m0, m5 = zero['manifest'], five['manifest']
    require(m0['initialization'] == 'SSL0_EXTERNAL_PRETRAINED'
            and m0['ssl_epochs'] == m0['ssl_steps'] == 0, 'SSL0 must be an external-only initialization')
    require(not any(key in m0 for key in ('ssl_root', 'ssl_manifest_sha', 'ssl_checkpoint_sha')),
            'SSL0 cannot carry an adapted SSL checkpoint')
    require(m0['external_checkpoint_sha'] == EXTERNAL_SHA, 'SSL0 external checkpoint identity differs')
    external = Path(m0['external_checkpoint_path'])
    checked_file(external, EXTERNAL_SHA, inputs)
    require(m0['hashes'].get(str(external)) == EXTERNAL_SHA, 'External initialization not frozen in SSL0 inputs')
    require(EXTERNAL_SHA in m5['hashes'].values(), 'SSL5 regression has different external foundation provenance')
    details0 = zero['result']['details']
    require(details0['initialization'] == 'SSL0_EXTERNAL_PRETRAINED' and details0['ssl_epochs'] == 0,
            'Result did not use external-only SSL0 initialization')
    require(details0['external_checkpoint_sha'] == EXTERNAL_SHA
            and details0['external_encoder_tensor_sha'] == m0['external_encoder_tensor_sha'],
            'Loaded SSL0 encoder identity differs')
    require(zero['audit']['initialization'] == 'SSL0_EXTERNAL_PRETRAINED'
            and zero['audit']['ssl_epochs'] == 0, 'Individual audit does not confirm SSL0')
    require(m5['ssl_epochs'] == 5 and m5['ssl_steps'] > 0, 'SSL5 arm requires five completed SSL epochs')
    ssl = Path(m5['ssl_root'])
    sm, sr, sc = read(ssl/'manifest.json'), read(ssl/'run/result.json'), read(ssl/'run/complete.json')
    checked_file(ssl/'manifest.json', m5['ssl_manifest_sha'], inputs)
    require(sm['hashes'] and EXTERNAL_SHA in sm['hashes'].values(), 'SSL5 external foundation provenance missing')
    for path, expected in sm['hashes'].items():
        checked_file(Path(path), expected, inputs)
    require(sm['seed'] == sm['recipe']['seed'] == m5['ssl_seed']
            and sm['recipe']['epochs'] == 5, 'SSL5 manifest seed/duration differs')
    require(sc['status'] == 'completed', 'SSL5 adaptation incomplete')
    checked_file(ssl/'run/result.json', sc['result_sha'], inputs)
    inputs[str(ssl/'run/complete.json')] = sha(ssl/'run/complete.json')
    require(sr['manifest_sha'] == m5['ssl_manifest_sha'], 'SSL5 result manifest differs')
    checked_file(ssl/'run/last.pt', m5['ssl_checkpoint_sha'], inputs)
    require(sr['checkpoint_sha'] == m5['ssl_checkpoint_sha'], 'SSL5 regression used another adaptation checkpoint')
    require(sm['fold'] == m5['fold'] and sm['split_sha256'] == m5['split_sha256'], 'SSL5 fold/split differs')
    for key in ('train_ids_sha', 'validation_ids_sha', 'train_n', 'raw_sha256', 'raw_bytes'):
        require(sm[key] == m5[key], f'SSL5 adaptation membership/provenance differs: {key}')
    require(sm['val_n'] == m5['validation_n'] and sm['total_images'] == m5['images'], 'SSL5 data counts differ')
    require(sr['epochs'] == 5 and sr['steps'] == m5['ssl_steps'] == math.ceil(m5['train_n']/32)*5,
            'SSL5 epoch/update count differs')
    require(sr['validation_images_used'] == 0 and len(sr['epochs_detail']) == 5,
            'SSL adaptation used validation or lacks epoch evidence')
    require(sm['target_arrays_loaded'] is False and sr['target_arrays_loaded'] is False,
            'SSL adaptation must not load target arrays')
    for epoch, item in enumerate(sr['epochs_detail'], 1):
        require(item['epoch'] == epoch and item['unique_train_images'] == item['train_images_seen'] == m5['train_n']
                and item['validation_images_used'] == 0, 'SSL epoch coverage failed')
        order = np.random.default_rng(sm['seed']+epoch-1).permutation(np.asarray(m5['train_indices'], dtype=np.int64))
        require(item['order_sha'] == array_sha(order), 'SSL epoch training order differs')
    details5 = five['result']['details']
    require(details5['ssl_epochs'] == 5 and details5['ssl_checkpoint_sha'] == m5['ssl_checkpoint_sha']
            and details5['encoder_tensor_sha'] == m5['ssl_encoder_tensor_sha'], 'Loaded SSL5 encoder differs')
    require(five['audit']['ssl_train_only'] and five['audit']['ssl_split_exact_match'], 'Individual audit lacks fold-clean SSL proof')
    require(m0['external_encoder_tensor_sha'] != m5['ssl_encoder_tensor_sha'], 'SSL0 and SSL5 encoders are identical')


def compare(ssl0, ssl5, output):
    ssl0, ssl5, output = Path(ssl0), Path(ssl5), Path(output)
    if output.exists():
        raise FileExistsError('Never overwrite an SSL ablation comparison')
    require(ssl0.resolve() != ssl5.resolve(), 'SSL ablation requires two distinct model arms')
    inputs = {}
    records = {label:verify_arm(root, label, inputs) for label, root in (('ssl0', ssl0), ('ssl5', ssl5))}
    left, right = records['ssl0']['manifest'], records['ssl5']['manifest']
    for key in COMMON:
        require(left[key] == right[key], f'Unmatched regression setting: {key}')
    require(left['recipe'] == right['recipe'], 'Every regression recipe value must match, including head LR and clipping')
    for key in ('indices', 'ids', 'target'):
        np.testing.assert_array_equal(records['ssl0']['fields'][key], records['ssl5']['fields'][key])
    for zero, five in zip(records['ssl0']['result']['history'], records['ssl5']['result']['history']):
        for key in ('epoch', 'updates', 'order_sha'):
            require(zero[key] == five[key], f'Unmatched training history: {key}')
    if 'matched_ssl5_manifest_sha' in left:
        require(left['matched_ssl5_manifest_sha'] == sha(ssl5/'manifest.json'), 'SSL0 matched a different SSL5 arm')
    verify_initialization(records, inputs)
    scores = {label:record['score'] for label, record in records.items()}
    report = dict(status='matched_single_model_ssl0_vs_ssl5_verified',
                  checked_utc=time.strftime('%FT%TZ', time.gmtime()), fold=left['fold'], seed=left['seed'],
                  roots={'ssl0':str(ssl0), 'ssl5':str(ssl5)}, primary=f'fixed terminal epoch{left["recipe"]["epochs"]}',
                  scores=scores, ssl5_minus_ssl0={key:scores['ssl5'][key]-scores['ssl0'][key] for key in METRICS[:-1]},
                  lower_is_better=True, ranking=sorted(scores, key=lambda label:(scores[label]['mean'], label)),
                  single_model_only=True, matched_regression_recipe=left['recipe'],
                  only_planned_training_difference='initial encoder: external CPDINO versus the same external CPDINO after fold-train-only SSL5',
                  same_head_initialization_training_order_and_validation=True,
                  initial_encoder_sha={'ssl0':left['external_encoder_tensor_sha'], 'ssl5':right['ssl_encoder_tensor_sha']},
                  historical_subset_scores={label:r['historical_score'] for label,r in records.items()},
                  retrospective_best={label:r['audit']['retrospective_best'] for label,r in records.items()},
                  regression_seconds={label:r['result']['seconds'] for label,r in records.items()},
                  source_sha256=sha(Path(__file__)), inputs=inputs,
                  limitations=['One reused development fold and seed do not establish a stable or official improvement',
                               'Only terminal epochs determine this comparison; intermediate best epochs are retrospective',
                               'Patient independence and near-duplicate leakage remain unverified'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssl0-root', '--ssl0', dest='ssl0', type=Path, required=True)
    parser.add_argument('--ssl5-root', '--ssl5', dest='ssl5', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.ssl0, args.ssl5, args.output)
