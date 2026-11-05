#!/usr/bin/env python3
"""Versioned read-only data audit with exact historical fold0/1 provenance.

Reuses the pinned original cross-fold data audit for all data, full-pixel,
slide, duplicate-connected-group and split-cleaning checks. Only the two
documented legacy splits may gain a provenance match: their original manifest
omitted split_path, but pinned generation sources and a pinned downstream
manifest/reference explicitly establish the same path, hash and row identities.
No manifests, splits, sources, labels or prior audit reports are modified.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import phase2_crossfold_data_20261015 as original

EXPECTED_HELPER = '45aca046c49da542b2ca7f05bf6564041ca26a6bc549934d4b472a917772ced1'
LEGACY = Path('experiments/ssl_screen_20260911')
DOWNSTREAM = Path('experiments/target_screen_20260912')
PINS = {
    str(LEGACY/'manifest.json'):'cc38e1c4b1ba8829aef0a54795bba286e3f0a80b3114d3cfef36ffb8b66a7b92',
    str(DOWNSTREAM/'manifest.json'):'2d82a7c8cb76a9d4b55aad908d1f66c900fb295e45916b01b3b70d1a317fa464',
    str(LEGACY/'source/prepare_ssl_screen.py'):'f8362b6784868c754db7729e7b36e3cafd8adf20b06d867b235a0035759abd3c',
    str(DOWNSTREAM/'source/prepare_target_screen.py'):'571ae05ff377abda2677e21cf73054c42bebf52cf61ce938840c1a464823acfd',
    str(DOWNSTREAM/'preflight/reference.npz'):'440629e79975a80e7b8e1a9d13d628def403127985d09eb56ec048b6a3b0e9b0',
    str(LEGACY/'preflight/pixel_fingerprints.npz'):'dde29080a4c95a9ea55a23956500f9aee5b469919ba9730648deffa617be81f3',
}
FOLDS = {
    '0':dict(path=str(LEGACY/'preflight/fold0.npz'),
             sha256='d45a329b8f68103cf9d13b4b89ab88efd258466407f9241412dd891f2eb71ad8',
             train=7151, validation=1791, primary=1781, historical_train=7161),
    '1':dict(path=str(LEGACY/'preflight/fold1.npz'),
             sha256='27124461604e49f45fed04416b220eb36ee9b28f14b97ea0214a10164bb98bbb',
             train=7157, validation=1791, primary=1787, historical_train=7161),
}
ORIGINAL_INSPECT = original.inspect_split


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Audit(original.Audit):
    def __init__(self, root):
        require(original.sha(original.__file__) == EXPECTED_HELPER,
                'Original data helper changed; refusing to reuse an unreviewed implementation')
        super().__init__(root)
        self.report.update(source_sha256=original.sha(__file__),
                           reused_helper_sha256=EXPECTED_HELPER,
                           legacy_provenance_policy='Exact pinned source/manifest/downstream reference chain; only documented fold0 and fold1 paths',
                           historical_inputs_modified=False)
        self.identify(Path(__file__).resolve())
        self.identify(Path(original.__file__).resolve())


def pinned_input(a, path, expected=None):
    path = str(path)
    actual_path = a.identify(path)
    observed = original.sha(actual_path)
    require(observed == (PINS[path] if expected is None else expected),
            f'Pinned legacy evidence changed: {path}')
    return actual_path


def verify_chain(a, fold, path, data, ids, y):
    """No filename inference: exact requested path plus two pinned manifests."""
    contract = FOLDS[fold]
    require(path.resolve() == a.path(contract['path']).resolve(), 'Not a documented legacy split path')
    pinned_input(a, contract['path'], contract['sha256'])
    for evidence in PINS:
        pinned_input(a, evidence)
    prior = a.read_json(LEGACY/'manifest.json')
    downstream = a.read_json(DOWNSTREAM/'manifest.json')
    record, later = prior['folds'][fold], downstream['folds'][fold]
    require('split_path' not in record, 'Legacy resolver applies only to the known omitted-field schema')
    require(a.path(later['split_path']).resolve() == path.resolve(), 'Downstream split_path does not identify this file')
    require(record['split_sha256'] == later['split_sha256'] == contract['sha256'], 'Fold-record split hash mismatch')
    for manifest, name in ((prior, 'original'), (downstream, 'downstream')):
        require(manifest['input_sha256'][contract['path']] == contract['sha256'],
                f'{name} protected-input split hash differs')
    require(downstream['input_sha256'][str(LEGACY/'manifest.json')] == PINS[str(LEGACY/'manifest.json')],
            'Downstream does not bind the pinned original manifest')
    old_source, later_source = str(LEGACY/'source/prepare_ssl_screen.py'), str(DOWNSTREAM/'source/prepare_target_screen.py')
    require(prior['source_sha256'][old_source] == downstream['source_sha256'][old_source] == PINS[old_source],
            'Original split generator is not pinned by both manifests')
    require(downstream['source_sha256'][later_source] == PINS[later_source], 'Downstream generator is not pinned')
    for key in ('train', 'validation', 'primary_validation', 'historical_train'):
        require(key in data, f'Legacy NPZ lacks {key}')
    tr, va, primary, historic = (data[key] for key in ('train', 'validation', 'primary_validation', 'historical_train'))
    require(len(tr) == record['train_rows'] == later['train_rows'] == contract['train'], 'Legacy train count differs')
    require(len(va) == record['validation_rows'] == later['validation_rows'] == contract['validation'], 'Legacy validation count differs')
    require(len(primary) == record['primary_validation_rows'] == later['primary_rows'] == contract['primary'], 'Legacy primary count differs')
    require(len(historic) == contract['historical_train'], 'Historical train count differs')
    require(len(historic)-len(tr) == record['removed_train_rows'], 'Legacy removed-train count differs')
    require(len(va)-len(primary) == record['excluded_primary_validation_rows'], 'Legacy primary-exclusion count differs')
    # Historical source explicitly compares these exact old split indices. This
    # extra read verifies the reconstruction input, without opening any weights.
    old_path = Path('experiments/region_seedcheck_20260906/runs')/f'full_quality_f{fold}_seed77_c256k2_mean_multitask_slidesmooth'/'split_indices.npz'
    pinned_input(a, old_path, record['historical_split_sha256'])
    require(prior['input_sha256'][str(old_path)] == downstream['input_sha256'][str(old_path)] == record['historical_split_sha256'],
            'Historical split is not bound by both manifests')
    old = a.npz(old_path)
    np.testing.assert_array_equal(historic, old['train'])
    np.testing.assert_array_equal(va, old['validation'])
    reference_path = DOWNSTREAM/'preflight/reference.npz'
    require(downstream['input_sha256'][str(reference_path)] == PINS[str(reference_path)], 'Downstream reference hash differs')
    reference = a.npz(reference_path)
    required = {'folds', 'indices', 'ids', 'target', 'primary'}
    require(required.issubset(reference), 'Downstream reference schema is incomplete')
    selected = reference['folds'] == int(fold)
    require(int(selected.sum()) == len(va), 'Downstream reference fold count differs')
    for name, actual in (('indices', va), ('ids', ids[va]), ('target', y[va])):
        expected = reference[name][selected]
        if name == 'ids':
            expected = expected.astype(str)
        np.testing.assert_array_equal(expected, actual)
    primary_mask = reference['primary'][selected]
    require(primary_mask.dtype == np.bool_ or np.isin(primary_mask, [0, 1]).all(), 'Malformed downstream primary flags')
    np.testing.assert_array_equal(va[primary_mask.astype(bool)], primary)
    # Rebind current metadata and current-pixel fingerprints to the historical
    # chain; all current pixel/slide/connected tests are still owned by original.
    shared = ('datasets/clear_ec_official/cache_native_meta_512.npz',
              'datasets/clear_ec_official/final_train_ids.csv',
              'datasets/clear_ec_fullframe_20260904/fullframe_meta.npz',
              'datasets/clear_ec_fullframe_20260904/fullframe_manifest.json')
    for name in shared:
        expected = prior['input_sha256'][name]
        require(downstream['input_sha256'][name] == expected, 'Metadata hash chain differs: '+name)
        pinned_input(a, name, expected)
    fingerprints = str(LEGACY/'preflight/pixel_fingerprints.npz')
    require(downstream['input_sha256'][fingerprints] == PINS[fingerprints], 'Downstream pixel-fingerprint hash differs')
    return dict(status='verified', fold=fold,
                original_manifest=str(LEGACY/'manifest.json'), original_manifest_sha256=PINS[str(LEGACY/'manifest.json')],
                downstream_manifest=str(DOWNSTREAM/'manifest.json'), downstream_manifest_sha256=PINS[str(DOWNSTREAM/'manifest.json')],
                original_generator=old_source, original_generator_sha256=PINS[old_source],
                downstream_generator=later_source, downstream_generator_sha256=PINS[later_source],
                explicit_downstream_split_path=later['split_path'], split_sha256=contract['sha256'],
                downstream_reference=str(reference_path), downstream_reference_sha256=PINS[str(reference_path)],
                historical_split_path=str(old_path), historical_split_sha256=record['historical_split_sha256'],
                checked=['exact known path; no filename-only inference', 'both manifests and generation sources pinned',
                         'original/downstream records and protected-input split hashes agree',
                         'actual NPZ counts and historical indices agree', 'downstream reference indices/IDs/targets/primary agree',
                         'current metadata/fingerprint hash chain agrees'],
                source_mapping_evidence='Pinned prepare_ssl_screen.py lines101-108 writes preflight/fold{fold}.npz; pinned prepare_target_screen.py lines80-81 consumes that exact path and prior fold hash',
                current_pixel_and_group_checks_reused_unchanged=True)


def inspect_split(a, path, ids, y, slides, hashes, connected):
    result, data = ORIGINAL_INSPECT(a, path, ids, y, slides, hashes, connected)
    fold = next((key for key, contract in FOLDS.items()
                 if a.path(contract['path']).resolve() == path.resolve()), None)
    if fold is None or data is None or result.get('provenance', {}).get('matched_manifest_folds'):
        return result, data
    try:
        evidence = verify_chain(a, fold, path, data, ids, y)
        result['provenance']['legacy_explicit_chain'] = evidence
        result['provenance']['matched_manifest_folds'] = [fold]
        result['checks'].append(dict(name='pinned_legacy_provenance_chain', passed=True))
        result['eligible_positive_target_candidate'] = bool(
            result['positive_targets_train_and_validation'] and all(check['passed'] for check in result['checks']))
    except (OSError, ValueError, KeyError, IndexError, TypeError, AssertionError) as exc:
        result['provenance']['legacy_explicit_chain'] = dict(status='failed', error=f'{type(exc).__name__}: {exc}')
        result['checks'].append(dict(name='pinned_legacy_provenance_chain', passed=False))
        result['eligible_positive_target_candidate'] = False
    return result, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--out', type=Path, required=True, help='New JSON path; parent must exist')
    parser.add_argument('--verify-pixels', action='store_true')
    parser.add_argument('--split', action='append', default=[])
    args = parser.parse_args()
    audit = Audit(args.root.resolve())
    out = audit.path(args.out)
    if out.exists() or not out.parent.is_dir():
        parser.error('Output must be a new file in an existing directory')
    original.inspect_split = inspect_split
    try:
        original.run(audit, args)
    except (OSError, ValueError, KeyError, IndexError, TypeError, AssertionError) as exc:
        audit.report['errors'].append(f'{type(exc).__name__}: {exc}')
    failed = [check['name'] for check in audit.report['checks'] if not check['passed']]
    audit.report['failed_shared_checks'] = failed
    audit.report['status'] = 'incomplete' if audit.report['errors'] else 'failed' if failed else 'shared_checks_passed'
    audit.report['training_started'] = False
    with out.open('x') as stream:
        json.dump(audit.report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=audit.report['status'], output=str(out),
                         inventory_count=audit.report.get('inventory_count', 0), failed_shared_checks=failed,
                         errors=len(audit.report['errors']), pixels_verified=audit.report.get('pixels', {}).get('verified', False),
                         legacy_provenance={key:audit.report['splits'].get(contract['path'], {}).get('provenance', {}).get('legacy_explicit_chain', {}).get('status', 'unavailable')
                                            for key,contract in FOLDS.items()})))
    return 3 if audit.report['errors'] else 2 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
