"""All-data SSL5 for the final recipe A fit: the frozen 20261015 runner with an empty validation set.

The frozen runner uses the validation partition only to exclude images from SSL. This
wrapper admits an empty validation array (training = all audited images) and replaces the
per-split preflight entry, which does not exist for an all-data split, by an addendum bound
to the exact split and preflight hashes; every shared, pixel and metadata check of the frozen
preflight_evidence is kept. Non-empty validation sets keep the original checks, so the
runner's own membership tests are unchanged. Frozen next to the runner sources.
"""
from pathlib import Path

import numpy as np

import phase2_crossfold_ssl_20261015 as S

ADDENDUM = Path('experiments/fulldata_recipeA_20261028/alldata_ssl_addendum.json')
ORIGINAL_MEMBERSHIP = S.validate_membership


def validate_membership(ids, train, val, primary=None):
    if len(val):
        return ORIGINAL_MEMBERSHIP(ids, train, val, primary)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('Image IDs must be a unique one-dimensional vector')
    if (train.ndim != 1 or not np.issubdtype(train.dtype, np.integer) or len(train) < 2
            or train.min() < 0 or train.max() >= len(ids) or len(np.unique(train)) != len(train)):
        raise ValueError('Invalid train indices')
    if val.ndim != 1 or not np.issubdtype(val.dtype, np.integer):
        raise ValueError('Empty validation must be an integer vector')
    if len(train) % 32 == 1:
        raise ValueError('SIGReg requires >=2 images; unsupported singleton final batch')
    if primary is not None:
        raise ValueError('All-data split has no primary validation')


def preflight_evidence(args, ids, train, val):
    addendum = S.read(ADDENDUM)
    if (addendum['split_path'] != str(args.split) or addendum['split_sha256'] != S.digest(args.split)
            or addendum['data_preflight_path'] != str(args.data_preflight)
            or addendum['data_preflight_sha256'] != S.digest(args.data_preflight)
            or addendum['fold'] != args.fold):
        raise ValueError('All-data addendum does not bind this split and preflight')
    if len(val) != 0 or len(train) != addendum['train_n'] or not np.array_equal(np.sort(train), np.arange(len(ids))):
        raise ValueError('All-data split must cover every audited image exactly once with no validation')
    audit = S.read(args.data_preflight)
    if (audit['status'] != 'shared_checks_passed' or audit['errors']
            or audit['failed_shared_checks'] or not audit['pixels']['verified']
            or not audit['checks'] or not all(c['passed'] for c in audit['checks'])):
        raise ValueError('Cross-fold data preflight has unverified or failed shared checks')
    if (audit['pixels']['whole_cache_sha256'] != S.EXPECTED_RAW
            or audit['pixels']['n'] != len(ids) or audit['pixels']['mismatch_rows'] != 0):
        raise ValueError('Preflight current-pixel identity differs')
    for path in (S.META, S.FULL_META):
        found = [record for name, record in audit['inputs'].items() if Path(name).resolve() == path.resolve()]
        if len(found) != 1 or found[0]['sha256'] != S.digest(path):
            raise ValueError(f'Preflight metadata identity differs: {path}')
    return dict(train=dict(n=len(train)), validation=dict(n=0), all_data=True, addendum=str(ADDENDUM))


if __name__ == '__main__':
    S.SOURCES = S.SOURCES + (Path(__file__),)
    S.validate_membership = validate_membership
    S.preflight_evidence = preflight_evidence
    raise SystemExit(S.main())
