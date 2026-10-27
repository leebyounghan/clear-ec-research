"""Fold-clean SSL for a fold whose validation holds zero targets; otherwise the frozen 20261015 runner.

The 20261015 data preflights mark fold2 ineligible only for the positive-target
regression protocol (two validation HEX targets are zero). SSL loads no targets.
This wrapper keeps every shared, pixel, split and metadata check of the frozen
preflight_evidence, replaces only the positive-target eligibility flag by the
explicit zero-policy addendum bound to this exact split and preflight file, and
freezes itself next to the runner sources so verify() re-checks it.
"""
from pathlib import Path
import sys

import phase2_crossfold_ssl_20261015 as S

ADDENDUM = Path('experiments/oof_recipeA_20261027/fold2_zero_policy_addendum.json')


def preflight_evidence(args, ids, train, val):
    addendum = S.read(ADDENDUM)
    if (addendum['split_path'] != str(args.split) or addendum['split_sha256'] != S.digest(args.split)
            or addendum['data_preflight_path'] != str(args.data_preflight)
            or addendum['data_preflight_sha256'] != S.digest(args.data_preflight)
            or addendum['fold'] != args.fold):
        raise ValueError('Zero-policy addendum does not bind this split and preflight')
    audit = S.read(args.data_preflight)
    if (audit['status'] != 'shared_checks_passed' or audit['errors']
            or audit['failed_shared_checks'] or not audit['pixels']['verified']
            or not audit['checks'] or not all(c['passed'] for c in audit['checks'])):
        raise ValueError('Cross-fold data preflight has unverified or failed shared checks')
    if (audit['pixels']['whole_cache_sha256'] != S.EXPECTED_RAW
            or audit['pixels']['n'] != len(ids) or audit['pixels']['mismatch_rows'] != 0):
        raise ValueError('Preflight current-pixel identity differs')
    matches = [entry for path, entry in audit['splits'].items()
               if Path(path).resolve() == args.split.resolve()]
    if len(matches) != 1:
        raise ValueError('Chosen split missing or ambiguous in data preflight')
    entry = matches[0]
    if (entry['eligible_positive_target_candidate']
            or entry['split_sha256'] != S.digest(args.split)
            or not entry['checks'] or not all(c['passed'] for c in entry['checks'])
            or entry['train']['n'] != len(train) or entry['validation']['n'] != len(val)
            or str(args.fold) not in entry['provenance']['matched_manifest_folds']):
        raise ValueError('Fold must be ineligible only through zero targets, with all split checks passed')
    if sorted(addendum['zero_validation_ids']) != sorted(addendum['zero_validation_ids_recomputed']):
        raise ValueError('Addendum zero rows inconsistent')
    for path in (S.META, S.FULL_META):
        found = [record for name, record in audit['inputs'].items()
                 if Path(name).resolve() == path.resolve()]
        if len(found) != 1 or found[0]['sha256'] != S.digest(path):
            raise ValueError(f'Preflight metadata identity differs: {path}')
    return entry


if __name__ == '__main__':
    S.SOURCES = S.SOURCES + (Path(__file__),)
    S.preflight_evidence = preflight_evidence
    raise SystemExit(S.main())
