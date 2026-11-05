"""Summarize a completed, audited regression pair without touching its artifacts.

Standard library only. Epoch20 remains primary; fixed windows6–10 and16–20
describe correlated development curves, not independent observations. No model,
prediction arrays, cluster state or incomplete run is loaded. This tool provides
evidence for human review and never chooses or launches a follow-up experiment.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

METRICS = ('CD', 'CV', 'HEX', 'mean')
SCALARS = ('online_train_loss', 'encoder_lr', 'head_lr',
           'gradient_norm_mean', 'gradient_clip_fraction')
EARLY, LATE = (6, 10), (16, 20)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, inputs):
    path = Path(path)
    inputs[str(path)] = sha(path)
    return json.loads(path.read_text())


def delta(left, right):
    return {key: left[key] - right[key] for key in METRICS}


def validate_history(manifest, result):
    rows = result['history']
    require(manifest['recipe']['epochs'] == 20, 'This fixed-window diagnostic requires terminal20')
    require([r['epoch'] for r in rows] == list(range(1, 21)), 'Incomplete or unordered history')
    for row in rows:
        require(row['score']['n'] == manifest['validation_n'], 'Validation population changed')
        require(row['updates'] == row['epoch'] * math.ceil(manifest['train_n']/16), 'Update count changed')
        require(all(math.isfinite(row['score'][key]) for key in METRICS), 'Nonfinite validation score')
    require(all(abs(rows[-1]['score'][key]-result['score'][key]) < 1e-6 for key in METRICS),
            'Terminal result/history score mismatch')
    for key in ('train_probe_score', *SCALARS):
        availability = [key in row for row in rows]
        require(all(availability) or not any(availability), 'Partially missing telemetry: ' + key)
        if all(availability):
            if key == 'train_probe_score':
                require(len({row[key]['n'] for row in rows}) == 1, 'Probe population changed')
                require(all(math.isfinite(row[key][metric]) for row in rows for metric in METRICS),
                        'Nonfinite probe score')
            else:
                require(all(math.isfinite(row[key]) for row in rows), 'Nonfinite telemetry: ' + key)
    return rows


def load_arm(root, inputs, require_audit):
    manifest = read(root/'manifest.json', inputs)
    run = root/'runs'/f'fold{manifest["fold"]}'/'cpdino_vitb'
    result, complete = read(run/'result.json', inputs), read(run/'complete.json', inputs)
    require(complete['status'] == 'completed' and complete['result_sha'] == sha(run/'result.json'),
            'Completed result missing or changed')
    require(result['manifest_sha'] == sha(root/'manifest.json'), 'Manifest/result binding failed')
    rows = validate_history(manifest, result)
    if (run/'history.json').exists():
        require(read(run/'history.json', inputs) == rows, 'history.json differs from result history')
    if require_audit:
        audit = read(root/'audit.json', inputs)
        require(audit['status'] == 'single_model_crossfold_regression_verified', 'Arm audit not verified')
        require(audit['manifest_sha'] == result['manifest_sha'], 'Arm audit manifest mismatch')
        require(len(audit['epoch_artifacts']) == 20, 'Incomplete audited epoch artifacts')
        for row, item in zip(rows, audit['epoch_artifacts']):
            require(row['epoch'] == item['epoch'], 'Audited epoch order mismatch')
            require(all(abs(row['score'][key]-item['score'][key]) < 1e-6 for key in METRICS),
                    'Audited epoch score differs from history')
    return dict(manifest=manifest, result=result, rows=rows)


def window(rows, bounds):
    selected = rows[bounds[0]-1:bounds[1]]
    output = dict(epochs=list(bounds), validation={key: mean(r['score'][key] for r in selected) for key in METRICS},
        train_probe=({key: mean(r['train_probe_score'][key] for r in selected) for key in METRICS}
                     if all('train_probe_score' in r for r in selected) else None))
    output.update({key: mean(r[key] for r in selected) if all(key in r for r in selected) else None
                   for key in SCALARS})
    return output


def arm_summary(record):
    rows = record['rows']
    early, late = window(rows, EARLY), window(rows, LATE)
    changes = dict(validation=delta(late['validation'], early['validation']),
        train_probe=delta(late['train_probe'], early['train_probe']) if late['train_probe'] else None)
    changes.update({key: late[key]-early[key] if late[key] is not None else None for key in SCALARS})
    best = min(rows, key=lambda row: row['score']['mean'])
    return dict(terminal20=rows[-1]['score'], early_window=early, late_window=late,
        late_minus_early=changes, telemetry_available={key: key in rows[0] for key in ('train_probe_score', *SCALARS)},
        retrospective_best=dict(epoch=best['epoch'], score=best['score'], diagnostic_only=True))


def summarize(records):
    left, right = (records[label]['manifest'] for label in ('baseline', 'candidate'))
    for key in ('fold', 'train_n', 'validation_n', 'validation_ids_sha', 'log_mean', 'log_std',
                'ssl_checkpoint_sha', 'expected_head_sha'):
        require(left[key] == right[key], 'Pair mismatch: ' + key)
    seeds = [m.get('seed', m['recipe'].get('seed')) for m in (left, right)]
    require(seeds[0] is not None and seeds[0] == seeds[1], 'Pair seeds differ or are absent')
    for a, b in zip(records['baseline']['rows'], records['candidate']['rows']):
        require(all(a[key] == b[key] for key in ('epoch', 'updates', 'order_sha')), 'Epoch data order differs')
    arms = {label: arm_summary(record) for label, record in records.items()}
    paired = {str(a['epoch']): delta(b['score'], a['score']) for a, b in
              zip(records['baseline']['rows'], records['candidate']['rows'])}
    terminal = paired['20']
    early = delta(arms['candidate']['early_window']['validation'], arms['baseline']['early_window']['validation'])
    late = delta(arms['candidate']['late_window']['validation'], arms['baseline']['late_window']['validation'])
    sign = lambda value: (value > 0) - (value < 0)
    changes = [value['late_minus_early'] for value in arms.values()]
    probes_known = all(c['train_probe'] is not None for c in changes)
    losses_known = all(c['online_train_loss'] is not None for c in changes)
    probe_contracts = [m.get('diagnostics', {}) for m in (left, right)]
    probe_pair_matched = probes_known and all(
        probe_contracts[0].get(key) is not None and probe_contracts[0].get(key) == probe_contracts[1].get(key)
        for key in ('train_probe_n', 'train_probe_seed'))
    return dict(fold=left['fold'], seed=seeds[0], primary='fixed terminal epoch20; unchanged', arms=arms,
        candidate_minus_baseline=dict(terminal20=terminal, mean_epoch6_10=early, mean_epoch16_20=late,
            epochs_candidate_lower=sum(v['mean'] < 0 for v in paired.values()),
            last5_epochs_candidate_lower=sum(paired[str(e)]['mean'] < 0 for e in range(16, 21)),
            terminal_and_last5_direction_agree=sign(terminal['mean']) == sign(late['mean']), per_epoch=paired),
        branch_evidence=dict(both_late_validation_mean_worse=all(c['validation']['mean'] > 0 for c in changes),
            both_late_train_probe_mean_better=all(c['train_probe']['mean'] < 0 for c in changes) if probes_known else None,
            both_late_online_train_loss_lower=all(c['online_train_loss'] < 0 for c in changes) if losses_known else None,
            both_cd_cv_better_hex_worse=all(c['validation']['CD'] < 0 and c['validation']['CV'] < 0 and
                                           c['validation']['HEX'] > 0 for c in changes),
            train_probe_contract_matched_between_arms=probe_pair_matched,
            terminal_vs_late_window_ranking_disagrees=sign(terminal['mean']) != sign(late['mean']),
            note='Descriptive signs only, without a statistical or practical-effect-size threshold; manual review required'),
        automatic_followup_selected=False, automatic_launch=False)


def analyze(baseline, candidate, comparison_path):
    inputs = {}
    require(baseline.resolve() != candidate.resolve(), 'Pair must contain two distinct arm roots')
    comparison = read(comparison_path, inputs)
    require(comparison['status'] == 'matched_single_model_pair_verified', 'Pair comparison not verified')
    records = {label: load_arm(root, inputs, True) for label, root in
               [('baseline', baseline), ('candidate', candidate)]}
    bound_hashes = set(comparison['inputs'].values())
    # Local copies may use results/... where the PVC used experiments/...;
    # exact content hashes bind each typed artifact independently of relocation.
    for root in (baseline, candidate):
        manifest = records['baseline' if root == baseline else 'candidate']['manifest']
        run = root/'runs'/f'fold{manifest["fold"]}'/'cpdino_vitb'
        for path in (root/'manifest.json', root/'audit.json', run/'result.json', run/'complete.json'):
            require(sha(path) in bound_hashes, 'Artifact not bound by completed pair audit: ' + str(path))
    output = summarize(records)
    require(comparison['fold'] == output['fold'] and comparison['seed'] == output['seed'], 'Comparison fold/seed mismatch')
    for label in ('baseline', 'candidate'):
        for key, value in comparison['configurations'][label].items():
            require(records[label]['manifest']['recipe'][key] == value, 'Comparison arm configuration mismatch')
        require(all(abs(output['arms'][label]['terminal20'][key]-comparison['scores'][label][key]) < 1e-6
                    for key in METRICS), 'Comparison terminal score mismatch')
    output.update(status='completed_pair_exploratory_curve_summary', inputs=inputs,
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
        windows_fixed_before_fold4_completion=True,
        limitations=['Window means average epoch metrics; they are not prediction or weight ensembles.',
                     'Epochs share images and a trajectory; their fluctuations are not confidence intervals.',
                     'Train probes are fixed subsets, not the full training population.',
                     'Best epochs are retrospective diagnostics and never change terminal20 primary.',
                     'This summary trusts the bound completed pair audit; it does not reload models or re-audit predictions.',
                     'Cross-fold consistency, seed variability and SSL0 efficacy require their separate comparisons.'])
    return output


def fixture(root):
    inputs = {}
    records = dict(baseline=load_arm(root/'results/phase2_lejepa5_regression_20261008', inputs, False),
        candidate=load_arm(root/'results/phase2_lejepa5_sweep_20261013/arms/head1e4_clip3', inputs, False))
    actual = summarize(records)
    prior = read(root/'results/phase2_crossfold_20261015/fold3_curve_analysis.json', inputs)
    checks = 0
    for label, old in [('baseline', 'head1e3_clip1'), ('candidate', 'head1e4_clip3')]:
        for name in ('early_window', 'late_window', 'late_minus_early'):
            for kind in ('validation', 'train_probe'):
                a, b = actual['arms'][label][name][kind], prior['arms'][old][name][kind]
                if a is None or b is None:
                    require(a is b, 'Fixture telemetry availability differs'); checks += 1
                else:
                    for key in METRICS:
                        require(math.isclose(a[key], b[key], rel_tol=0, abs_tol=1e-12), 'Fixture window differs'); checks += 1
    require(actual['candidate_minus_baseline']['terminal20'] == prior['candidate_minus_baseline']['terminal20'], 'Fixture terminal delta differs')
    require(actual['candidate_minus_baseline']['mean_epoch16_20']['mean'] == prior['candidate_minus_baseline']['mean_epoch16_20'], 'Fixture late delta differs')
    require(actual['candidate_minus_baseline']['epochs_candidate_lower'] == 9, 'Fixture epoch ranking count differs')
    require(actual['branch_evidence']['terminal_vs_late_window_ranking_disagrees'], 'Fixture ranking sensitivity lost')
    require(actual['branch_evidence']['both_late_train_probe_mean_better'] is None, 'Absent baseline probe must not imply a result')
    # An unfinished pair must fail before any curve summary can be produced.
    incomplete = dict(records['candidate']['result'], history=records['candidate']['rows'][:-1])
    try:
        validate_history(records['candidate']['manifest'], incomplete)
    except ValueError:
        pass
    else:
        raise AssertionError('Incomplete history accepted')
    return dict(status='passed', fixture='existing completed fold3 baseline/candidate; no fold4 run read',
        window_scalar_checks=checks, terminal_delta=actual['candidate_minus_baseline']['terminal20']['mean'],
        late_window_delta=actual['candidate_minus_baseline']['mean_epoch16_20']['mean'],
        candidate_winning_epochs=9, missing_baseline_probe_preserved=True, incomplete_history_rejected=True,
        primary_unchanged=True, inputs=inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    analyze_parser = sub.add_parser('analyze')
    for name in ('baseline', 'candidate', 'comparison', 'output'):
        analyze_parser.add_argument('--'+name, type=Path, required=True)
    test_parser = sub.add_parser('fixture')
    test_parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.command == 'fixture':
        print(json.dumps(fixture(args.root.resolve()), allow_nan=False)); return
    require(not args.output.exists(), 'Never overwrite an analysis result')
    report = analyze(args.baseline, args.candidate, args.comparison)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=report['status'], output=str(args.output),
        candidate_minus_baseline=report['candidate_minus_baseline']['terminal20'],
        branch_evidence=report['branch_evidence']), allow_nan=False))


if __name__ == '__main__':
    main()
