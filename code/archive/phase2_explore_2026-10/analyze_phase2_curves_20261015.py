#!/usr/bin/env python3
"""Reproduce aggregate fold3 curve diagnostics from existing local evidence.

Standard library only; no remote reads, model loading, training or source edits.
Writes one new JSON file and refuses overwrite. The fixed epoch20 audit remains
primary. Window comparisons and best epochs are exploratory, not independent
replications or a replacement model-selection rule. No fold4 outcomes are read.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev

METRICS = ('CD', 'CV', 'HEX', 'mean')
NEW_ARMS = ('head1e4_clip1', 'head1e4_clip3', 'head3e4_clip3', 'head1e3_clip3')
SWEEP = Path('results/phase2_lejepa5_sweep_20261013')
EARLY = (6, 10)
LATE = (16, 20)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def clean_record(record):
    return {k: v for k, v in record.items() if k != 'epoch_complete'}


def score_mean(rows, key):
    if not all(key in row for row in rows):
        return None
    return {metric: mean(row[key][metric] for row in rows) for metric in METRICS}


def scalar_mean(rows, key):
    return mean(row[key] for row in rows) if all(key in row for row in rows) else None


def window(rows, bounds):
    selected = [row for row in rows if bounds[0] <= row['epoch'] <= bounds[1]]
    if len(selected) != bounds[1] - bounds[0] + 1:
        raise ValueError('Incomplete window')
    return dict(epochs=list(bounds), validation=score_mean(selected, 'score'),
        train_probe=score_mean(selected, 'train_probe_score'),
        online_train_loss=scalar_mean(selected, 'online_train_loss'),
        gradient_norm_mean=scalar_mean(selected, 'gradient_norm_mean'),
        clipping_fraction=scalar_mean(selected, 'gradient_clip_fraction'),
        validation_epoch_std=pstdev(row['score']['mean'] for row in selected),
        averaging_note='Mean of epoch metrics; not an ensemble, weight average or independent error estimate')


def difference(late, early):
    result = {}
    for key in late:
        if key in ('validation', 'train_probe'):
            result[key] = ({m: late[key][m] - early[key][m] for m in METRICS}
                           if late[key] is not None and early[key] is not None else None)
        elif key in ('online_train_loss', 'gradient_norm_mean', 'clipping_fraction'):
            result[key] = late[key] - early[key] if late[key] is not None and early[key] is not None else None
    return result


def lr_at_epoch_end(recipe, epoch, updates=448):
    step = epoch * updates - 1
    if step < updates:
        ratio = (step + 1) / updates
    else:
        phase = (step - updates) / (updates * (recipe['epochs'] - 1) - 1)
        ratio = .1 + .9 * .5 * (1 + math.cos(math.pi * phase))
    return dict(encoder_lr=recipe['encoder_lr'] * ratio, head_lr=recipe['head_lr'] * ratio)


class Reader:
    def __init__(self, root):
        self.root, self.inputs, self.checks = root, {}, []

    def file(self, relative):
        path = self.root / relative
        self.inputs[str(relative)] = dict(bytes=path.stat().st_size, sha256=sha(path))
        return path

    def json(self, relative):
        return json.loads(self.file(relative).read_text())

    def check(self, name, passed):
        self.checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise ValueError('Evidence check failed: ' + name)


def load_arm(reader, name, directory, result_available=True):
    manifest = reader.json(directory / 'manifest.json')
    run = directory / 'runs/fold3/cpdino_vitb'
    availability = dict(result_json=result_available, history_json=False,
                        logged_history_json=False, local_weights_loaded=False,
                        per_image_predictions_loaded=False)
    if result_available:
        result = reader.json(run / 'result.json')
        complete = reader.json(run / 'complete.json')
        reader.check(name + ': completion hash', complete['status'] == 'completed' and
            complete['result_sha'] == reader.inputs[str(run / 'result.json')]['sha256'])
        reader.check(name + ': manifest hash', result['manifest_sha'] ==
            reader.inputs[str(directory / 'manifest.json')]['sha256'])
        history = result['history']
    else:
        history = [clean_record(row) for row in reader.json(directory / 'logged_history.json')]
        availability['logged_history_json'] = True
    if (reader.root / run / 'history.json').exists():
        recorded = reader.json(run / 'history.json')
        reader.check(name + ': result/history equality', recorded == history)
        availability['history_json'] = True
    if result_available and (reader.root / directory / 'logged_history.json').exists():
        logged = [clean_record(row) for row in reader.json(directory / 'logged_history.json')]
        reader.check(name + ': result/logged history equality', logged == history)
        availability['logged_history_json'] = True
    reader.check(name + ': all 20 epochs', [row['epoch'] for row in history] == list(range(1, 21)))
    reader.check(name + ': updates and validation population', all(
        row['updates'] == 448 * row['epoch'] and row['score']['n'] == 1791 for row in history))
    reader.check(name + ': finite validation values', all(
        math.isfinite(row['score'][key]) for row in history for key in METRICS))
    telemetry = all('train_probe_score' in row for row in history)
    availability.update(train_probe=telemetry, online_train=all('online_train_loss' in row for row in history),
                        logged_lr=all('encoder_lr' in row for row in history),
                        clipping=all('gradient_clip_fraction' in row for row in history))
    for row in history:
        expected = lr_at_epoch_end(manifest['recipe'], row['epoch'])
        if 'encoder_lr' in row:
            reader.check(name + f': epoch{row["epoch"]} observed LR matches recipe', all(
                math.isclose(row[k], v, rel_tol=1e-10, abs_tol=1e-15) for k, v in expected.items()))
        if telemetry:
            reader.check(name + f': epoch{row["epoch"]} probe size', row['train_probe_score']['n'] == 256)
    return manifest, history, availability


def summarize(name, manifest, history, availability):
    early, late = window(history, EARLY), window(history, LATE)
    best = min(history, key=lambda row: row['score']['mean'])
    final = history[-1]
    epochs = []
    for row in history:
        observed = 'encoder_lr' in row
        lr = {k: row[k] for k in ('encoder_lr', 'head_lr')} if observed else lr_at_epoch_end(manifest['recipe'], row['epoch'])
        epochs.append(dict(epoch=row['epoch'], validation=row['score'],
            train_probe=row.get('train_probe_score'), online_train_loss=row.get('online_train_loss'),
            online_train_mape=row.get('online_train_mape'),
            online_standardized_log_l1=row.get('online_standardized_log_l1'),
            **lr, lr_source='logged' if observed else 'derived from frozen recipe; not observed telemetry',
            gradient_norm_mean=row.get('gradient_norm_mean'), gradient_norm_max=row.get('gradient_norm_max'),
            clipping_fraction=row.get('gradient_clip_fraction'), updates=row['updates'], order_sha=row['order_sha']))
    return dict(config={k: manifest['recipe'][k] for k in (
            'seed', 'epochs', 'encoder_lr', 'head_lr', 'gradient_clip', 'effective_batch')},
        source_availability=availability, terminal20=final['score'],
        retrospective_best=dict(epoch=best['epoch'], score=best['score'], diagnostic_only=True,
            terminal_minus_best=final['score']['mean'] - best['score']['mean'],
            train_probe_at_best=best.get('train_probe_score'), train_probe_at_terminal=final.get('train_probe_score')),
        early_window=early, late_window=late, late_minus_early=difference(late, early),
        final_gradient_norm=final.get('gradient_norm_mean'), final_clipping_fraction=final.get('gradient_clip_fraction'),
        epoch_records=epochs)


def analyze(root):
    reader = Reader(root)
    audit = reader.json(SWEEP / 'audit.json')
    reader.check('sweep audit verified', audit['status'] == 'ssl5_lr_clip_sweep_verified')
    paths = {
        'head1e3_clip1': Path('results/phase2_lejepa5_regression_20261008'),
        'head3e4_clip1': Path('results/phase2_lejepa5_tuning_20261012/arms/full_head_low'),
        **{name: SWEEP / 'arms' / name for name in NEW_ARMS}}
    arms, manifests, histories = {}, {}, {}
    for name, directory in paths.items():
        manifest, history, availability = load_arm(reader, name, directory, name != 'head3e4_clip1')
        reader.check(name + ': final score matches sweep audit', history[-1]['score'] == audit['scores'][name])
        arms[name] = summarize(name, manifest, history, availability)
        manifests[name], histories[name] = manifest, history
    baseline = 'head1e3_clip1'
    for name in paths:
        reader.check(name + ': same validation and initialization', all(
            manifests[name][key] == manifests[baseline][key]
            for key in ('validation_ids_sha', 'ssl_checkpoint_sha', 'expected_head_sha')))
        reader.check(name + ': identical per-epoch ordering', all(
            left['order_sha'] == right['order_sha'] for left, right in zip(histories[name], histories[baseline])))
    log_records = []
    for line in reader.file(SWEEP / 'completed_training.log').open():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and 'epoch_complete' in row:
            log_records.append(clean_record(row))
    reader.check('sweep log has 80 epochs', len(log_records) == 80)
    for index, name in enumerate(NEW_ARMS):
        reader.check(name + ': original completed log equals copied history',
                     log_records[index * 20:(index + 1) * 20] == histories[name])
    candidate = 'head1e4_clip3'
    paired = {str(epoch): {key: histories[candidate][epoch - 1]['score'][key] -
                          histories[baseline][epoch - 1]['score'][key] for key in METRICS}
              for epoch in range(1, 21)}
    terminal_delta = paired['20']['mean']
    terminal_rank = sorted(arms, key=lambda name: arms[name]['terminal20']['mean'])
    last5_rank = sorted(arms, key=lambda name: arms[name]['late_window']['validation']['mean'])
    aggregate = dict(
        epochs_of_retrospective_minimum=dict(Counter(str(a['retrospective_best']['epoch']) for a in arms.values())),
        later_window_validation_mean_improved_arms=[n for n, a in arms.items() if a['late_minus_early']['validation']['mean'] < 0],
        later_window_target_improved_counts={k: sum(a['late_minus_early']['validation'][k] < 0 for a in arms.values()) for k in METRICS},
        later_window_train_probe_improved_arms=[n for n, a in arms.items()
            if a['late_minus_early']['train_probe'] is not None and a['late_minus_early']['train_probe']['mean'] < 0],
        terminal20_ranking=terminal_rank, last5_metric_average_ranking_diagnostic=last5_rank)
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),
        analysis_source_sha256=sha(Path(__file__)), status='fold3_exploratory_curve_analysis',
        primary_preserved='Fixed epoch20 mean of CD/CV/HEX MAPE on all 1791 fold3 validation images',
        no_fold4_outcomes_read=True, inputs=reader.inputs, evidence_checks=reader.checks,
        comparability=dict(fold=3, seed=42, validation_n=1791, train_n=7161,
            shared_ssl_checkpoint=manifests[baseline]['ssl_checkpoint_sha'],
            shared_validation_ids_sha=manifests[baseline]['validation_ids_sha'],
            same_initial_head=True, same_epoch_orders=True,
            fixed_probe_n=256, fixed_probe_seed=12012,
            baseline_train_probe_and_gradient_telemetry_unavailable=True,
            head3e4_clip1_local_evidence='logged history + manifest + verified sweep final audit; local per-arm result not present',
            window_definitions_are_exploratory=True), arms=arms, aggregate=aggregate,
        candidate_minus_baseline=dict(terminal20=paired['20'],
            relative_terminal_improvement_percent=-terminal_delta/arms[baseline]['terminal20']['mean']*100,
            mean_epoch16_20=arms[candidate]['late_window']['validation']['mean']-arms[baseline]['late_window']['validation']['mean'],
            epochs_candidate_lower=sum(row['mean'] < 0 for row in paired.values()),
            per_epoch=paired),
        interpretation=[
            'The six single-epoch minima are retrospective selections from reused, correlated validation curves; they are not six independent confirmations of an optimal stop epoch.',
            'The later fixed window improves mean validation in four of six arms; a universal deterioration after epoch7 is not supported.',
            'In all six arms CD and CV improve from epochs6-10 to16-20 while HEX worsens. In measured arms training-probe improvement can coexist with target-specific validation deterioration.',
            'Candidate terminal advantage is tiny and its last-five metric average is worse than baseline. The terminal20 rule remains unchanged; this window is only a stability diagnostic.',
            'Clipping frequency records how often the norm exceeds a threshold, not a proof that clipping is incorrect or the causal performance bottleneck.'],
        proposed_followup_decision=dict(status='proposal_only_pending_fold4_pair_and_curve_review',
            no_experiment_selected_or_launched=True,
            review_inputs=['fold4 matched terminal20 audit', 'per-target and probe curves',
                           'same-window summaries rather than one chosen minimum', 'time before deadline'],
            branches=[
                dict(condition='Both fold4 arms show worse late-window overall MAPE while probe and training loss improve; worsening is not confined to a single lucky earlier epoch.',
                     proposal='Prioritize a predeclared shorter fit with unchanged schedule horizon to isolate stopping duration; compare under a new paired seed.',
                     distinction='epochs10/schedule20 tests duration; epochs10/schedule10 also changes LR trajectory and is a different intervention.'),
                dict(condition='Fold4 repeats the fold3 pattern: later CD/CV gain, HEX worsening, overall late-window flat or improved.',
                     proposal='Prefer testing earlier decay at fixed total exposure over blindly stopping at epoch7: one candidate is epochs20/schedule10 with the existing LR floor, versus epochs20/schedule20, same new seed.',
                     rationale='Preserves opportunities for CD/CV improvement while testing whether lower late updates reduce HEX deterioration; this is a hypothesis, not an expected gain.'),
                dict(condition='Candidate/baseline ranking reverses across folds or depends on a single terminal fluctuation, without a consistent shared schedule signal.',
                     proposal='Prioritize one matched new-seed baseline/candidate pair before expanding schedules.'),
                dict(condition='Remaining time is insufficient for the complete paired experiment plus audit under the maximum2 H200 policy.',
                     proposal='Do not start an incomplete comparison; retain verified single models and finish evidence collection.')],
            constraints=['No final schedule choice before fold4 results are reviewed.',
                'Declare terminal epoch, LR horizon, fold, seed and selection rule before launching the follow-up.',
                'Keep total concurrent H200 <=2 and finish by 2026-10-19T03:59:00Z.',
                'Verify a promising follow-up on another seed/fold; reused fold3/fold4 remain development evidence.']),
        limitations=['No prediction-level significance or bootstrap computed; aggregate histories cannot establish sampling uncertainty.',
            'Epoch values share the same validation images and a training trajectory; epoch standard deviation is not a confidence interval.',
            'The train probe is 256 fixed training images, not the entire training distribution; absolute train/validation gaps mix population differences.',
            'Online training metrics average augmented examples under changing weights and are distinct from checkpoint evaluation.',
            'This analysis does not load model weights or independently recompute per-image predictions.',
            'Patient independence and near-duplicate leakage remain unresolved.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--out', type=Path, default=Path('results/phase2_crossfold_20261015/fold3_curve_analysis.json'))
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out if args.out.is_absolute() else root / args.out
    if out.exists() or not out.parent.is_dir():
        parser.error('Output must be a new file with an existing parent directory')
    report = analyze(root)
    with out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], output=str(out),
        all_evidence_checks_passed=all(c['passed'] for c in report['evidence_checks']),
        aggregate=report['aggregate'], candidate_terminal_delta=report['candidate_minus_baseline']['terminal20'],
        followup_status=report['proposed_followup_decision']['status'])))


if __name__ == '__main__':
    main()
