"""Verify and compare two independently scored single-model regression arms."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def compare(baseline, candidate, output):
    if output.exists():
        raise FileExistsError('Never overwrite a pair comparison')
    records, inputs, fields = {}, {}, None
    for label, root in [('baseline', baseline), ('candidate', candidate)]:
        manifest = read(root/'manifest.json')
        audit = read(root/'audit.json')
        run = root/'runs'/f'fold{manifest["fold"]}'/'cpdino_vitb'
        result = read(run/'result.json')
        complete = read(run/'complete.json')
        assert audit['status'] == 'single_model_crossfold_regression_verified'
        assert audit['manifest_sha'] == sha(root/'manifest.json') == result['manifest_sha']
        assert complete['status'] == 'completed' and complete['result_sha'] == sha(run/'result.json')
        assert sha(run/'predictions.npz') == result['predictions_sha']
        assert sha(run/'last.pt') == result['checkpoint_sha']
        assert result['single_model'] and result['ranks_synchronized']
        with np.load(run/'predictions.npz') as z:
            assert z['pred'].shape == z['target'].shape == (manifest['validation_n'], 3)
            assert z['ids'].shape == z['indices'].shape == (manifest['validation_n'],)
            assert np.isfinite(z['target']).all()
            np.testing.assert_array_equal(z['indices'], manifest['validation_indices'])
            assert hashlib.sha256('\n'.join(z['ids'].astype(str)).encode()).hexdigest() == manifest['validation_ids_sha']
            current = {k: z[k].copy() for k in ['indices', 'ids', 'target']}
            if fields is None:
                fields = current
            for key in current:
                np.testing.assert_array_equal(current[key], fields[key])
            assert np.isfinite(z['pred']).all() and (z['target'] > 0).all()
            values = (np.abs(z['pred']-z['target'])/z['target']).mean(0)*100
            score = dict(zip(['CD', 'CV', 'HEX'], map(float, values)))
            score.update(mean=float(values.mean()), n=len(z['pred']))
        for key in score:
            assert abs(score[key]-result['score'][key]) < 1e-6
            assert abs(score[key]-audit['score'][key]) < 1e-6
        for file in [root/'manifest.json', root/'audit.json', run/'result.json', run/'complete.json', run/'predictions.npz']:
            inputs[str(file)] = sha(file)
        records[label] = dict(root=str(root), score=score, manifest=manifest,
                              history=result['history'], best=audit['retrospective_best'],
                              seconds=result['seconds'], historical_subset_score=result['historical_subset_score'])
    left, right = records['baseline']['manifest'], records['candidate']['manifest']
    for key in ['fold', 'seed', 'split_sha256', 'train_ids_sha', 'validation_ids_sha',
                'ssl_checkpoint_sha', 'expected_head_sha', 'log_mean', 'log_std']:
        assert left[key] == right[key], key
    for key in set(left['recipe']) | set(right['recipe']):
        if key not in ['head_lr', 'gradient_clip']:
            assert left['recipe'][key] == right['recipe'][key], key
    for l, r in zip(records['baseline']['history'], records['candidate']['history']):
        for key in ['epoch', 'updates', 'order_sha']:
            assert l[key] == r[key], key
    assert len(records['baseline']['history']) == len(records['candidate']['history']) == left['recipe']['epochs']
    scores = {k: r['score'] for k, r in records.items()}
    delta = {k: scores['candidate'][k]-scores['baseline'][k] for k in ['CD', 'CV', 'HEX', 'mean']}
    report = dict(status='matched_single_model_pair_verified', checked_utc=time.strftime('%FT%TZ', time.gmtime()),
                  fold=left['fold'], seed=left['seed'], primary=f'fixed terminal epoch{left["recipe"]["epochs"]}',
                  scores=scores, candidate_minus_baseline=delta, lower_is_better=True,
                  ranking=sorted(scores, key=lambda k: (scores[k]['mean'], k)),
                  same_ssl_head_initialization_data_order_and_validation=True, single_model_only=True,
                  configurations={k: {x:r['manifest']['recipe'][x] for x in ['head_lr','gradient_clip','epochs','schedule_epochs']} for k,r in records.items()},
                  retrospective_best={k:r['best'] for k,r in records.items()},
                  historical_subset_scores={k:r['historical_subset_score'] for k,r in records.items()},
                  training_seconds={k:r['seconds'] for k,r in records.items()},
                  source_sha256=sha(Path(__file__)), inputs=inputs,
                  limitations=['Reused development fold; not an independent official score',
                               'A small single-seed difference does not establish a stable improvement',
                               'Intermediate best epochs are retrospective diagnostics, not primary selection'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    compare(args.baseline, args.candidate, args.output)
