"""Aggregate diagnostics of existing fold3 predictions; no model fitting/selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path) as z:
        out = {k: z[k].copy() for k in ('indices', 'ids', 'target', 'pred')}
    assert out['pred'].shape == out['target'].shape == (1791, 3)
    assert np.isfinite(out['pred']).all() and np.isfinite(out['target']).all()
    assert (out['target'] > 0).all()
    return out


def connected(slides, fingerprints):
    parent, seen = {str(s): str(s) for s in slides}, {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for slide, fingerprint in zip(slides, fingerprints):
        slide, fingerprint = str(slide), str(fingerprint)
        if fingerprint in seen:
            parent[find(slide)] = find(seen[fingerprint])
        else:
            seen[fingerprint] = slide
    return np.asarray([find(str(s)) for s in slides])


def main(root):
    baseline = root/'experiments/phase2_lejepa5_regression_20261008/runs/fold3/cpdino_vitb'
    candidate = root/'experiments/phase2_lejepa5_sweep_20261013/arms/head1e4_clip3/runs/fold3/cpdino_vitb'
    paths = {'baseline20': baseline/'predictions.npz', 'candidate20': candidate/'predictions.npz',
             'candidate7': candidate/'predictions_epoch_07.npz'}
    records = {name: load(path) for name, path in paths.items()}
    reference = records['baseline20']
    for record in records.values():
        for key in ('indices', 'ids', 'target'):
            np.testing.assert_array_equal(record[key], reference[key])
    for run in (baseline, candidate):
        result = json.loads((run/'result.json').read_text())
        complete = json.loads((run/'complete.json').read_text())
        assert sha(run/'predictions.npz') == result['predictions_sha']
        assert sha(run/'result.json') == complete['result_sha']
    metadata = root/'datasets/clear_ec_official/cache_native_meta_512.npz'
    fingerprints = root/'experiments/ssl_screen_20260911/preflight/pixel_fingerprints.npz'
    labels = root/'datasets/clear_ec_official/final_train_ids.csv'
    with np.load(metadata, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y']
    np.testing.assert_array_equal(ids[reference['indices']], reference['ids'])
    np.testing.assert_array_equal(y[reference['indices']], reference['target'])
    with labels.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    by_id = {}
    for row in rows:
        key = Path(row['ID']).stem
        assert key not in by_id
        by_id[key] = row
    slides = np.array([str(by_id[Path(i).stem]['slide_id']).strip() for i in ids])
    with np.load(fingerprints) as z:
        pixel_hashes = z['sha256'].astype(str)
    assert len(pixel_hashes) == len(ids) and all(slides)
    groups = connected(slides, pixel_hashes)[reference['indices']]
    names, inverse = np.unique(groups, return_inverse=True)
    group_n = np.bincount(inverse)
    errors = {k: abs(v['pred'].astype('float64')/v['target'] - 1)*100 for k,v in records.items()}
    delta = errors['candidate20']-errors['baseline20']
    grouped = np.column_stack([np.bincount(inverse, weights=delta[:,i]) for i in range(3)])
    rng = np.random.default_rng(15102026)
    samples = []
    for _ in range(2000):
        selected = rng.integers(0, len(names), len(names))
        values = grouped[selected].sum(0)/group_n[selected].sum()
        samples.append([*values, values.mean()])
    samples = np.asarray(samples)
    intervals = np.quantile(samples, [.025,.975], axis=0)
    metrics = ['CD','CV','HEX']
    bins = {}
    for col, metric in enumerate(metrics):
        values = reference['target'][:,col]
        edges = np.quantile(values, [.25,.5,.75])
        which = np.searchsorted(edges, values, side='right')
        bins[metric] = []
        for i in range(4):
            mask = which == i
            if not mask.any():
                continue
            bins[metric].append(dict(bin=i+1,n=int(mask.sum()),target_min=float(values[mask].min()),
                target_max=float(values[mask].max()),
                mape={key:float(value[mask,col].mean()) for key,value in errors.items()},
                candidate20_minus_baseline20=float(delta[mask,col].mean()),
                candidate20_minus_candidate7=float((errors['candidate20']-errors['candidate7'])[mask,col].mean())))
    result = dict(status='paired_existing_prediction_diagnostics',fold=3,n=len(delta),
        connected_slide_exact_groups=len(names),bootstrap_replicates=2000,bootstrap_seed=15102026,
        scores_float64={k:dict(zip(metrics+['mean'],[*v.mean(0).tolist(),float(v.mean())])) for k,v in errors.items()},
        candidate20_minus_baseline20=dict(zip(metrics+['mean'],[*delta.mean(0).tolist(),float(delta.mean())])),
        exploratory_cluster_bootstrap_95_percent_interval={k:intervals[:,i].tolist() for i,k in enumerate(metrics+['mean'])},
        target_quartile_bins=bins,inputs={str(p.relative_to(root)):sha(p) for p in [*paths.values(),metadata,fingerprints,labels]},
        limitations=['Exploratory uncertainty after selection among six arms; not a confirmatory confidence interval',
                    'Cluster unit is slide/exact-pixel connected group, not verified patient',
                    'Does not measure random-training-seed variability or external generalization',
                    'Epoch7 was chosen retrospectively; diagnostic only',
                    'Float64 reaggregation differs slightly from primary float32 logged scores'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
