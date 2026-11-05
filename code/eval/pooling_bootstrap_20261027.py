"""Paired cluster bootstrap for the four std-vs-mean pooling pairs; CPU only.

Per pair: per-image difference of the three-target mean APE (std minus mean),
resampled 10,000 times over id-prefix groups (slide-like clusters; not verified
patients). Reports the point delta, 95% percentile interval, P(delta < 0), and
the pooled estimate over all four pairs (groups resampled within each pair).
"""
import json
from pathlib import Path

import numpy as np

E = Path('experiments')
PAIRS = {
    'fold4_seed46': (E/'phase2_pooling_20261020/arms/mean', E/'phase2_pooling_20261020/arms/std'),
    'fold0_seed46': (E/'phase2_backbone_fold0_regression_ssl5_20261022', E/'phase2_pooling_replication_20261026/f0_s46_std'),
    'fold4_seed47': (E/'phase2_pooling_replication_20261026/f4_s47_mean', E/'phase2_pooling_replication_20261026/f4_s47_std'),
    'fold0_seed47': (E/'phase2_pooling_replication_20261026/f0_s47_mean', E/'phase2_pooling_replication_20261026/f0_s47_std'),
}
B, SEED = 10000, 20261027


def load(root):
    fold = json.loads((root/'manifest.json').read_text())['fold']
    with np.load(root/'runs'/f'fold{fold}'/'cpdino_vitb'/'predictions.npz') as z:
        return z['ids'].astype(str), z['target'].astype(np.float64), z['pred'].astype(np.float64)


def ape(pred, y):
    return (np.abs(pred-y)/y*100)


def cluster_bootstrap(diff, groups, rng):
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=diff, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    draws = rng.integers(0, len(uniq), size=(B, len(uniq)))
    return (sums[draws].sum(1)/counts[draws].sum(1))


def main():
    rng = np.random.default_rng(SEED)
    out, pooled = {}, []
    for name, (m, s) in PAIRS.items():
        ids_m, y_m, p_m = load(m)
        ids_s, y_s, p_s = load(s)
        assert (ids_m == ids_s).all() and np.array_equal(y_m, y_s)
        diff_targets = ape(p_s, y_s)-ape(p_m, y_m)
        diff = diff_targets.mean(1)
        groups = np.array([i.rsplit('-', 1)[0] for i in ids_m])
        boot = cluster_bootstrap(diff, groups, rng)
        pooled.append(boot)
        out[name] = dict(n_images=len(diff), n_groups=len(np.unique(groups)),
                         delta_mean_pp=float(diff.mean()), ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                         p_improve=float((boot < 0).mean()),
                         delta_by_target_pp=dict(zip(('CD', 'CV', 'HEX'), diff_targets.mean(0).tolist())),
                         ci95_by_target=[[float(np.percentile(cluster_bootstrap(diff_targets[:, j], groups, rng), q)) for q in (2.5, 97.5)] for j in range(3)])
    pooled = np.mean(pooled, 0)
    result = dict(method='paired cluster bootstrap over id-prefix groups, 10,000 resamples, percentile interval',
                  pairs=out, pooled=dict(delta_mean_pp=float(np.mean([v['delta_mean_pp'] for v in out.values()])),
                                         ci95=[float(np.percentile(pooled, 2.5)), float(np.percentile(pooled, 97.5))],
                                         p_improve=float((pooled < 0).mean()), sign_test_one_sided_p=0.5**len(out)),
                  limits=['Groups are id prefixes, not verified patients', 'Four pairs on two reused development folds; the recipe decision is already taken'])
    Path(E/'phase2_pooling_replication_20261026'/'bootstrap.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: (v['delta_mean_pp'], v['ci95'], v['p_improve']) for k, v in out.items()}), json.dumps(result['pooled']))


if __name__ == '__main__':
    main()
