"""Read of one encoder arm against its matched reference (lane GPU 0, copied to lane GPU 2).
Rule: mean of the last five epochs on each fold, paired on the same frames, CI from donor bootstrap across all folds read. A fold read needs epochs 16-20; an arm with fewer finished epochs prints a partial line only.
Added read: new member added to the best pipeline, on the rows of the folds read: H x exp(w * (log new - log recipe A TTA)), w = 0.25 and 0.125 fixed in the gate, stretch fit on the other folds as in best_pipeline.
Usage: read_arm_20261106.py <name> <arm root pattern with {f}> [folds, default 0,4] [--ref <pattern with {f}>]"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E

X = Path('/workspace/clear-ec-research/experiments')
REF = {0: 'phase2_pooling_replication_20261026/f0_s46_std', 4: 'phase2_pooling_20261020/arms/std'}
K = ['CD', 'CV', 'HEX']


def epochs(root, f):
    run = X / root / 'runs' / f'fold{f}' / 'cpdino_vitb'
    return sorted(run.glob('predictions_epoch_*.npz')), run


def load_last5(root, f, n_total=None):
    files, run = epochs(root, f)
    if not files:
        return None, 0
    last = int(files[-1].stem[-2:])
    use = [p for p in files if int(p.stem[-2:]) > last - 5]
    if len(use) < 5:
        return None, last
    out = []
    for p in use:
        with np.load(p, allow_pickle=True) as z:
            out.append((z['indices'].astype(int), z['pred'].astype(np.float64), z['target'].astype(np.float64)))
    return out, last


def stats_of(arm):
    ix, _, y = arm[0]
    e = np.mean([E.ape(p, y) for _, p, _ in arm], 0)
    return ix, y, e, np.mean([np.log(p) for _, p, _ in arm], 0)


def main():
    name, pat = sys.argv[1], sys.argv[2]
    folds = [int(v) for v in sys.argv[3].split(',')] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else [0, 4]
    refpat = sys.argv[sys.argv.index('--ref') + 1] if '--ref' in sys.argv else None
    b = E.load_base(); gp = E.donors(b['ids']); pos = {int(i): n for n, i in enumerate(b['idx'])}
    res = dict(name=name, folds={}); E1, E0, G = [], [], []; rows_all = []; logs = []
    for f in folds:
        arm, last = load_last5(pat.format(f=f), f)
        ref, _ = load_last5(refpat.format(f=f) if refpat else REF[f], f)
        files, run = epochs(pat.format(f=f), f)
        if arm is None:
            res['folds'][f] = dict(status='partial', last_epoch=last)
            continue
        ix, y, e1, lg = stats_of(arm); ix0, y0, e0, _ = stats_of(ref)
        assert np.array_equal(ix, ix0) and np.array_equal(y, y0)
        rows = np.array([pos[int(i)] for i in ix])
        s1, s0 = E.official(e1), E.official(e0)
        hist = json.loads((run / 'history.json').read_text())
        res['folds'][f] = dict(last_epoch=last, arm={k: round(s1[k], 4) for k in K + ['mean']}, ref={k: round(s0[k], 4) for k in K + ['mean']},
                               delta=round(s1['mean'] - s0['mean'], 4), per={k: round(s1[k] - s0[k], 4) for k in K},
                               curve=[round(h['score']['mean'], 3) for h in hist], train_probe=round(hist[-1]['train_probe_score']['mean'], 3))
        E1.append(e1); E0.append(e0); G.append(gp[rows]); rows_all.append(rows); logs.append(lg)
    if E1:
        e1, e0, g = np.concatenate(E1), np.concatenate(E0), np.concatenate(G)
        pb = E.paired_bootstrap(e1, e0, g)
        s1, s0 = E.official(e1), E.official(e0)
        res['pooled'] = dict(n=len(e1), delta=round(pb['delta'], 4), ci95=[round(v, 4) for v in pb['ci95']], p_better=round(pb['p_better'], 3), per={k: round(s1[k] - s0[k], 4) for k in K})
    print(json.dumps(res, indent=1), flush=True)
    out = X / 'lane_gpu2_20261106' / 'reads'; out.mkdir(exist_ok=True)
    (out / f'{name}.json').write_text(json.dumps(res, indent=1))
    if E1 and '--added' in sys.argv:
        sys.path.insert(0, str(X / 'meta_ceiling_20261103/source'))
        import best_pipeline_20261106 as BP
        rows = np.concatenate(rows_all); lg = np.concatenate(logs)
        A = BP._b['A']
        assert np.array_equal(BP.idx, b['idx'])
        added = {}
        for w in (0.125, 0.25, 0.5):
            c = np.zeros_like(BP.H); c[rows] = w * (lg - np.log(A[rows]))
            P = BP.pipe(BP.add(BP.BEST, {ff: c for ff in range(5)}))
            eb, er = E.ape(P[rows], BP.y[rows]), E.ape(BP.REF_BEST[rows], BP.y[rows])
            pb = E.paired_bootstrap(eb, er, gp[rows]); s1, s0 = E.official(eb), E.official(er)
            added[str(w)] = dict(delta=round(pb['delta'], 4), ci95=[round(v, 4) for v in pb['ci95']], per={k: round(s1[k] - s0[k], 4) for k in K}, best_on_rows=round(s0['mean'], 4))
        res['added'] = added
        print(json.dumps(dict(added=added), indent=1), flush=True)
        (out / f'{name}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
