"""Read of one family arm against the matched recipe A reference (lane GPU 3, 2026-11-06). Same rule and added formula as lane GPU 0's read_arm_20261106.py, so lanes compare.
Member read: mean of the epoch 16-20 errors per row on each fold, paired on the same frames, CI from donor bootstrap over all folds read. A fold without epochs 16-20 prints a partial line.
Added read: H x exp(w (log new - log recipe A TTA)) on the rows of the folds read, w 0.125, 0.25 (gate) and 0.5 (diagnostic), stretch from best_pipeline. Two forms: new = log mean of the last five epochs, and each of the five epochs alone with errors averaged (no checkpoint ensemble).
Usage: read_family_20261106.py <name> <arm root pattern with {f}> [--folds 0,4] [--ref pattern] [--added]"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E

X = Path('/workspace/clear-ec-research/experiments')
LANE = X / 'lane_gpu3_20261106'
REF = {0: 'phase2_pooling_replication_20261026/f0_s46_std', 4: 'phase2_pooling_20261020/arms/std'}
K = ['CD', 'CV', 'HEX']


def run_dir(root, f):
    d = sorted((X / root / 'runs' / f'fold{f}').glob('*'))
    return d[0] if d else None


def last5(root, f):
    run = run_dir(root, f)
    files = sorted(run.glob('predictions_epoch_*.npz')) if run else []
    if not files:
        return None, 0, run
    last = int(files[-1].stem[-2:])
    use = [p for p in files if int(p.stem[-2:]) > last - 5]
    if len(use) < 5:
        return None, last, run
    out = []
    for p in use:
        with np.load(p, allow_pickle=True) as z:
            out.append((z['indices'].astype(int), z['pred'].astype(np.float64), z['target'].astype(np.float64)))
    return out, last, run


def main():
    name, pat = sys.argv[1], sys.argv[2]
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    folds = [int(v) for v in arg('--folds', '0,4').split(',')]; refpat = arg('--ref')
    b = E.load_base(); gp = E.donors(b['ids']); pos = {int(i): n for n, i in enumerate(b['idx'])}
    res = dict(name=name, folds={}); E1, E0, G, rows_all, logs, single = [], [], [], [], [], []
    for f in folds:
        arm, last, run = last5(pat.format(f=f), f)
        ref, _, _ = last5(refpat.format(f=f) if refpat else REF[f], f)
        if arm is None:
            res['folds'][f] = dict(status='partial', last_epoch=last); continue
        ix, y = arm[0][0], arm[0][2]; ix0, y0 = ref[0][0], ref[0][2]
        assert np.array_equal(ix, ix0) and np.array_equal(y, y0)
        keep = np.array([int(i) in pos for i in ix])
        ix, y = ix[keep], y[keep]
        e1 = np.mean([E.ape(p[keep], y) for _, p, _ in arm], 0); e0 = np.mean([E.ape(p[keep], y) for _, p, _ in ref], 0)
        lg = np.mean([np.log(p[keep]) for _, p, _ in arm], 0); lg0 = np.mean([np.log(p[keep]) for _, p, _ in ref], 0)
        rows = np.array([pos[int(i)] for i in ix]); s1, s0 = E.official(e1), E.official(e0)
        hist = json.loads((run / 'history.json').read_text())
        ry = np.log(y)
        res['folds'][f] = dict(last_epoch=last, n=len(ix), arm={k: round(s1[k], 4) for k in K + ['mean']}, ref={k: round(s0[k], 4) for k in K + ['mean']}, delta=round(s1['mean'] - s0['mean'], 4), per={k: round(s1[k] - s0[k], 4) for k in K},
                               residual_r_with_ref={k: round(float(np.corrcoef(lg[:, j] - ry[:, j], lg0[:, j] - ry[:, j])[0, 1]), 3) for j, k in enumerate(K)}, curve=[round(h['score']['mean'], 3) for h in hist],
                               train_probe=round(hist[-1]['train_probe_score']['mean'], 3), seconds_per_epoch=round(float(np.mean([h['epoch_seconds'] for h in hist]))))
        E1.append(e1); E0.append(e0); G.append(gp[rows]); rows_all.append(rows); logs.append(lg); single.append(np.stack([np.log(p[keep]) for _, p, _ in arm]))
    if E1:
        e1, e0, g = np.concatenate(E1), np.concatenate(E0), np.concatenate(G)
        pb = E.paired_bootstrap(e1, e0, g); s1, s0 = E.official(e1), E.official(e0)
        res['pooled'] = dict(n=len(e1), delta=round(pb['delta'], 4), ci95=[round(v, 4) for v in pb['ci95']], p_better=round(pb['p_better'], 3), per={k: round(s1[k] - s0[k], 4) for k in K})
    print(json.dumps(res), flush=True)
    out = LANE / 'reads'; out.mkdir(exist_ok=True); (out / f'{name}.json').write_text(json.dumps(res, indent=1))
    if E1 and '--added' in sys.argv:
        sys.path.insert(0, str(X / 'meta_ceiling_20261103/source'))
        import best_pipeline_20261106 as BP
        rows = np.concatenate(rows_all); lg = np.concatenate(logs); sg = np.concatenate(single, 1); A = BP._b['A']
        assert np.array_equal(BP.idx, b['idx'])
        er = E.ape(BP.REF_BEST[rows], BP.y[rows]); added = {}
        def err(l, w):
            c = np.zeros_like(BP.H); c[rows] = w * (l - np.log(A[rows]))
            return E.ape(BP.pipe(BP.add(BP.BEST, {ff: c for ff in range(5)}))[rows], BP.y[rows])
        for w in (0.125, 0.25, 0.5):
            eb = err(lg, w); pb = E.paired_bootstrap(eb, er, gp[rows]); s1, s0 = E.official(eb), E.official(er)
            es = np.mean([err(sg[i], w) for i in range(len(sg))], 0); pbs = E.paired_bootstrap(es, er, gp[rows]); s2 = E.official(es)
            added[str(w)] = dict(last5_log_mean=dict(delta=round(pb['delta'], 4), ci95=[round(v, 4) for v in pb['ci95']], per={k: round(s1[k] - s0[k], 4) for k in K}),
                                 each_epoch=dict(delta=round(pbs['delta'], 4), ci95=[round(v, 4) for v in pbs['ci95']], per={k: round(s2[k] - s0[k], 4) for k in K}), best_on_rows=round(s0['mean'], 4))
        res['added'] = added
        print(json.dumps(dict(added=added)), flush=True); (out / f'{name}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
