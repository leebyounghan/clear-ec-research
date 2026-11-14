"""Backbone lane read (2026-11-08). Same rule and added formula as lane GPU 3's read_family_20261106.py, with an epoch window.
Member read: mean of the per-epoch errors over epochs LO..HI per row, each fold, paired on the same frames with recipe A
(seed 46, same window); CI from donor bootstrap over the folds read. Also against the mean of recipe A seeds 46-48.
Target: the lane target is the two-fold mean of the member read (recipe A seed 46, epochs 16-20: 9.046).
Added read: best pipeline H x exp(w (log new - log recipe A TTA)), w 0.125 / 0.25, new = log mean over the window
(checkpoint ensemble) and each epoch alone with errors averaged.
Usage: read_bb_20261108.py <name> <run root pattern with {f}, relative to experiments/> [--folds 0,4] [--window 8,12] [--added]"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E

X = Path('/workspace/clear-ec-research/experiments')
LANE = X / 'lane_backbone_20261108'
REF = {0: {46: 'phase2_pooling_replication_20261026/f0_s46_std', 47: 'phase2_pooling_replication_20261026/f0_s47_std', 48: 'oof_recipeA_20261027/fold0/regression_std_s48'},
       4: {46: 'phase2_pooling_20261020/arms/std', 47: 'phase2_pooling_replication_20261026/f4_s47_std', 48: 'oof_recipeA_20261027/fold4/regression_std_s48'}}
K = ['CD', 'CV', 'HEX']


def preds(root, f, lo, hi):
    d = sorted((X / root / 'runs' / f'fold{f}').glob('*'))
    if not d:
        return None, 0, None
    run = d[0]; files = sorted(run.glob('predictions_epoch_*.npz')); have = [int(p.stem[-2:]) for p in files]
    use = [p for p in files if lo <= int(p.stem[-2:]) <= hi]
    if len(use) < hi - lo + 1:
        return None, max(have) if have else 0, run
    out = []
    for p in use:
        with np.load(p, allow_pickle=True) as z:
            out.append((z['indices'].astype(int), z['pred'].astype(np.float64), z['target'].astype(np.float64)))
    return out, max(have), run


def main():
    name, pat = sys.argv[1], sys.argv[2]
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    folds = [int(v) for v in arg('--folds', '0,4').split(',')]; lo, hi = [int(v) for v in arg('--window', '8,12').split(',')]
    b = E.load_base(); gp = E.donors(b['ids']); pos = {int(i): n for n, i in enumerate(b['idx'])}
    res = dict(name=name, pattern=pat, window=[lo, hi], folds={}); E1, E0, E3, G, rows_all, logs, single = [], [], [], [], [], [], []
    for f in folds:
        arm, last, run = preds(pat.format(f=f), f, lo, hi)
        if arm is None:
            res['folds'][f] = dict(status='partial', last_epoch=last); continue
        refs = {s: preds(r, f, lo, hi)[0] for s, r in REF[f].items()}
        ix, y = arm[0][0], arm[0][2]
        for s, rr in refs.items():
            assert np.array_equal(rr[0][0], ix) and np.array_equal(rr[0][2], y)
        keep = np.array([int(i) in pos for i in ix]); ix, y = ix[keep], y[keep]
        mem = lambda P: np.mean([E.ape(p[keep], y) for _, p, _ in P], 0)
        e1, e0 = mem(arm), mem(refs[46]); e3 = np.mean([mem(refs[s]) for s in refs], 0)
        lg = np.mean([np.log(p[keep]) for _, p, _ in arm], 0); lg0 = np.mean([np.log(p[keep]) for _, p, _ in refs[46]], 0)
        rows = np.array([pos[int(i)] for i in ix]); s1, s0, s3 = E.official(e1), E.official(e0), E.official(e3)
        hist = json.loads((run / 'history.json').read_text()); ry = np.log(y)
        ens = E.official(E.ape(np.exp(lg), y))
        res['folds'][f] = dict(last_epoch=last, n=len(ix), arm={k: round(s1[k], 4) for k in K + ['mean']}, ref46={k: round(s0[k], 4) for k in K + ['mean']},
                               ref3seeds=round(s3['mean'], 4), delta46=round(s1['mean'] - s0['mean'], 4), delta3=round(s1['mean'] - s3['mean'], 4),
                               per46={k: round(s1[k] - s0[k], 4) for k in K}, window_log_mean_ensemble=round(ens['mean'], 4),
                               residual_r_with_ref={k: round(float(np.corrcoef(lg[:, j] - ry[:, j], lg0[:, j] - ry[:, j])[0, 1]), 3) for j, k in enumerate(K)},
                               curve=[round(h['score']['mean'], 3) for h in hist], train_probe=[round(h['train_probe_score']['mean'], 3) for h in hist if 'train_probe_score' in h][-1:],
                               seconds_per_epoch=round(float(np.mean([h['train_seconds'] + h.get('eval_seconds', 0) for h in hist]))))
        E1.append(e1); E0.append(e0); E3.append(e3); G.append(gp[rows]); rows_all.append(rows); logs.append(lg); single.append(np.stack([np.log(p[keep]) for _, p, _ in arm]))
    if E1:
        e1, e0, e3, g = np.concatenate(E1), np.concatenate(E0), np.concatenate(E3), np.concatenate(G)
        pb = E.paired_bootstrap(e1, e0, g); pb3 = E.paired_bootstrap(e1, e3, g); s1, s0 = E.official(e1), E.official(e0)
        fm = [res['folds'][f]['arm']['mean'] for f in folds if 'arm' in res['folds'].get(f, {})]
        res['pooled'] = dict(n=len(e1), folds_read=len(fm), two_fold_mean_of_member=round(float(np.mean(fm)), 4), delta46=round(pb['delta'], 4), ci95=[round(v, 4) for v in pb['ci95']],
                             delta3=round(pb3['delta'], 4), ci95_3=[round(v, 4) for v in pb3['ci95']], per46={k: round(s1[k] - s0[k], 4) for k in K})
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
            eb = err(lg, w); pbb = E.paired_bootstrap(eb, er, gp[rows]); s1, s0 = E.official(eb), E.official(er)
            es = np.mean([err(sg[i], w) for i in range(len(sg))], 0); pbs = E.paired_bootstrap(es, er, gp[rows]); s2 = E.official(es)
            added[str(w)] = dict(window_log_mean=dict(delta=round(pbb['delta'], 4), ci95=[round(v, 4) for v in pbb['ci95']], per={k: round(s1[k] - s0[k], 4) for k in K}),
                                 each_epoch=dict(delta=round(pbs['delta'], 4), ci95=[round(v, 4) for v in pbs['ci95']], per={k: round(s2[k] - s0[k], 4) for k in K}), best_on_rows=round(s0['mean'], 4))
        res['added'] = added
        print(json.dumps(dict(added=added)), flush=True); (out / f'{name}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
