"""Combined read of family arms as one added member (lane GPU 3, gate in protocol.json, amendment 2026-11-07 15:50Z).
Member: equal-weight log mean of the arms, last five epochs, folds 0 and 4, seed 46. Added: H x exp(w (log member - log recipe A TTA)), w 0.125 and 0.25 (gate), 0.5 (diagnostic). Per-epoch guard: epochs 16-20 one at a time, matched by epoch number across arms, errors averaged.
Usage: read_combined_20261107.py <name> <arm pattern with {f}> [<arm pattern> ...]"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import read_family_20261106 as R

E, X, LANE, K = R.E, R.X, R.LANE, R.K
FOLDS = (0, 4)


def main():
    name, pats = sys.argv[1], sys.argv[2:]
    b = E.load_base(); gp = E.donors(b['ids']); pos = {int(i): n for n, i in enumerate(b['idx'])}
    rows_all, by_fold = [], {}
    for f in FOLDS:
        arms = [R.last5(p.format(f=f), f)[0] for p in pats]
        ref, _, _ = R.last5(R.REF[f], f)
        ix, y = ref[0][0], ref[0][2]
        for a in arms:
            assert len(a) == 5
            for t in a:
                assert np.array_equal(t[0], ix) and np.array_equal(t[2], y)
        keep = np.array([int(i) in pos for i in ix])
        single = np.mean([np.stack([np.log(t[1][keep]) for t in a]) for a in arms], 0)
        by_fold[f] = dict(single=single, y=y[keep], ref=np.stack([np.log(t[1][keep]) for t in ref]), ix=ix[keep])
        rows_all.append(np.array([pos[int(i)] for i in ix[keep]]))
    rows = np.concatenate(rows_all); sg = np.concatenate([by_fold[f]['single'] for f in FOLDS], 1)
    lg = sg.mean(0); y = np.concatenate([by_fold[f]['y'] for f in FOLDS])
    rf = np.concatenate([by_fold[f]['ref'] for f in FOLDS], 1).mean(0)
    e1, e0 = E.ape(np.exp(lg), y), E.ape(np.exp(rf), y); pb = E.paired_bootstrap(e1, e0, gp[rows]); s1, s0 = E.official(e1), E.official(e0)
    res = dict(name=name, arms=pats, folds=list(FOLDS), n=len(rows), member=dict(score={k: round(s1[k], 4) for k in K + ['mean']}, ref_log_mean={k: round(s0[k], 4) for k in K + ['mean']}, delta=round(pb['delta'], 4), ci95=[round(v, 4) for v in pb['ci95']]),
               residual_r_with_ref={k: round(float(np.corrcoef(lg[:, j] - np.log(y[:, j]), rf[:, j] - np.log(y[:, j]))[0, 1]), 3) for j, k in enumerate(K)})
    sys.path.insert(0, str(X / 'meta_ceiling_20261103/source'))
    import best_pipeline_20261106 as BP
    assert np.array_equal(BP.idx, b['idx'])
    A = BP._b['A']; er = E.ape(BP.REF_BEST[rows], BP.y[rows]); added = {}
    def err(l, w):
        c = np.zeros_like(BP.H); c[rows] = w * (l - np.log(A[rows]))
        return E.ape(BP.pipe(BP.add(BP.BEST, {ff: c for ff in range(5)}))[rows], BP.y[rows])
    for w in (0.125, 0.25, 0.5):
        eb = err(lg, w); p = E.paired_bootstrap(eb, er, gp[rows]); a1, a0 = E.official(eb), E.official(er)
        es = np.mean([err(sg[i], w) for i in range(len(sg))], 0); ps = E.paired_bootstrap(es, er, gp[rows]); a2 = E.official(es)
        added[str(w)] = dict(last5_log_mean=dict(delta=round(p['delta'], 4), ci95=[round(v, 4) for v in p['ci95']], per={k: round(a1[k] - a0[k], 4) for k in K}),
                             each_epoch=dict(delta=round(ps['delta'], 4), ci95=[round(v, 4) for v in ps['ci95']], per={k: round(a2[k] - a0[k], 4) for k in K}), best_on_rows=round(a0['mean'], 4))
    res['added'] = added
    print(json.dumps(res), flush=True)
    out = LANE / 'reads'; out.mkdir(exist_ok=True); (out / f'{name}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
