"""Leave-one-source-out shift read (backbone lane, 2026-11-10). Source defines the split only.
Run A trains on P1 ('NNNN-YY', old) minus its hold-out; run B on P2 ('NNNN-NNNN aaaa', new) minus its hold-out.
Member error per row = mean of the per-epoch APE over epochs LO..HI (raw track unless --track), zero labels unscored per metric.
Shift penalty for source X = error on X's hold-out rows from the other-source model - error on the same rows from X's own model.
Also: cross-source error on all rows of X, and prediction bias = mean log(pred / label) on those rows.
Usage: read_shift_20261110.py <root A> <root B> [--window 8,12] [--track raw] [--name tag]"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
X = Path('/workspace/clear-ec-research/experiments'); D = X / 'lane_backbone_20261108'
K = ['CD', 'CV', 'HEX']


def load(root, fold, lo, hi, track):
    run = sorted((X / root / track / 'runs' / f'fold{fold}').glob('*'))[0]; P = []
    for e in range(lo, hi + 1):
        with np.load(run / f'predictions_epoch_{e:02d}.npz', allow_pickle=True) as z:
            P.append((z['indices'].astype(int), z['pred'].astype(np.float64), z['target'].astype(np.float64)))
    ix = P[0][0]; assert all(np.array_equal(p[0], ix) for p in P)
    return ix, np.stack([p[1] for p in P]), P[0][2]


def stats(ix, preds, y, rows):
    m = np.isin(ix, rows); yy = y[m]; pp = preds[:, m]
    err = np.nanmean([E.ape(p, yy) for p in pp], 0)  # mean over epochs of per-row APE
    s = E.official(err)
    ok = yy > 0; lr = np.where(ok, np.log(np.where(ok, pp.mean(0), 1) / np.where(ok, yy, 1)), np.nan)  # bias of the epoch-mean prediction
    return dict(n=int(m.sum()), **{k: round(s[k], 3) for k in K + ['mean']}, bias_mean_log_ratio={k: round(float(np.nanmean(lr[:, j])), 4) for j, k in enumerate(K)})


def main():
    rA, rB = sys.argv[1], sys.argv[2]
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    lo, hi = [int(v) for v in arg('--window', '8,12').split(',')]; track = arg('--track', 'raw'); name = arg('--name', 'shift')
    sp = np.load(D / 'data/source_split_20261110.npz', allow_pickle=True); P1, P2, hold = sp['P1'], sp['P2'], sp['holdout']
    h1, h2 = np.intersect1d(P1, hold), np.intersect1d(P2, hold)
    ixA, pA, yA = load(rA, 201, lo, hi, track); ixB, pB, yB = load(rB, 202, lo, hi, track)
    res = dict(name=name, window=[lo, hi], track=track,
               P1_in_source=stats(ixA, pA, yA, h1), P1_cross_on_holdout=stats(ixB, pB, yB, h1), P1_cross_all=stats(ixB, pB, yB, P1),
               P2_in_source=stats(ixB, pB, yB, h2), P2_cross_on_holdout=stats(ixA, pA, yA, h2), P2_cross_all=stats(ixA, pA, yA, P2))
    res['penalty_P1'] = {k: round(res['P1_cross_on_holdout'][k] - res['P1_in_source'][k], 3) for k in K + ['mean']}
    res['penalty_P2'] = {k: round(res['P2_cross_on_holdout'][k] - res['P2_in_source'][k], 3) for k in K + ['mean']}
    print(json.dumps(res, indent=1)); (D / 'reads' / f'{name}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
