"""E3 comparison (CPU): box-restricted std pooling vs the standard recipe A member (and the masked-pool probe where it
exists) on the held fold: raw TTA4 and single-view MAPE per target, framed / unframed strata, paired donor bootstrap.
    python ... --fold 0
"""
import argparse, json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, 'experiments/eval_harness_20261031/source')
import eval_oof_20261031 as H
E = Path('experiments')
STD = {0: 'phase2_pooling_replication_20261026/f0_s46_std', 1: 'oof_recipeA_20261027/fold1/regression_std_s46', 2: 'oof_recipeA_20261027/fold2/regression_std_s46_v2', 3: 'oof_recipeA_20261027/fold3/regression_std_s46', 4: 'phase2_pooling_20261020/arms/std'}


def load(root, f):
    run = E/root/'runs'/f'fold{f}'/'cpdino_vitb'; p = run/'predictions_tta4.npz'
    if p.exists():
        with np.load(p) as z:
            return dict(idx=z['indices'].astype(int), y=z['target'].astype(float), tta=z['pred_tta4'].astype(float), single=z['pred_single'].astype(float), has_tta=True)
    q = run/'predictions.npz'                      # audit output: single-view predictions of the terminal epoch
    if q.exists():
        with np.load(q) as z:
            return dict(idx=z['indices'].astype(int), y=z['target'].astype(float), tta=z['pred'].astype(float), single=z['pred'].astype(float), has_tta=False)
    return None


def main(a):
    f = a.fold
    runs = {'standard': load(STD[f], f), 'box': load(f'green_day3_20261103/e3_boxpool/f{f}_s46_stdbox', f), 'masked': load(f'masked_pool_regression_20261031/f{f}_s46_stdmasked', f)}
    runs = {k: v for k, v in runs.items() if v is not None}
    base = H.load_base(); pos = {int(i): n for n, i in enumerate(base['idx'])}
    st = H.image_stats(base); g = H.donors(base['ids'])
    ref = runs['standard']; order = {int(i): n for n, i in enumerate(ref['idx'])}
    res = dict(fold=f, n=int(len(ref['idx'])), runs={}, deltas={})
    rows = np.array([pos[int(i)] for i in ref['idx']]); framed = st['black_frac'][rows] > 0.01; groups = g[rows]
    e = {}
    if not all(v['has_tta'] for v in runs.values()):   # fair comparison: single view everywhere until the box run has TTA
        for v in runs.values(): v['tta'] = v['single']
        print('NOTE: single-view comparison (box run has no TTA predictions yet)', flush=True)
    for k, v in runs.items():
        sel = np.array([order[int(i)] for i in v['idx']]); assert np.array_equal(np.sort(v['idx']), np.sort(ref['idx']))
        tta = np.zeros_like(v['tta']); tta[sel] = v['tta']; single = np.zeros_like(v['single']); single[sel] = v['single']
        e[k] = H.ape(tta, ref['y']); es = H.ape(single, ref['y'])
        res['runs'][k] = dict(tta4=H.official(e[k]), single=H.official(es), framed=H.official(e[k][framed]), unframed=H.official(e[k][~framed]), has_tta=bool(v['has_tta']))
        print('%-9s tta4 %s | single mean %.4f | framed %.4f unframed %.4f' % (k, {kk: round(vv, 4) for kk, vv in res['runs'][k]['tta4'].items() if kk != 'n'}, res['runs'][k]['single']['mean'], res['runs'][k]['framed']['mean'], res['runs'][k]['unframed']['mean']), flush=True)
    for k in runs:
        if k == 'standard':
            continue
        res['deltas'][k] = dict(all=H.paired_bootstrap(e[k], e['standard'], groups), framed=H.paired_bootstrap(e[k][framed], e['standard'][framed], groups[framed]), unframed=H.paired_bootstrap(e[k][~framed], e['standard'][~framed], groups[~framed]),
                                per_target={t: float(np.nanmean(e[k][:, j])-np.nanmean(e['standard'][:, j])) for j, t in enumerate(H.K)})
        print('delta %-7s vs standard: all %.4f %s | framed %.4f | unframed %.4f | per target %s' % (k, res['deltas'][k]['all']['delta'], res['deltas'][k]['all']['ci95'], res['deltas'][k]['framed']['delta'], res['deltas'][k]['unframed']['delta'], {t: round(v, 4) for t, v in res['deltas'][k]['per_target'].items()}), flush=True)
    out = E/'green_day3_20261103/e3_boxpool'/f'compare_fold{f}.json'; out.write_text(json.dumps(res, indent=1, default=float)); print('written', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--fold', type=int, required=True); main(ap.parse_args())
