"""Pre-registered comparison for style_specialist_20261102 (see protocol.json): framed-style specialist vs the
generalist recipe A member with the same fold / seed / SSL encoder, terminal-epoch single-view validation predictions.
Member level on framed / unframed / all rows; hybrid level with the specialist substituted on framed rows only
(0.5 * member x the fold's LOFO recipe-A scalars + 0.5 * Phase-1 five; generalist scalars, i.e. no re-fit), paired
donor-cluster bootstrap. `pooled` concatenates the development folds 0 and 1 and applies the continue/stop rule."""
import json, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E

X = '/workspace/clear-ec-research/experiments/'
OUT = X + 'style_specialist_20261102/'
REF = {0: 'phase2_pooling_replication_20261026/f0_s46_std', 1: 'oof_recipeA_20261027/fold1/regression_std_s46'}
b = E.load_base(); y = b['y']
SC = json.load(open(X + 'recipeA_analysis_20261028/hybrid_oof.json'))['recipeA_lofo_scalars']
BF = E.image_stats(b)['black_frac']
DON = E.donors(b['ids'])


def fold_arrays(f):
    rows = np.where(b['fold'] == f)[0]
    rp = {int(b['idx'][r]): n for n, r in enumerate(rows)}

    def load(root):
        with np.load(X + root + f'/runs/fold{f}/cpdino_vitb/predictions.npz') as z:
            p = np.full((len(rows), 3), np.nan)
            for i, v in zip(z['indices'].astype(int), z['pred'].astype(np.float64)):
                p[rp[int(i)]] = v
        assert np.isfinite(p).all(), root
        return p

    gen, spec = load(REF[f]), load(f'style_specialist_20261102/f{f}_s46_framed')
    frame = BF[rows] > 0.01
    sub = np.where(frame[:, None], spec, gen)
    sc = np.array(SC[str(f)]); P1 = b['P1'][rows]; yy = y[rows]
    return dict(rows=rows, frame=frame, yy=yy, g=DON[rows],
                eG=E.ape(gen, yy), eX=E.ape(spec, yy), hG=E.ape(0.5 * gen * sc + 0.5 * P1, yy), hX=E.ape(0.5 * sub * sc + 0.5 * P1, yy))


def report(d, tag):
    def pb(e1, e0, m):
        return E.paired_bootstrap(e1[m], e0[m], np.unique(d['g'][m], return_inverse=True)[1])
    fr, allm = d['frame'], np.ones(len(d['frame']), bool)
    mean = lambda e, m: round(E.official(e, m)['mean'], 4)
    return dict(tag=tag, n=int(len(fr)), n_frame=int(fr.sum()),
                member=dict(generalist={k: mean(d['eG'], m) for k, m in (('all', allm), ('frame', fr), ('no_frame', ~fr))},
                            specialist={k: mean(d['eX'], m) for k, m in (('all', allm), ('frame', fr), ('no_frame', ~fr))},
                            delta_frame=pb(d['eX'], d['eG'], fr), delta_no_frame=pb(d['eX'], d['eG'], ~fr)),
                hybrid_substituted=dict(delta_all=pb(d['hX'], d['hG'], allm), delta_frame=pb(d['hX'], d['hG'], fr)))


if __name__ == '__main__':
    per = {f: fold_arrays(f) for f in (0, 1)}
    res = {f'fold{f}': report(per[f], f'fold{f}') for f in (0, 1)}
    cat = {k: np.concatenate([per[f][k] for f in (0, 1)]) for k in per[0]}
    cat['g'] = np.unique(cat['g'], return_inverse=True)[1]
    res['pooled'] = report(cat, 'folds 0+1')
    dfr = res['pooled']['member']['delta_frame']['delta']
    both = all(res[f'fold{f}']['member']['delta_frame']['delta'] < 0 for f in (0, 1))
    res['continue_rule_pass'] = bool(dfr <= -0.15 and both)
    res['stop_rule_fires'] = bool(dfr > -0.05)
    json.dump(res, open(OUT + 'compare_dev.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))
