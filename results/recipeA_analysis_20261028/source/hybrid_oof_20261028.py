"""Candidate compositions A/B/C on the five-fold OOF: recipe A (TTA + LOFO scalars) vs hybrid with the Phase 1 members.

Historical members (per-image out-of-fold, decoded, Phase 1 runs): seed42, seed77, seed2026, metric35,
predbag2_35 for every fold; the SSL/raw sixth exists only for folds 2-4, so the historical ensemble here is
the equal mean of the five members on all folds, times the fixed Phase 1 scalars (reported separately
with the six where available). Recipe A members: seeds with predictions_tta4.npz on every fold (seed 46 now;
47/48 added automatically when complete), equal mean, LOFO scalars. Hybrid = w*recipeA + (1-w)*historical
with w fixed 0.5 (B) or chosen leave-one-fold-out from {0.25,0.33,0.5,0.67} (C). Paired cluster bootstrap for
B and C against A. CPU only; reports everything, promotes nothing.
"""
import datetime
import json
from pathlib import Path

import numpy as np

E = Path('experiments')
ANALYSIS = E/'recipeA_analysis_20261028'
SCALE = np.array([0.9970446110788349, 0.987972130802936, 0.9892234872488541])
HIST = {'seed42': 'followup_20260902/runs/fixed_base_f{f}_seed42_c256k1_mean_multitask_slidesmooth/best_oof.npz',
        'seed77': 'submission_20260904/runs/base_seed77_f{f}_seed77_c256k1_mean_multitask_slidesmooth/best_oof.npz',
        'seed2026': 'submission_20260904/runs/base_seed2026_f{f}_seed2026_c256k1_mean_multitask_slidesmooth/best_oof.npz',
        'metric35': 'methods_20260904/runs/metric35_f{f}_seed42_c256k1_mean_multitask_slidesmooth/best_oof.npz',
        'predbag2_35': 'methods_20260904/runs/predbag2_35_f{f}_seed42_c256k2_mean_multitask_slidesmooth/best_oof.npz'}
SIXTH = 'raw_ssl_next_20260913/ssl_runs/fold{f}/standard/oof.npz'
RECIPE = {46: {0: 'phase2_pooling_replication_20261026/f0_s46_std', 1: 'oof_recipeA_20261027/fold1/regression_std_s46',
               2: 'oof_recipeA_20261027/fold2/regression_std_s46_v2', 3: 'oof_recipeA_20261027/fold3/regression_std_s46',
               4: 'phase2_pooling_20261020/arms/std'},
          47: {0: 'phase2_pooling_replication_20261026/f0_s47_std', 4: 'phase2_pooling_replication_20261026/f4_s47_std',
               **{f: f'oof_recipeA_20261027/fold{f}/regression_std_s47' for f in (1, 2, 3)}},
          48: {f: f'oof_recipeA_20261027/fold{f}/regression_std_s48' for f in range(5)}}
WS = (0.25, 0.33, 0.5, 0.67)
B, SEED = 10000, 20261028
T = ('CD', 'CV', 'HEX')


def official(p, y):
    s = y > 0
    with np.errstate(all='ignore'):
        v = np.nanmean(np.where(s, np.abs(p-y)/np.where(s, y, 1.), np.nan), 0)*100
    return dict(CD=float(v[0]), CV=float(v[1]), HEX=float(v[2]), mean=float(v.mean()), n=int(len(y)))


def weighted_median(v, w):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, c[-1]/2.)])


def fit_scalars(p, y):
    return np.array([weighted_median(y[y[:, j] > 0, j]/p[y[:, j] > 0, j], p[y[:, j] > 0, j]/y[y[:, j] > 0, j]) for j in range(3)])


def bootstrap(diff, groups):
    u, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, weights=diff, minlength=len(u))
    c = np.bincount(inv, minlength=len(u)).astype(float)
    d = np.random.default_rng(SEED).integers(0, len(u), size=(B, len(u)))
    return s[d].sum(1)/c[d].sum(1)


def main():
    with np.load(ANALYSIS/'seed46_oof_final.npz') as z:
        idx, y, fold = z['indices'], z['target'], z['fold']
    pos = {int(i): k for k, i in enumerate(idx)}
    with np.load('datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True) as z:
        ids = z['ids'].astype(str)
    # recipe A members available on all five folds
    members = {}
    for seed, roots in RECIPE.items():
        arr = np.full((len(idx), 3), np.nan)
        ok = True
        for f, root in roots.items():
            p = E/root/'runs'/f'fold{f}'/'cpdino_vitb'/'predictions_tta4.npz'
            if not p.exists() or not (E/root/'audit.json').exists():
                ok = False
                break
            with np.load(p) as z:
                rows = np.array([pos[int(i)] for i in z['indices']])
                arr[rows] = z['pred_tta4']
        if ok and np.isfinite(arr).all():
            members[seed] = arr
    recipe_raw = np.mean(list(members.values()), 0)
    # historical five (and six where available), decoded, Phase 1 scalars
    hist5 = np.full((len(idx), 3), np.nan)
    hist6 = np.full((len(idx), 3), np.nan)
    six_folds = []
    for f in range(5):
        preds = []
        for name, pat in HIST.items():
            with np.load(E/pat.format(f=f), allow_pickle=True) as z:
                rows = np.array([pos[int(i)] for i in z['indices']])
                assert (ids[z['indices']] == z['ids'].astype(str)).all()
                preds.append((rows, z['pred'].astype(np.float64)))
        rows = preds[0][0]
        assert all(np.array_equal(r, rows) for r, _ in preds)
        five = np.mean([p for _, p in preds], 0)
        hist5[rows] = five
        p6 = E/SIXTH.format(f=f)
        if p6.exists():
            with np.load(p6) as z:
                r6 = np.array([pos[int(i)] for i in z['indices']])
                assert np.array_equal(np.sort(r6), np.sort(rows))
                six = five.copy()
                six[np.argsort(rows)] = (5*five[np.argsort(rows)]+z['pred'].astype(np.float64)[np.argsort(r6)])/6
                hist6[rows] = six
                six_folds.append(f)
    assert np.isfinite(hist5).all()
    hist5c, hist6c = hist5*SCALE, hist6*SCALE
    # LOFO scalars for recipe A and LOFO hybrid weight
    A = np.empty_like(recipe_raw)
    scalars = {}
    for f in range(5):
        held = fold == f
        c = fit_scalars(recipe_raw[~held], y[~held])
        scalars[f] = c.tolist()
        A[held] = recipe_raw[held]*c
    Bh = 0.5*A+0.5*hist5c
    C = np.empty_like(A)
    wsel = {}
    for f in range(5):
        held = fold == f
        best = min(WS, key=lambda w: official(w*A[~held]+(1-w)*hist5c[~held], y[~held])['mean'])
        wsel[f] = best
        C[held] = best*A[held]+(1-best)*hist5c[held]
    groups = np.array([i.rsplit('-', 1)[0] for i in ids[idx]])

    def compare(name, p):
        s = official(p, y)
        d = ((np.abs(p-y)/np.where(y > 0, y, 1.)*(y > 0)).mean(1)-(np.abs(A-y)/np.where(y > 0, y, 1.)*(y > 0)).mean(1))*100
        bt = bootstrap(d, groups)
        return dict(score=s, delta_vs_A_pp=s['mean']-official(A, y)['mean'], ci95=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                    p_better_than_A=float((bt < 0).mean()), per_fold={f: official(p[fold == f], y[fold == f])['mean'] for f in range(5)})
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), recipeA_members=sorted(members), rows=int(len(idx)),
                  recipeA_lofo_scalars=scalars, A_recipeA_tta_scalar=dict(score=official(A, y), per_fold={f: official(A[fold == f], y[fold == f])['mean'] for f in range(5)}),
                  historical_five_calibrated=dict(score=official(hist5c, y), per_fold={f: official(hist5c[fold == f], y[fold == f])['mean'] for f in range(5)}),
                  historical_six_calibrated_where_available=dict(folds=six_folds, score=official(hist6c[np.isin(fold, six_folds)], y[np.isin(fold, six_folds)]) if six_folds else None),
                  B_hybrid_w0p5=compare('B', Bh), C_hybrid_lofo_w=dict(**compare('C', C), w_by_held_fold=wsel),
                  w_grid_full_oof={w: official(w*A+(1-w)*hist5c, y)['mean'] for w in WS},
                  error_correlation_A_vs_hist5={t: float(np.corrcoef(np.abs(A[:, j]-y[:, j])/np.where(y[:, j] > 0, y[:, j], 1.), np.abs(hist5c[:, j]-y[:, j])/np.where(y[:, j] > 0, y[:, j], 1.))[0, 1]) for j, t in enumerate(T)},
                  notes=['historical OOF rows are out-of-fold per image under the Phase 1 folds; combining per image with recipe A OOF keeps the hybrid out-of-fold',
                         'historical members used validation-selected checkpoints (optimistic)', 'development folds reused; patient independence unresolved'])
    (ANALYSIS/'hybrid_oof.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(ANALYSIS/'hybrid_oof.npz', indices=idx, ids=ids[idx], target=y, fold=fold, A_recipeA_tta_scalar=A,
                        historical_five_calibrated=hist5c, B_hybrid_w0p5=Bh, C_hybrid_lofo_w=C,
                        recipeA_members=np.array(sorted(members)))
    print(json.dumps(dict(members=sorted(members), A=round(report['A_recipeA_tta_scalar']['score']['mean'], 4), hist5=round(report['historical_five_calibrated']['score']['mean'], 4),
                          B=round(report['B_hybrid_w0p5']['score']['mean'], 4), B_ci=report['B_hybrid_w0p5']['ci95'], C=round(report['C_hybrid_lofo_w']['score']['mean'], 4), w=wsel,
                          grid={str(k): round(v, 4) for k, v in report['w_grid_full_oof'].items()}, corr=report['error_correlation_A_vs_hist5'])), flush=True)


if __name__ == '__main__':
    main()
