"""Recipe A seed-46 OOF analysis: TTA gate, scalar-calibration LOFO gate, per-fold table. CPU only.

Pre-registered rules (experiments/recipe_A_20261027/preregistration_20261027.json):
- TTA (four flips, mean in standardized-log space) is adopted only if its OOF delta is <= 0.
- Calibration = per-target scalar multipliers, MAPE-optimal weighted median of y/p with weights p/y
  (the Phase 1 estimator); benefit estimated leave-one-fold-out; adopted only if the LOFO delta <= 0;
  final multipliers fitted on all five folds are reported but not applied here.
Scoring: public zero rule (zero target unscored for that metric). --allow-missing previews with
the folds available; the gate decisions are only final with all five folds.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

E = Path('experiments')
ROOTS = {0: E/'phase2_pooling_replication_20261026/f0_s46_std', 1: E/'oof_recipeA_20261027/fold1/regression_std_s46',
         2: E/'oof_recipeA_20261027/fold2/regression_std_s46_v2', 3: E/'oof_recipeA_20261027/fold3/regression_std_s46',
         4: E/'phase2_pooling_20261020/arms/std'}
META = Path('datasets/clear_ec_official/cache_native_meta_512.npz')
OUT = E/'recipeA_analysis_20261028'
TARGETS = ('CD', 'CV', 'HEX')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def official(pred, y):
    scored = y > 0
    with np.errstate(all='ignore'):
        v = np.nanmean(np.where(scored, np.abs(pred-y)/np.where(scored, y, 1.), np.nan), 0)*100
    return dict(CD=float(v[0]), CV=float(v[1]), HEX=float(v[2]), mean=float(v.mean()), n=int(len(y)),
                zero_unscored=dict(zip(TARGETS, (~scored).sum(0).tolist())))


def weighted_median(values, weights):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, c[-1]/2.)])


def fit_scalars(pred, y):
    """Per-target c minimising sum |c p - y| / y over rows with y > 0."""
    out = []
    for j in range(3):
        m = y[:, j] > 0
        out.append(weighted_median(y[m, j]/pred[m, j], pred[m, j]/y[m, j]))
    return np.array(out)


def load(fold, root):
    run = root/'runs'/f'fold{fold}'/'cpdino_vitb'
    p = run/'predictions_tta4.npz'
    if not (root/'audit.json').exists() or not p.exists():
        return None
    with np.load(p) as z:
        d = dict(indices=z['indices'].copy(), ids=z['ids'].astype(str), target=z['target'].astype(np.float64),
                 single=z['pred_single'].astype(np.float64), tta=z['pred_tta4'].astype(np.float64))
    with np.load(run/'predictions.npz') as z:
        np.testing.assert_array_equal(z['indices'], d['indices'])
        stored = z['pred'].astype(np.float64)
    d['stored_vs_recomputed_max_rel'] = float((np.abs(stored-d['single'])/stored).max())
    d['sha'] = sha(p)
    return d


def main(allow_missing):
    with np.load(META, allow_pickle=True) as z:
        ids_all, y_all = z['ids'].astype(str), z['Y'].astype(np.float64)
    folds = {}
    for f, root in ROOTS.items():
        d = load(f, root)
        if d is None:
            if not allow_missing:
                raise FileNotFoundError(f'fold {f} missing: {root}')
            continue
        np.testing.assert_array_equal(ids_all[d['indices']], d['ids'])
        np.testing.assert_allclose(y_all[d['indices']], d['target'], rtol=1e-6)
        folds[f] = d
    present = sorted(folds)
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), folds_present=present,
                  complete=len(present) == 5, roots={f: str(ROOTS[f]) for f in present},
                  prediction_sha={f: folds[f]['sha'] for f in present}, per_fold={}, oof={}, tta_gate={}, calibration={})
    for f in present:
        d = folds[f]
        s, t = official(d['single'], d['target']), official(d['tta'], d['target'])
        report['per_fold'][f] = dict(single=s, tta4=t, tta_delta_pp=t['mean']-s['mean'],
                                     stored_vs_recomputed_max_rel=d['stored_vs_recomputed_max_rel'])
    idx = np.concatenate([folds[f]['indices'] for f in present])
    y = np.concatenate([folds[f]['target'] for f in present])
    single = np.concatenate([folds[f]['single'] for f in present])
    tta = np.concatenate([folds[f]['tta'] for f in present])
    fold_of = np.concatenate([np.full(len(folds[f]['indices']), f) for f in present])
    assert len(np.unique(idx)) == len(idx)
    report['oof']['single'] = official(single, y)
    report['oof']['tta4'] = official(tta, y)
    tta_delta = report['oof']['tta4']['mean']-report['oof']['single']['mean']
    report['tta_gate'] = dict(delta_pp=tta_delta, adopt=bool(tta_delta <= 0), rule='adopt if OOF delta <= 0',
                              per_target_delta_pp={k: report['oof']['tta4'][k]-report['oof']['single'][k] for k in TARGETS},
                              folds_improving=int(sum(report['per_fold'][f]['tta_delta_pp'] < 0 for f in present)), final=len(present) == 5)
    for name, base in (('single', single), ('tta4', tta)):
        lofo = np.empty_like(base)
        scalars = {}
        for f in present:
            held = fold_of == f
            c = fit_scalars(base[~held], y[~held])
            scalars[f] = c.tolist()
            lofo[held] = base[held]*c
        uncal, cal = official(base, y), official(lofo, y)
        full = fit_scalars(base, y)
        report['calibration'][name] = dict(uncalibrated=uncal, lofo_calibrated=cal, lofo_delta_pp=cal['mean']-uncal['mean'],
                                           per_target_delta_pp={k: cal[k]-uncal[k] for k in TARGETS},
                                           lofo_scalars_by_held_fold=scalars, full_fit_scalars=dict(zip(TARGETS, full.tolist())),
                                           adopt=bool(cal['mean']-uncal['mean'] <= 0), rule='adopt if LOFO delta <= 0',
                                           phase1_reference_scalars=dict(CD=0.9970446110788349, CV=0.987972130802936, HEX=0.9892234872488541))
    base_name = 'tta4' if report['tta_gate']['adopt'] else 'single'
    cal = report['calibration'][base_name]
    report['candidate_single_model_pipeline'] = dict(base=base_name, calibration='scalar' if cal['adopt'] else 'none',
                                                     oof_mean=(cal['lofo_calibrated'] if cal['adopt'] else cal['uncalibrated'])['mean'],
                                                     note='seed46 single model; ensemble and matched historical comparison are separate')
    report['limitations'] = ['Development folds reused; recipe choices made inside these folds; optimistic',
                             'Patient independence and near-duplicate leakage unresolved',
                             'Calibration LOFO uses four folds to fit and one to score; final scalars are fitted on all five']
    OUT.mkdir(exist_ok=True)
    tag = 'final' if report['complete'] else f'preview_{len(present)}folds'
    path = OUT/f'seed46_analysis_{tag}.json'
    path.write_text(json.dumps(report, indent=2))
    np.savez_compressed(OUT/f'seed46_oof_{tag}.npz', indices=idx, target=y, single=single, tta4=tta, fold=fold_of)
    print(json.dumps(dict(folds=present, oof_single=round(report['oof']['single']['mean'], 4), oof_tta4=round(report['oof']['tta4']['mean'], 4),
                          tta_delta=round(tta_delta, 4), calib_lofo_delta_single=round(report['calibration']['single']['lofo_delta_pp'], 4),
                          calib_lofo_delta_tta4=round(report['calibration']['tta4']['lofo_delta_pp'], 4),
                          full_scalars_tta4=report['calibration']['tta4']['full_fit_scalars'], out=str(path))), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--allow-missing', action='store_true')
    main(ap.parse_args().allow_missing)
