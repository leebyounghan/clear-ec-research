"""GREEN stacking on the recipe A seed-46 OOF: do point-map features correct the calibrated TTA predictions?

Base = recipe A seed46 four-flip TTA x leave-one-fold-out scalars (the adopted single-model pipeline).
Features per image, from the fold's own point maps (frozen detector, threshold .25; point specialist,
threshold .35), computed on the whole frame because no ROI exists at inference: log1p(#local maxima above
threshold, 5x5-cell window ~ NMS16), mean probability, fraction above threshold, log mean nearest-neighbour
distance and its coefficient of variation. Stacker: per-target ridge on the log residual with standardized
features, alpha chosen by inner leave-one-fold-out on the fitting folds. Outer leave-one-fold-out over the
five folds. Rows whose point labels were seen by the fold's head (train-green images inside that fold's
validation set) are excluded from fitting and scoring. Gate: >=0.03pp mean improvement overall and on >=2
folds, with no CD/CV/HEX regression >0.05pp. CPU only; never promotes anything.
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

E = Path('experiments')
ANALYSIS = E/'recipeA_analysis_20261028'
MAPS = E/'green_point_features_20261027/extract'
SOURCES = {'detector': ('recipeA_s46', .25), 'specialist': ('specialist_s46', .35)}
STRIDE, OFFSET = 8.0, 3.5
ALPHAS = (0.1, 1., 10., 100.)
TARGETS = ('CD', 'CV', 'HEX')


def official(pred, y):
    scored = y > 0
    with np.errstate(all='ignore'):
        v = np.nanmean(np.where(scored, np.abs(pred-y)/np.where(scored, y, 1.), np.nan), 0)*100
    return dict(CD=float(v[0]), CV=float(v[1]), HEX=float(v[2]), mean=float(v.mean()), n=int(len(y)))


def map_features(prob, threshold):
    p = prob.astype(np.float32)
    peak = (p >= threshold) & (p == maximum_filter(p, size=5, mode='nearest'))
    yy, xx = np.nonzero(peak)
    n = len(yy)
    if n >= 2:
        pts = np.stack([OFFSET+STRIDE*xx, OFFSET+STRIDE*yy], 1)
        d, _ = cKDTree(pts).query(pts, k=2)
        nn = d[:, 1]
        log_nn, cv_nn = float(np.log(nn.mean())), float(nn.std()/max(nn.mean(), 1e-6))
    else:
        log_nn, cv_nn = np.log(1312.), 0.
    return [np.log1p(n), float(p.mean()), float((p >= threshold).mean()), log_nn, cv_nn]


def ridge_fit(X, r, alpha):
    mu, sd = X.mean(0), X.std(0)+1e-9
    Z = (X-mu)/sd
    Zb = np.c_[Z, np.ones(len(Z))]
    reg = alpha*np.eye(Zb.shape[1])
    reg[-1, -1] = 0.
    w = np.linalg.solve(Zb.T@Zb+reg, Zb.T@r)
    return mu, sd, w


def ridge_predict(model, X):
    mu, sd, w = model
    return np.c_[(X-mu)/sd, np.ones(len(X))]@w


def main(allow_missing, sources, mode):
    gates = json.loads((ANALYSIS/'seed46_analysis_final.json').read_text())
    scal = {int(k): np.array(v) for k, v in gates['calibration']['tta4']['lofo_scalars_by_held_fold'].items()}
    with np.load(ANALYSIS/'seed46_oof_final.npz') as z:
        idx, y, tta, fold = z['indices'], z['target'], z['tta4'], z['fold']
    calibrated = tta*np.stack([scal[f] for f in fold])
    # residual_on_calibrated: stack on the adopted pipeline (intercept competes with the MAPE-optimal scalar).
    # features_vs_scalar: stack (intercept + features) on uncalibrated TTA and compare against the LOFO scalar
    # calibration, so the features are judged against the calibration they would replace.
    base = calibrated if mode == 'residual_on_calibrated' else tta
    reference = calibrated
    feats = np.full((len(idx), 5*len(sources)), np.nan)
    leaked = np.zeros(len(idx), bool)
    pos = {int(i): k for k, i in enumerate(idx)}
    folds_ok = []
    for f in range(5):
        parts, seen = [], None
        for s in sources:
            tag, th = SOURCES[s]
            p = MAPS/f'fold{f}_{tag}.npz'
            if not p.exists():
                parts = None
                break
            with np.load(p) as z:
                rows = np.array([pos[int(i)] for i in z['indices']])
                parts.append((rows, np.array([map_features(m, th) for m in z['probability']])))
                seen = z['point_labels_seen'] if seen is None else (seen | z['point_labels_seen'])
        if parts is None:
            if not allow_missing:
                raise FileNotFoundError(f'maps missing for fold {f}')
            continue
        for j, (rows, F) in enumerate(parts):
            feats[rows, 5*j:5*j+5] = F
        leaked[parts[0][0]] = seen
        folds_ok.append(f)
    use = np.isin(fold, folds_ok) & ~leaked
    X_all = np.c_[feats, np.log(base)]
    r_all = np.where(y > 0, np.log(np.where(y > 0, y, 1.))-np.log(base), np.nan)
    corrected = base.copy()
    per_fold, chosen = {}, {}
    for k in folds_ok:
        held = use & (fold == k)
        train = use & (fold != k)
        pred_held = base[held].copy()
        alphas_k = []
        for j in range(3):
            tr = train & np.isfinite(r_all[:, j])
            best = None
            for a in ALPHAS:
                err = []
                for inner in folds_ok:
                    if inner == k:
                        continue
                    fit = tr & (fold != inner)
                    val = tr & (fold == inner)
                    if fit.sum() < 50 or val.sum() < 10:
                        continue
                    m = ridge_fit(X_all[fit], r_all[fit, j], a)
                    err.append(np.abs(np.exp(ridge_predict(m, X_all[val]))*base[val, j]-y[val, j])/y[val, j])
                score = float(np.concatenate(err).mean()) if err else np.inf
                if best is None or score < best[0]:
                    best = (score, a)
            alphas_k.append(best[1])
            m = ridge_fit(X_all[tr], r_all[tr, j], best[1])
            pred_held[:, j] = base[held, j]*np.exp(ridge_predict(m, X_all[held]))
        corrected[held] = pred_held
        chosen[k] = alphas_k
        b, c = official(reference[held], y[held]), official(pred_held, y[held])
        per_fold[k] = dict(n_scored=int(held.sum()), n_leaked_excluded=int((leaked & (fold == k)).sum()), reference_calibrated=b, stacked=c,
                           delta_pp=c['mean']-b['mean'], alphas=alphas_k)
    b, c = official(reference[use], y[use]), official(corrected[use], y[use])
    delta = c['mean']-b['mean']
    improving = sum(v['delta_pp'] <= -0.03 for v in per_fold.values())
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), mode=mode, sources=list(sources), folds=folds_ok,
                  complete=len(folds_ok) == 5, rows_scored=int(use.sum()), rows_leaked_excluded=int(leaked.sum()),
                  base=b, stacked=c, delta_pp=delta, per_target_delta_pp={t: c[t]-b[t] for t in TARGETS},
                  component_regression_over_0p05=[t for t in TARGETS if c[t]-b[t] > 0.05], per_fold=per_fold,
                  gate=dict(rule='>=0.03pp overall and on >=2 folds, no component worse by >0.05pp',
                            passes=bool(delta <= -0.03 and improving >= 2 and not any(c[t]-b[t] > 0.05 for t in TARGETS)), folds_at_least_0p03=improving),
                  features=['log1p(peaks)', 'mean_prob', 'frac_above', 'log_mean_nn_dist', 'cv_nn_dist'],
                  limitations=['peaks via 5x5 local maxima, not the probe greedy NMS', 'point heads trained on 20 images', 'development folds reused'])
    tag = ('final' if report['complete'] else f'preview_{len(folds_ok)}folds')+'_'+mode+'_'+'_'.join(sources)
    (ANALYSIS/f'green_stacking_{tag}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(mode=mode, sources=list(sources), folds=folds_ok, reference=round(b['mean'], 4), stacked=round(c['mean'], 4), delta=round(delta, 4),
                          per_fold={k: round(v['delta_pp'], 4) for k, v in per_fold.items()}, gate=report['gate']['passes'],
                          excluded=report['rows_leaked_excluded'])), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--allow-missing', action='store_true')
    ap.add_argument('--sources', nargs='+', default=['detector', 'specialist'], choices=list(SOURCES))
    ap.add_argument('--mode', default='features_vs_scalar', choices=['features_vs_scalar', 'residual_on_calibrated'])
    a = ap.parse_args()
    main(a.allow_missing, a.sources, a.mode)
