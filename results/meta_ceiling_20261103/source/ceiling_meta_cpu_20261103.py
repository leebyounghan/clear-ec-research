"""Why the OOF error stopped falling: CPU meta-analysis of the existing out-of-fold predictions (2026-11-03).

Inputs are the shipped hybrid B OOF (0.5 recipe A + 0.5 Phase-1 five), the recipe A seed members, the Phase-1 members,
the green detector features and label-free image statistics. No new model is trained except small gradient-boosting
probes on the residual. Sections (all on the official zero rule, donor-grouped folds):
  variance   : seed-to-seed variance of recipe A and the 1/S extrapolation of member averaging
  shared     : how much of the log error is common to recipe A and Phase-1; per-image oracle selection ceilings
  dilution   : explained variance, calibration slope of log y on log B, LOFO affine-in-log recalibration
  learnable  : can any available signal predict B's residual? LOFO gradient boosting, three feature sets
  concentration : Lorenz share of the error, error and bias by label decile
  structure  : cross-metric and fellow-eye residual correlation, group (year x style) bias share
Workers follow the pod rule: clamp(16 - load1 - 1, 2, 12) forked processes, one thread each.
"""
import json, os, re, sys
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E

X = Path('/workspace/clear-ec-research/experiments')
OUT = X / 'meta_ceiling_20261103/cpu'
K = ('CD', 'CV', 'HEX')
SCALE = np.array([0.9970446110788349, 0.987972130802936, 0.9892234872488541])
HIST = {'seed42': 'followup_20260902/runs/fixed_base_f{f}_seed42_c256k1_mean_multitask_slidesmooth/best_oof.npz',
        'seed77': 'submission_20260904/runs/base_seed77_f{f}_seed77_c256k1_mean_multitask_slidesmooth/best_oof.npz',
        'seed2026': 'submission_20260904/runs/base_seed2026_f{f}_seed2026_c256k1_mean_multitask_slidesmooth/best_oof.npz',
        'metric35': 'methods_20260904/runs/metric35_f{f}_seed42_c256k1_mean_multitask_slidesmooth/best_oof.npz',
        'predbag2_35': 'methods_20260904/runs/predbag2_35_f{f}_seed42_c256k2_mean_multitask_slidesmooth/best_oof.npz'}
RECIPE = {46: {0: 'phase2_pooling_replication_20261026/f0_s46_std', 1: 'oof_recipeA_20261027/fold1/regression_std_s46',
               2: 'oof_recipeA_20261027/fold2/regression_std_s46_v2', 3: 'oof_recipeA_20261027/fold3/regression_std_s46',
               4: 'phase2_pooling_20261020/arms/std'},
          47: {0: 'phase2_pooling_replication_20261026/f0_s47_std', 4: 'phase2_pooling_replication_20261026/f4_s47_std',
               **{f: f'oof_recipeA_20261027/fold{f}/regression_std_s47' for f in (1, 2, 3)}},
          48: {f: f'oof_recipeA_20261027/fold{f}/regression_std_s48' for f in range(5)}}


def r4(v):
    return float(np.round(v, 4))


def mape(p, y):
    return E.official(E.ape(p, y))


def load_members(b):
    pos = {int(i): k for k, i in enumerate(b['idx'])}
    seeds, single = {}, {}
    for s, roots in RECIPE.items():
        a, a1 = np.full((len(pos), 3), np.nan), np.full((len(pos), 3), np.nan)
        for f, root in roots.items():
            p = X / root / 'runs' / f'fold{f}' / 'cpdino_vitb' / 'predictions_tta4.npz'
            if not p.exists():
                break
            with np.load(p) as z:
                r = np.array([pos[int(i)] for i in z['indices']])
                a[r], a1[r] = z['pred_tta4'], z['pred_single']
        if np.isfinite(a).all():
            seeds[s], single[s] = a, a1
    hist = {}
    for name, pat in HIST.items():
        a = np.full((len(pos), 3), np.nan)
        for f in range(5):
            with np.load(X / pat.format(f=f), allow_pickle=True) as z:
                a[np.array([pos[int(i)] for i in z['indices']])] = z['pred'].astype(np.float64) * SCALE
        hist[name] = a
    return seeds, single, hist


def variance_section(b, seeds, single):
    y = b['y']; out = dict(seeds=sorted(seeds))
    sc = np.array([json.load(open(X / 'recipeA_analysis_20261028/hybrid_oof.json'))['recipeA_lofo_scalars'][str(f)] for f in b['fold']])
    for name, M in (('tta4', seeds), ('single', single)):
        S = sorted(M)
        L = np.stack([np.log(M[s] * sc) for s in S])                              # S x N x 3
        ly = np.log(np.where(y > 0, y, np.nan))
        sig2 = np.nanmean(L.var(0, ddof=1), 0) if len(S) > 1 else np.full(3, np.nan)
        tot = np.nanmean((L.mean(0) - ly) ** 2, 0)
        curve = {}
        for k in range(1, len(S) + 1):
            from itertools import combinations
            vals = [mape(np.exp(L[list(c)].mean(0)), y)['mean'] for c in combinations(range(len(S)), k)]
            curve[k] = float(np.mean(vals))
        ks = np.array(sorted(curve)); ms = np.array([curve[k] for k in ks])
        fit = np.polyfit(1 / ks, ms, 1) if len(ks) > 1 else [np.nan, np.nan]
        out[name] = dict(seed_var_log=dict(zip(K, map(r4, sig2))), mse_log_of_mean=dict(zip(K, map(r4, tot))),
                         seed_var_share_of_single_mse={k: r4(sig2[j] / (tot[j] + sig2[j] * (1 - 1 / len(S)))) for j, k in enumerate(K)},
                         mape_by_members=curve, extrapolated_infinite_members=r4(fit[1]),
                         gain_from_more_seeds_vs_current=r4(fit[1] - curve[len(S)]))
    return out


def shared_section(b, seeds, hist, gf):
    y = b['y']; ly = np.log(np.where(y > 0, y, np.nan))
    rA, rP = np.log(b['A']) - ly, np.log(b['P1']) - ly
    out = {}
    for j, k in enumerate(K):
        ok = np.isfinite(ly[:, j])
        c = np.cov(rA[ok, j], rP[ok, j])
        out[k] = dict(var_A=r4(c[0, 0] * 1e4), var_P1=r4(c[1, 1] * 1e4), cov=r4(c[0, 1] * 1e4), corr=r4(c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])),
                      shared_fraction_of_B_var=r4(c[0, 1] / np.var(0.5 * rA[ok, j] + 0.5 * rP[ok, j])),
                      unit='1e-4 log^2')
    # per-image oracle selection among candidate sets: the most any router/gate over these models could reach
    cands = dict(A=b['A'], P1=b['P1'], B=b['B'])
    sets = {'A|P1': ['A', 'P1'], 'A|P1|B': ['A', 'P1', 'B']}
    for s in seeds:
        cands[f's{s}'] = seeds[s]
    for n in hist:
        cands[n] = hist[n]
    sets['all_members'] = [n for n in cands]
    e = {n: E.ape(p, y) for n, p in cands.items()}
    out['oracle_select'] = {}
    for name, members in sets.items():
        st = np.stack([e[m] for m in members])
        out['oracle_select'][name] = dict(n_models=len(members), mape={kk: r4(v) for kk, v in mape_from_e(np.nanmin(st, 0)).items()})
    # oracle best convex weight per image between A and P1 is the same as selecting the closer one when y lies outside
    # [A,P1] and 0 error otherwise: report the share of rows where y lies between the two
    btw = ((b['A'] - y) * (b['P1'] - y) < 0) & (y > 0)
    out['y_between_A_and_P1_share'] = {k: r4(btw[:, j].mean()) for j, k in enumerate(K)}
    out['B_mape'] = {kk: r4(v) for kk, v in mape_from_e(e['B']).items()}
    return out


def mape_from_e(e):
    m = np.nanmean(e, 0)
    return dict(zip(K, m), mean=m.mean())


def dilution_section(b):
    y, B, fold = b['y'], b['B'], b['fold']
    out = {}
    for j, k in enumerate(K):
        ok = y[:, j] > 0
        ly, lb = np.log(y[ok, j]), np.log(B[ok, j])
        slope, icpt = np.polyfit(lb, ly, 1)
        r2 = 1 - np.var(ly - lb) / np.var(ly)
        out[k] = dict(var_log_y=r4(np.var(ly) * 1e4), mse_log=r4(np.mean((ly - lb) ** 2) * 1e4), r2_vs_identity=r4(r2),
                      corr=r4(np.corrcoef(lb, ly)[0, 1]), slope_logy_on_logB=r4(slope), sd_ratio_pred_over_label=r4(lb.std() / ly.std()))
    # LOFO affine-in-log recalibration fitted on MAPE: p' = B^b * c, b on a grid, c by the MAPE-optimal weighted median
    grid = np.round(np.arange(0.85, 1.36, 0.025), 3)
    newp = np.full_like(B, np.nan); chosen = {}
    for f in range(5):
        tr, te = fold != f, fold == f
        for j in range(3):
            ok = tr & (y[:, j] > 0)
            best = None
            for g in grid:
                q = np.exp(g * (np.log(B[ok, j]) - np.log(B[ok, j]).mean()) + np.log(B[ok, j]).mean())
                c = wmed(y[ok, j] / q, q / y[ok, j])
                m = np.mean(np.abs(c * q - y[ok, j]) / y[ok, j])
                if best is None or m < best[0]:
                    best = (m, g, c, np.log(B[ok, j]).mean())
            _, g, c, mu = best
            chosen[(f, K[j])] = float(g)
            newp[te, j] = c * np.exp(g * (np.log(B[te, j]) - mu) + mu)
    out['lofo_affine_log'] = dict(slopes={f'{f}_{k}': v for (f, k), v in chosen.items()}, B=mape(B, y), recal=mape(newp, y),
                                  paired=E.paired_bootstrap(E.ape(newp, y), E.ape(B, y), E.donors(b['ids'])))
    return out


def wmed(v, w):
    o = np.argsort(v); c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, c[-1] / 2.)])


def features(b, gf, st):
    year = np.array([E.year_of(i) for i in b['ids']], float)
    eye = np.array([1. if re.search(r'O\s*D|0DCN|ODCN', i) else (0. if re.search(r'OSCN|O\s*S', i) else np.nan) for i in b['ids']])
    newfmt = np.array([1. if re.match(r'^\d{4}-\d{4}', i) else 0. for i in b['ids']])
    la, lp, lb = np.log(b['A']), np.log(b['P1']), np.log(b['B'])
    preds = np.c_[la, lp, la - lp]
    meta = np.c_[st['mean'], st['contrast'], st['black_frac'], st['focus'], year, eye, newfmt]
    def g(name):
        v = gf[name].astype(float).copy()
        if name + '_valid' in gf.files:
            v[~gf[name + '_valid']] = np.nan
        v[gf['seen_leaked']] = np.nan
        return v
    green = np.c_[g('n_det'), g('f_active10'), g('e2_sg2_s46_hex_mean_p'), g('hex_B_hex6_lofo')]
    return dict(preds=preds, preds_meta=np.c_[preds, meta], preds_meta_green=np.c_[preds, meta, green])


def _fit(task):
    from sklearn.ensemble import HistGradientBoostingRegressor
    Xf, t, fold, f, j = task
    tr = (fold != f) & np.isfinite(t[:, j])
    m = HistGradientBoostingRegressor(loss='absolute_error', max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                      min_samples_leaf=60, l2_regularization=1.0, random_state=0)
    m.fit(Xf[tr], t[tr, j])
    te = fold == f
    return f, j, np.where(te)[0], m.predict(Xf[te])


def learnable_section(b, feats, workers):
    y, B, fold = b['y'], b['B'], b['fold']
    t = np.log(np.where(y > 0, y, np.nan)) - np.log(B)
    g = E.donors(b['ids']); out = {}
    ctx = get_context('fork')
    for name, Xf in feats.items():
        tasks = [(Xf, t, fold, f, j) for f in range(5) for j in range(3)]
        that = np.full_like(B, np.nan)
        with ctx.Pool(workers) as pool:
            for f, j, rows, pr in pool.imap_unordered(_fit, tasks, chunksize=1):
                that[rows, j] = pr
        res = {}
        for shrink in (0.5, 1.0):
            p = B * np.exp(shrink * that)
            res[f'shrink{shrink}'] = dict(score=mape(p, y), paired=E.paired_bootstrap(E.ape(p, y), E.ape(B, y), g),
                                          per_fold={f: r4(mape(p[fold == f], y[fold == f])['mean'] - mape(B[fold == f], y[fold == f])['mean']) for f in range(5)})
        ok = np.isfinite(t)
        res['residual_r2'] = {k: r4(1 - np.nanmean((t[:, j] - that[:, j]) ** 2) / np.nanmean((t[:, j] - np.nanmean(t[:, j])) ** 2)) for j, k in enumerate(K)}
        res['residual_corr'] = {k: r4(np.corrcoef(t[ok[:, j], j], that[ok[:, j], j])[0, 1]) for j, k in enumerate(K)}
        res['n_features'] = int(Xf.shape[1])
        out[name] = res
    return out


def concentration_section(b):
    y, B = b['y'], b['B']; e = E.ape(B, y); out = {}
    rng = np.random.default_rng(0)
    for j, k in enumerate(K):
        v = np.sort(e[np.isfinite(e[:, j]), j])[::-1]
        sh = {q: r4(v[:int(len(v) * q)].sum() / v.sum()) for q in (0.01, 0.05, 0.1, 0.2)}
        hn = np.sort(np.abs(rng.normal(size=200000)))[::-1]
        sh_ref = {q: r4(hn[:int(len(hn) * q)].sum() / hn.sum()) for q in (0.01, 0.05, 0.1, 0.2)}
        ok = y[:, j] > 0
        q = np.quantile(y[ok, j], np.linspace(0, 1, 11))
        dec = np.clip(np.searchsorted(q, y[:, j], side='right') - 1, 0, 9)
        lr = np.log(B[:, j]) - np.log(np.where(ok, y[:, j], np.nan))
        by = [dict(decile=d + 1, lo=r4(q[d]), hi=r4(q[d + 1]), n=int((ok & (dec == d)).sum()), mape=r4(np.nanmean(e[ok & (dec == d), j])),
                   mean_log_resid_pct=r4(100 * np.nanmean(lr[ok & (dec == d)]))) for d in range(10)]
        mid = np.nanmean(e[ok & (dec >= 3) & (dec <= 6), j])
        out[k] = dict(top_share=sh, half_normal_reference=sh_ref, by_label_decile=by,
                      mape_if_all_deciles_like_middle_four=r4(mid), mape=r4(np.nanmean(e[ok, j])))
    return out


def structure_section(b, st):
    y, B = b['y'], b['B']
    ly = np.log(np.where(y > 0, y, np.nan)); r = np.log(B) - ly
    out = dict(cross_metric={})
    for a_, b_ in ((0, 1), (0, 2), (1, 2)):
        ok = np.isfinite(r[:, a_]) & np.isfinite(r[:, b_])
        out['cross_metric'][f'{K[a_]}-{K[b_]}'] = dict(resid_corr=r4(np.corrcoef(r[ok, a_], r[ok, b_])[0, 1]),
                                                     label_corr=r4(np.corrcoef(ly[ok, a_], ly[ok, b_])[0, 1]))
    # fellow eyes: same donor key, one OD and one OS, both scored
    key = [re.sub(r'[\s-]*O[DS]CN$', '', i.replace('0DCN', 'ODCN')) for i in b['ids']]
    side = [('OD' if re.search(r'ODCN$|0DCN$', i) else 'OS' if re.search(r'OSCN$', i) else None) for i in b['ids']]
    d = {}
    for n, (kk, s) in enumerate(zip(key, side)):
        if s:
            d.setdefault(kk, {})[s] = n
    pairs = np.array([(v['OD'], v['OS']) for v in d.values() if 'OD' in v and 'OS' in v])
    out['fellow_eye'] = dict(n_pairs=int(len(pairs)), same_fold_share=r4(np.mean(b['fold'][pairs[:, 0]] == b['fold'][pairs[:, 1]])))
    for j, k in enumerate(K):
        ok = np.isfinite(r[pairs[:, 0], j]) & np.isfinite(r[pairs[:, 1], j])
        p0, p1 = pairs[ok, 0], pairs[ok, 1]
        out['fellow_eye'][k] = dict(label_corr=r4(np.corrcoef(ly[p0, j], ly[p1, j])[0, 1]), pred_corr=r4(np.corrcoef(np.log(B[p0, j]), np.log(B[p1, j]))[0, 1]),
                                    resid_corr=r4(np.corrcoef(r[p0, j], r[p1, j])[0, 1]))
    # group bias share: how much of the log MSE is a per-(year, style) mean offset
    year = np.array([E.year_of(i) for i in b['ids']]); frame = st['black_frac'] > 0.01
    grp = np.unique(np.c_[year, frame], axis=0, return_inverse=True)[1].ravel()
    out['group_bias'] = {}
    for j, k in enumerate(K):
        ok = np.isfinite(r[:, j])
        gm = np.bincount(grp[ok], weights=r[ok, j]) / np.maximum(np.bincount(grp[ok]), 1)
        p = B[:, j] * np.exp(-gm[grp])
        e0 = np.nanmean(E.ape(B, y)[:, j]); e1 = np.nanmean(np.where(y[:, j] > 0, np.abs(p - y[:, j]) / np.where(y[:, j] > 0, y[:, j], 1) * 100, np.nan))
        out['group_bias'][k] = dict(n_groups=int(len(gm)), mse_share=r4(np.sum(np.bincount(grp[ok]) * gm ** 2) / np.sum(r[ok, j] ** 2)),
                                    oracle_group_offset_mape=r4(e1), mape=r4(e0))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1)))
    b = E.load_base(); st = E.image_stats(b)
    gf = np.load(X / 'submission2_search_20261103/green_features.npz', allow_pickle=True)
    assert np.array_equal(gf['indices'], b['idx'])
    seeds, single, hist = load_members(b)
    res = dict(workers=workers, n=int(len(b['y'])))
    res['variance'] = variance_section(b, seeds, single); print('variance done', flush=True)
    res['shared'] = shared_section(b, seeds, hist, gf); print('shared done', flush=True)
    res['dilution'] = dilution_section(b); print('dilution done', flush=True)
    res['concentration'] = concentration_section(b)
    res['structure'] = structure_section(b, st); print('structure done', flush=True)
    res['learnable'] = learnable_section(b, features(b, gf, st), workers); print('learnable done', flush=True)
    (OUT / 'ceiling_meta.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, default=float)[:3000])


if __name__ == '__main__':
    main()
