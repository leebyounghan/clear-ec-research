"""Follow-ups to ceiling_meta_cpu_20261103 (2026-11-03): (1) fellow-eye residual correlation with the label-dependent
shrinkage removed, and the (non-deployable, per-case inference) value of the fellow eye's prediction; (2) who carries
the CD error tail; (3) green-feature residual stacker ablations: without the E2 member feature, per-metric, with the
shrink chosen inside the training folds (nested)."""
import json, os, re, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
E = C.E; K = C.K; r4 = C.r4

def fit_predict(task):
    from sklearn.ensemble import HistGradientBoostingRegressor
    Xf, t, tr, te, j = task
    m = HistGradientBoostingRegressor(loss='absolute_error', max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                      min_samples_leaf=60, l2_regularization=1.0, random_state=0)
    ok = tr & np.isfinite(t[:, j]); m.fit(Xf[ok], t[ok, j])
    return m.predict(Xf[te])

def main():
    b = E.load_base(); st = E.image_stats(b); y, B, fold = b['y'], b['B'], b['fold']
    gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True)
    ly = np.log(np.where(y > 0, y, np.nan)); r = np.log(B) - ly; out = {}
    # (1) fellow eyes
    key = [re.sub(r'[\s-]*O[DS]CN$', '', i.replace('0DCN', 'ODCN')) for i in b['ids']]
    side = [('OD' if re.search(r'ODCN$|0DCN$', i) else 'OS' if re.search(r'OSCN$', i) else None) for i in b['ids']]
    d = {}
    for n, (kk, s) in enumerate(zip(key, side)):
        if s: d.setdefault(kk, {})[s] = n
    P = np.array([(v['OD'], v['OS']) for v in d.values() if 'OD' in v and 'OS' in v])
    out['fellow'] = {}
    for j, k in enumerate(K):
        ok = np.isfinite(r[P[:, 0], j]) & np.isfinite(r[P[:, 1], j]); a, c = P[ok, 0], P[ok, 1]
        # residual of log y given own prediction (calibrated residual): what the image did not explain
        A_ = np.c_[np.ones(len(a) * 2), np.log(B[np.r_[a, c], j])]; coef = np.linalg.lstsq(A_, ly[np.r_[a, c], j], rcond=None)[0]
        u = ly[:, j] - (coef[0] + coef[1] * np.log(B[:, j]))
        # residual of the prediction given own label (shrinkage removed)
        A2 = np.c_[np.ones(len(a) * 2), ly[np.r_[a, c], j]]; c2 = np.linalg.lstsq(A2, r[np.r_[a, c], j], rcond=None)[0]
        v = r[:, j] - (c2[0] + c2[1] * ly[:, j])
        # value of the fellow eye's prediction (in-sample OLS on pairs, both directions)
        Xo = np.c_[np.ones(len(a) * 2), np.log(B[np.r_[a, c], j]), np.log(B[np.r_[c, a], j])]
        cf = np.linalg.lstsq(Xo, ly[np.r_[a, c], j], rcond=None)[0]
        mse0 = np.mean((ly[np.r_[a, c], j] - A_ @ coef) ** 2); mse1 = np.mean((ly[np.r_[a, c], j] - Xo @ cf) ** 2)
        out['fellow'][k] = dict(n=int(len(a)), corr_calibrated_resid=r4(np.corrcoef(u[a], u[c])[0, 1]),
                                corr_resid_given_own_label=r4(np.corrcoef(v[a], v[c])[0, 1]),
                                fellow_pred_coef=r4(cf[2]), own_pred_coef=r4(cf[1]), mse_drop_with_fellow_pred=r4(1 - mse1 / mse0))
    # (2) CD tail
    e = E.ape(B, y); j = 0; ok = y[:, 0] > 0
    top = np.argsort(np.where(ok, e[:, 0], -1))[::-1][:90]
    year = np.array([E.year_of(i) for i in b['ids']]); fr = st['black_frac'] > 0.01
    out['cd_tail_top1pct'] = dict(n=90, mean_ape=r4(e[top, 0].mean()), share_over=r4((B[top, 0] > y[top, 0]).mean()),
                                  label_median=r4(np.median(y[top, 0])), label_all_median=r4(np.median(y[ok, 0])),
                                  label_lt1500=int((y[top, 0] < 1500).sum()), label_lt1500_all=int((y[ok, 0] < 1500).sum()),
                                  framed_share=r4(fr[top].mean()), framed_share_all=r4(fr.mean()),
                                  year_counts={int(k): int(v) for k, v in zip(*np.unique(year[top], return_counts=True))},
                                  examples=[dict(id=str(b['ids'][i]), y=float(y[i, 0]), B=round(float(B[i, 0]), 1), A=round(float(b['A'][i, 0]), 1),
                                                 P1=round(float(b['P1'][i, 0]), 1)) for i in top[:12]])
    # (3) stacker ablations, nested shrink (choose shrink in {0.25,0.5,0.75,1} by inner LOFO on the 4 training folds)
    F = C.features(b, gf, st)
    base = F['preds_meta']; g = F['preds_meta_green'][:, -4:]
    sets = {'green_all': F['preds_meta_green'], 'green_no_e2': np.c_[base, g[:, [0, 1, 3]]], 'green_det_only': np.c_[base, g[:, [0, 1]]]}
    t = ly - np.log(B); S = (0.25, 0.5, 0.75, 1.0); workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1)))
    grp = E.donors(b['ids']); out['stacker'] = {}
    ctx = get_context('fork')
    for name, Xf in sets.items():
        tasks, keys = [], []
        for f in range(5):
            for j in range(3):
                tasks.append((Xf, t, fold != f, fold == f, j)); keys.append(('outer', f, None, j))
                for h in range(5):
                    if h == f: continue
                    tr = (fold != f) & (fold != h); tasks.append((Xf, t, tr, fold == h, j)); keys.append(('inner', f, h, j))
        with ctx.Pool(workers) as pool:
            preds = pool.map(fit_predict, tasks, chunksize=1)
        outer = np.full_like(B, np.nan); inner = {}
        for kk, pr in zip(keys, preds):
            if kk[0] == 'outer': outer[fold == kk[1], kk[3]] = pr
            else: inner.setdefault((kk[1], kk[3]), np.full(len(B), np.nan))[fold == kk[2]] = pr
        p = B.copy(); chosen = {}
        for f in range(5):
            for j in range(3):
                tr = (fold != f) & (y[:, j] > 0); th = inner[(f, j)]
                cand = {s: np.mean(np.abs(B[tr, j] * np.exp(s * th[tr]) - y[tr, j]) / y[tr, j]) for s in (0.0,) + S}
                s = min(cand, key=cand.get); chosen[f'{f}_{K[j]}'] = s
                p[fold == f, j] = B[fold == f, j] * np.exp(s * outer[fold == f, j])
        sc = C.mape(p, y)
        out['stacker'][name] = dict(score={k: r4(v) for k, v in sc.items() if k != 'n'}, chosen_shrink=chosen,
                                    paired=E.paired_bootstrap(E.ape(p, y), E.ape(B, y), grp),
                                    per_fold={f: r4(C.mape(p[fold == f], y[fold == f])['mean'] - C.mape(B[fold == f], y[fold == f])['mean']) for f in range(5)})
        print(name, out['stacker'][name]['score'], out['stacker'][name]['paired'], flush=True)
    (C.OUT / 'ceiling_meta_followup.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ('fellow',)}, default=float)); print(json.dumps({k: v for k, v in out['cd_tail_top1pct'].items() if k != 'examples'}, default=float))
    for x in out['cd_tail_top1pct']['examples']: print(x)

if __name__ == '__main__':
    main()
