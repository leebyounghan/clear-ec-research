"""CV lane (2026-11-11, PI via coordinator): can our OOF explain submission 1's test pattern? Test minus OOF: CD -0.062, CV -0.061, HEX +0.395
(test 6.250 / 10.143 / 10.510 on 1000 images; OOF of the same pipeline, hybrid B, 6.312 / 10.204 / 10.115). No test images or labels are used.
Per-row OOF predictions: recipeA_analysis_20261028/hybrid_oof.npz B_hybrid_w0p5 (= submission 1, hybrid B; its OOF score matches the three numbers).
usage: hex_test_gap_20261111.py step1 | step2 | step3 | all       writes reads/hex_test_gap_20261111.json"""
import json, re, sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
from pathlib import Path
X = Path('/workspace/clear-ec-research/experiments'); OUT = X / 'lane_cv_20261108/reads'; OUT.mkdir(exist_ok=True)
z = np.load(X / 'recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True)
idx = z['indices'].astype(int); ids = np.array([str(i).strip() for i in z['ids']]); y = z['target'].astype(float); P = z['B_hybrid_w0p5'].astype(float); n = len(y)
V = y > 0; APE = np.where(V, np.abs(P - y) / np.where(V, y, 1) * 100, np.nan); K = ('CD', 'CV', 'HEX')
M = np.nanmean(APE, 0); G = np.array([-0.062, -0.061, 0.395]); T = M + G
donor = E.donors(ids); RNG = np.random.default_rng(20261111); R = 20000


def mean_rows(rows): return np.nanmean(APE[rows], 0)


def step1():
    """1000-row draws: row-random and donor-clustered (whole donors until >= 1000 rows)."""
    out = {'oof_mean': M.round(4).tolist(), 'observed_gap': G.tolist(), 'donors': int(donor.max() + 1), 'rows_per_donor_median': float(np.median(np.bincount(donor)))}
    rows_of = [np.where(donor == d)[0] for d in range(donor.max() + 1)]
    for kind in ('row_random', 'donor_clustered'):
        D = np.empty((R, 3)); sizes = []
        for r in range(R):
            if kind == 'row_random': rows = RNG.choice(n, 1000, replace=False)
            else:
                order = RNG.permutation(len(rows_of)); acc = []; c = 0
                for d in order:
                    acc.append(rows_of[d]); c += len(rows_of[d])
                    if c >= 1000: break
                rows = np.concatenate(acc); sizes.append(len(rows))
            D[r] = mean_rows(rows) - M
        mu, sd, C = D.mean(0), D.std(0), np.cov(D.T); Ci = np.linalg.inv(C); d2 = np.einsum('ij,jk,ik->i', D - mu, Ci, D - mu); g2 = float((G - mu) @ Ci @ (G - mu))
        pct = [float((D[:, 0] <= G[0]).mean()), float((D[:, 1] <= G[1]).mean()), float((D[:, 2] >= G[2]).mean())]
        joint = float(((D[:, 0] <= G[0]) & (D[:, 1] <= G[1]) & (D[:, 2] >= G[2])).mean())
        out[kind] = dict(sd=sd.round(4).tolist(), mean=mu.round(4).tolist(), corr=np.corrcoef(D.T).round(3).tolist(), z=((G - mu) / sd).round(2).tolist(),
                         tail_prob_each=[round(p, 4) for p in pct], joint_tail_prob=round(joint, 5), mahalanobis_d2=round(g2, 2), mahalanobis_tail=round(float((d2 >= g2).mean()), 5),
                         draw_size=dict(median=int(np.median(sizes)), p5=int(np.percentile(sizes, 5)), p95=int(np.percentile(sizes, 95))) if sizes else 1000)
    return out


def covariates():
    """Per-row covariates (raw values or categories). Returns {name: (values, kind)} with kind 'num' or 'cat'."""
    pos = {int(i): k for k, i in enumerate(idx)}; cov = {}
    cov['label_CD'], cov['label_CV'] = (y[:, 0], 'num'), (y[:, 1], 'num'); cov['label_HEX'] = (np.where(V[:, 2], y[:, 2], np.nan), 'num')
    cov['pred_CD'], cov['pred_CV'], cov['pred_HEX'] = (P[:, 0], 'num'), (P[:, 1], 'num'), (P[:, 2], 'num')
    fmt = np.array([re.sub(r'[0-9]', '9', re.sub(r'[A-Z]', 'A', s)) for s in ids]); cov['source_P1_P2'] = (np.where(fmt == '9999-99', 'P1', 'P2'), 'cat')
    cov['id_format'] = (np.where(fmt == '9999-99', 'P1 nnnn-yy', np.where(np.char.find(fmt, '-AAAA') >= 0, 'P2 hyphen', 'P2 space')), 'cat')
    st = np.load(X / 'eval_harness_20261031/image_stats.npz')
    for k in ('mean', 'contrast', 'black_frac', 'focus'):
        v = np.full(n, np.nan)
        for i, a in zip(st['idx'].astype(int), st[k]): v[pos[int(i)]] = a
        cov[f'img_{k}'] = (v, 'num')
    sg = np.load(X / 'lane_hex_20261108/preds/seg_all.npz'); nm = list(sg['names']); F = np.full((n, sg['F'].shape[1]), np.nan)
    for i, a in zip(sg['frames'].astype(int), sg['F']): F[pos[int(i)]] = a
    for k in ('n_int', 'area_cv', 'h6', 'cd_mm2', 'coverage'): cov[f'seg_{k}'] = (F[:, nm.index(k)], 'num')
    gf = np.load(X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); gpos = {int(i): k for k, i in enumerate(gf['indices'])}
    for k, vk in (('f_active10', 'f_active10_valid'), ('n_det', 'n_det_valid')):
        v = np.full(n, np.nan)
        for i in range(n):
            j = gpos.get(int(idx[i]))
            if j is not None and gf[vk][j]: v[i] = gf[k][j]
        cov[f'det_{k}'] = (v, 'num')
    lw = np.load(X / 'meta_ceiling_20261103/sweep/legible_window.npz', allow_pickle=True); lnm = list(lw['names'].astype(str)); LF = np.full((n, len(lnm)), np.nan)
    LF[:] = lw['F'][idx] if len(lw['F']) > idx.max() else np.nan
    if 'top_hull_n' in lnm and 'whole_n' in lnm: cov['legible_fraction'] = (LF[:, lnm.index('top_hull_n')] / np.maximum(LF[:, lnm.index('whole_n')], 1), 'num')
    return cov


def bins_of(v, kind, nb=10):
    if kind == 'cat':
        cats = sorted(set(v.tolist())); return np.array([cats.index(c) for c in v]), [str(c) for c in cats]
    ok = np.isfinite(v); edges = np.unique(np.nanpercentile(v[ok], np.linspace(0, 100, nb + 1))); b = np.full(len(v), len(edges) - 1)
    b[ok] = np.clip(np.searchsorted(edges, v[ok], side='right') - 1, 0, len(edges) - 2)
    labels = [f'[{edges[i]:.4g}, {edges[i + 1]:.4g})' for i in range(len(edges) - 1)] + (['missing'] if (~ok).any() else [])
    if not (~ok).any(): b = np.clip(b, 0, len(edges) - 2)
    return b, labels


def tilt(b, targets, metrics=(0, 1, 2)):
    """Minimum-KL reweighting of bins (q_b = p_b exp(lam . A_b) / Z) so that the weighted mean APE hits targets on the given metrics."""
    nb = b.max() + 1; p = np.bincount(b, minlength=nb) / n
    Ab = np.array([[np.nanmean(APE[b == k, m]) if np.isfinite(APE[b == k, m]).any() else M[m] for m in range(3)] for k in range(nb)]); mets = list(metrics); t = np.asarray(targets)[mets]
    def dual(lam):
        s = Ab[:, mets] @ lam; s = s - s.max(); w = p * np.exp(s); Z = w.sum(); return np.log(Z) + (Ab[:, mets] @ lam).max() - lam @ t
    res = minimize(dual, np.zeros(len(mets)), method='BFGS', options=dict(maxiter=500, gtol=1e-9))
    s = Ab[:, mets] @ res.x; w = p * np.exp(s - s.max()); q = w / w.sum(); fit = q @ Ab
    kl = float(np.sum(q[q > 0] * np.log(q[q > 0] / p[q > 0])))
    return dict(q=q, p=p, fit_gap=fit - M, kl=kl, resid=float(np.max(np.abs((fit - np.asarray(targets))[mets]))), ratio=q / np.maximum(p, 1e-12), Ab=Ab)


def describe(tl, labels, k=3):
    r = tl['ratio']; order = np.argsort(-np.abs(np.log(np.maximum(r, 1e-9)))); return [f'{labels[i]}: x{r[i]:.2f} ({tl["p"][i]*100:.1f}% -> {tl["q"][i]*100:.1f}%)' for i in order[:k]]


def step2():
    cov = covariates(); res = {}
    for name, (v, kind) in cov.items():
        b, lab = bins_of(v, kind); tl = tilt(b, T); th = tilt(b, T, metrics=(2,))
        res[name] = dict(feasible=bool(tl['resid'] < 0.005), resid=round(tl['resid'], 4), kl=round(tl['kl'], 4), max_ratio=round(float(tl['ratio'].max()), 2), min_ratio=round(float(tl['ratio'].min()), 2),
                         shift=describe(tl, lab), fit_gap=tl['fit_gap'].round(3).tolist(), hex_only=dict(kl=round(th['kl'], 4), implied_gap=th['fit_gap'].round(3).tolist(), shift=describe(th, lab)))
    feas = sorted([k for k in res if res[k]['feasible']], key=lambda k: res[k]['kl']); best = (feas or sorted(res, key=lambda k: res[k]['resid']))[:3]; pairs = {}
    for i in range(len(best)):
        for j in range(i + 1, len(best)):
            (v1, k1), (v2, k2) = cov[best[i]], cov[best[j]]; b1, l1 = bins_of(v1, k1, 5); b2, l2 = bins_of(v2, k2, 5); b = b1 * (b2.max() + 1) + b2
            _, inv = np.unique(b, return_inverse=True); labs = {}
            for u in np.unique(inv): r0 = np.where(inv == u)[0][0]; labs[u] = f'{best[i]} {l1[b1[r0]]} & {best[j]} {l2[b2[r0]]}'
            tl = tilt(inv, T); pairs[f'{best[i]} x {best[j]}'] = dict(feasible=bool(tl['resid'] < 0.005), resid=round(tl['resid'], 4), kl=round(tl['kl'], 4), max_ratio=round(float(tl['ratio'].max()), 2), shift=describe(tl, [labs[u] for u in range(inv.max() + 1)]))
    return dict(single=res, ranked_feasible=feas, pairs=pairs)


def step3():
    out = {}
    # (a) HEX label shift alone (+0.395 on HEX), implications for CD and CV
    b, lab = bins_of(np.where(V[:, 2], y[:, 2], np.nan), 'num'); th = tilt(b, T, metrics=(2,))
    low = (y[:, 2] < 0.45) & V[:, 2]; qrow = th['q'][b] / th['p'][b] / n
    out['a_label_shift'] = dict(kl=round(th['kl'], 4), implied_gap=th['fit_gap'].round(3).tolist(), shift=describe(th, lab, 4),
                                share_hex_below_0p45=[round(float(low.mean()), 4), round(float(qrow[low].sum()), 4)], mean_hex_label=[round(float(y[V[:, 2], 2].mean()), 4), round(float((qrow * np.where(V[:, 2], y[:, 2], 0)).sum() / qrow[V[:, 2]].sum()), 4)])
    # (b) level offset: test labels = y (1 + b)
    bs = np.round(np.arange(-0.10, 0.1001, 0.0025), 4); curves = {}
    for m, k in enumerate(K):
        vv = V[:, m]; curves[k] = [float(np.mean(np.abs(P[vv, m] - y[vv, m] * (1 + bb)) / (y[vv, m] * (1 + bb))) * 100 - M[m]) for bb in bs]
    hx = np.array(curves['HEX']); cross = [float(bs[i]) for i in range(1, len(bs)) if (hx[i - 1] - 0.395) * (hx[i] - 0.395) <= 0]
    out['b_offset'] = dict(b_grid=bs.tolist(), d_mape=curves, hex_b_for_plus_0p395=cross, cd_min=[round(min(curves['CD']), 4), float(bs[int(np.argmin(curves['CD']))])], cv_min=[round(min(curves['CV']), 4), float(bs[int(np.argmin(curves['CV']))])])
    return out


def hex_label_bins():
    return bins_of(np.where(V[:, 2], y[:, 2], np.nan), 'num')


def step2x():
    """Yardsticks for step 2: (i) how large a composition shift ordinary 1000-image sampling produces (KL of a random 1000-row histogram vs
    the pool, per covariate, 10 bins); (ii) the needed label_HEX tilt as a z-score along its own direction (projection of the random
    histogram deviation on log q/p); (iii) the full label_HEX bin tables for the all-three and HEX-only tilts."""
    cov = covariates(); rng = np.random.default_rng(7); D = 4000; out = {'kl_random_1000': {}}
    draws = [rng.choice(n, 1000, replace=False) for _ in range(D)]
    for name, (v, kind) in cov.items():
        b, _ = bins_of(v, kind); nb = b.max() + 1; p = np.bincount(b, minlength=nb) / n; kl = []
        for rows in draws:
            ph = np.bincount(b[rows], minlength=nb) / 1000; m = ph > 0; kl.append(float(np.sum(ph[m] * np.log(ph[m] / p[m]))))
        out['kl_random_1000'][name] = dict(median=round(float(np.median(kl)), 4), p99=round(float(np.percentile(kl, 99)), 4))
    b, lab = hex_label_bins(); nb = b.max() + 1; t3 = tilt(b, T); th = tilt(b, T, metrics=(2,)); p = t3['p']
    for nm, tl in (('all3', t3), ('hexonly', th)):
        d = np.log(np.maximum(tl['q'], 1e-12) / np.maximum(p, 1e-12)); need = float(np.sum((tl['q'] - p) * d)); proj = []
        for rows in draws: proj.append(float(np.sum((np.bincount(b[rows], minlength=nb) / 1000 - p) * d)))
        out[f'label_HEX_{nm}_direction_z'] = round(need / float(np.std(proj)), 2)
        out[f'label_HEX_{nm}_direction_tail'] = round(float((np.array(proj) >= need).mean()), 4)
    hv = np.where(V[:, 2], y[:, 2], np.nan); rows = []
    for k in range(nb):
        rows.append(dict(bin=lab[k], p=round(float(p[k]), 4), q_all3=round(float(t3['q'][k]), 4), q_hexonly=round(float(th['q'][k]), 4), ape=t3['Ab'][k].round(3).tolist()))
    out['label_HEX_table'] = rows
    def summ(q):
        w = q[b] / p[b] / n; m = V[:, 2]; w = w[m] / w[m].sum(); h = y[m, 2]; mu = float(w @ h)
        return dict(share_below_0p45=round(float(w[h < 0.45].sum()), 4), share_ge_0p66=round(float(w[h >= 0.66].sum()), 4), mean=round(mu, 4), sd=round(float(np.sqrt(w @ (h - mu) ** 2)), 4))
    out['label_HEX_summary'] = dict(pool=summ(p), all3=summ(t3['q']), hexonly=summ(th['q']))
    return out


def step4():
    """Decision matrix (OOF only, nothing applied): HEX MAPE change of each HEX adjustment under each hypothesis, vs no adjustment.
    Hypotheses: H0 noise (uniform rows); H0 given the gap (random 1000-row draws whose HEX gap is within 0.05 of +0.395, and the subset
    that also has CD and CV gaps <= 0); offset b- / b+ (test HEX labels = y (1 + b), the two b that cost +0.395); composition (rows
    reweighted by the label_HEX all-three tilt). Adjustments: none; scale 1 + b- and 1 + b+ (the exact offset fixes); the
    composition-optimal stretch+scale fit on the tilted OOF (in-sample, so an upper bound on its gain)."""
    h = V[:, 2]; yh = y[:, 2]; ph = P[:, 2]; b, _ = hex_label_bins(); t3 = tilt(b, T); w_comp = np.where(h, t3['q'][b] / t3['p'][b], 0.0)
    st3 = json.loads((OUT / 'hex_test_gap_20261111.json').read_text())['step3']['b_offset']['hex_b_for_plus_0p395']; bm, bp = min(st3), max(st3)
    lph = np.log(np.where(h, ph, 1.0)); mu = lph[h].mean()
    def wmed(vals, wts):
        o = np.argsort(vals); c = np.cumsum(wts[o]); return float(vals[o][np.searchsorted(c, c[-1] / 2)])
    best = None
    for g in np.round(np.arange(0.80, 1.4001, 0.02), 2):          # MAPE-optimal scale for weighted |c q - y| / y is the weighted median of y/q with weights w q / y
        q = np.exp(g * (lph - mu) + mu); c = wmed((yh / q)[h], (w_comp * q / np.where(h, yh, 1))[h]); m = np.sum(w_comp[h] * np.abs(c * q[h] - yh[h]) / yh[h]) / w_comp[h].sum()
        if best is None or m < best[0]: best = (m, g, c)
    adj = {'none': ph, f'scale {1 + bm:.4f}': ph * (1 + bm), f'scale {1 + bp:.4f}': ph * (1 + bp), f'stretch g={best[1]:.2f} c={best[2]:.4f}': best[2] * np.exp(best[1] * (lph - mu) + mu)}
    ape_rows = {k: np.where(h, np.abs(v - yh) / np.where(h, yh, 1) * 100, np.nan) for k, v in adj.items()}
    def mape(pred, lab_mult=1.0, w=None):
        w = np.where(h, 1.0, 0.0) if w is None else w; lab = yh * lab_mult; return float(np.sum(w[h] * np.abs(pred[h] - lab[h]) / lab[h]) / w[h].sum() * 100)
    hyps = {'H0 noise': dict(), f'offset b={bm:+.4f}': dict(lab_mult=1 + bm), f'offset b={bp:+.4f}': dict(lab_mult=1 + bp), 'composition (label_HEX tilt)': dict(w=w_comp)}
    mat = {}
    for hn, kw in hyps.items():
        base = mape(adj['none'], **kw); mat[hn] = dict(base_hex_mape=round(base, 3), **{an: round(mape(av, **kw) - base, 3) for an, av in adj.items() if an != 'none'})
    rng = np.random.default_rng(11); gapsel = []; sub = []
    for _ in range(40000):
        rows = rng.choice(n, 1000, replace=False); g3 = np.nanmean(APE[rows], 0) - M
        if abs(g3[2] - 0.395) <= 0.05:
            d = {an: float(np.nanmean(ape_rows[an][rows]) - np.nanmean(ape_rows['none'][rows])) for an in adj if an != 'none'}; gapsel.append(d)
            if g3[0] <= 0 and g3[1] <= 0: sub.append(d)
    for nm, L in (('H0 given HEX gap 0.395+-0.05', gapsel), ('H0 given HEX gap and CD, CV gaps <= 0', sub)):
        mat[nm] = dict(n_draws=len(L), **{an: round(float(np.mean([d[an] for d in L])), 3) for an in adj if an != 'none'}, **{f'{an} P(helps)': round(float(np.mean([d[an] < 0 for d in L])), 3) for an in adj if an != 'none'})
    return dict(b_minus=bm, b_plus=bp, stretch_fit=dict(g=best[1], c=round(best[2], 4), tilted_mape=round(best[0] * 100, 3)), matrix=mat)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'; path = OUT / 'hex_test_gap_20261111.json'
    res = json.loads(path.read_text()) if path.exists() else {}
    if what == 'step2x': res['step2x'] = step2x(); print(json.dumps({k: v for k, v in res['step2x'].items() if k != 'kl_random_1000'}), flush=True); print(json.dumps(res['step2x']['kl_random_1000']), flush=True)
    if what == 'step4': res['step4'] = step4(); print(json.dumps(res['step4'], indent=1), flush=True)
    if what in ('step1', 'all'): res['step1'] = step1(); print(json.dumps(res['step1']), flush=True)
    if what in ('step2', 'all'): res['step2'] = step2(); print(json.dumps({k: res['step2'][k] for k in ('ranked_feasible', 'pairs')}), flush=True)
    if what in ('step3', 'all'): res['step3'] = step3(); print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('b_grid', 'd_mape')} for k, v in res['step3'].items()}), flush=True)
    path.write_text(json.dumps(res, indent=1, default=float))
