"""HEX candidate chain (HEX lane, 2026-11-08, PI rule: keep anything that lowers HEX, stack the next try on it; never choose a
setting on the rows you report). A candidate is {outer fold f: pre-stretch HEX for all rows from models that never saw fold
f's labels}. Final HEX for fold f = stretch fit on the other folds' rows of that vector, applied to fold f. Each step offers
options; for outer fold f the option is chosen by HEX on the other four folds (their own OOF values), so no fold picks its own
setting. A step is kept when it lowers HEX on folds 0+4 against the previous candidate. CD and CV stay as in the best pipeline.
usage: chain.py [steps...]  writes preds/chain_hex.npz (candidate + final HEX) and appends to RESULTS.tsv."""
import sys, json, time, os
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source')
import hexlib as L
BP = L.BP; N = L.N; y, fold, good, idx = L.y, L.fold, L.good, L.idx; NS = BP.N; allr = np.ones(N, bool)
ok = (y[:, 2] > 0)


def lg(v): v = np.asarray(v, float); return np.where(v > 0, np.log(np.where(v > 0, v, 1)), np.nan)


def final(cand):
    P = L.REF.copy()
    for f in range(5):
        C_ = BP.H.copy(); C_[:, 2] = cand[f]; prm = NS.fit_stretch(C_, y, (fold != f) & good); P[fold == f, 2] = NS.apply_stretch(C_[fold == f], prm)[:, 2]
    return P


def oofhex(cand):
    """per-row OOF final HEX: row in fold g gets fold g's final value"""
    return final(cand)[:, 2]


def choose(options, label):
    """options: list of (name, cand). For each outer f choose the option with lowest HEX on folds != f (rows' own OOF)."""
    finals = [final(c)[:, 2] for _, c in options]; E = [np.abs(p - y[:, 2]) / np.where(ok, y[:, 2], 1) for p in finals]
    out = {}; picks = []
    for f in range(5):
        sc = [float(e[(fold != f) & ok].mean()) for e in E]; j = int(np.argmin(sc)); picks.append(options[j][0]); out[f] = options[j][1][f]
    return out, picks


def nested_cor(cand, F, alpha=10.0):
    """ridge on log residual (log y - log cand[f]) with features F (N x k) + log cand[f], fit on folds != f"""
    F = np.asarray(F, float).reshape(N, -1); out = {}
    for f in range(5):
        lb = lg(cand[f]); X = F.copy(); tr = (fold != f) & good & ok; miss = ~np.isfinite(X); med = np.nanmedian(X[tr], 0); X[miss] = np.take(med, np.where(miss)[1]); X = np.c_[X, lb]
        t = lg(np.where(ok, y[:, 2], 1)) - lb; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Z = (X - mu) / sd; A = Z[tr]
        w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (t[tr] - t[tr].mean())); out[f] = cand[f] * np.exp(Z @ w + t[tr].mean())
    return out


def read(name, cand, prev):
    P = final(cand); Pp = final(prev)
    def h(Q, r): return float(np.mean(np.abs(Q[r & ok, 2] - y[r & ok, 2]) / y[r & ok, 2]) * 100)
    o = dict(step=name, f04=round(h(P, L.S04), 4), all5=round(h(P, allr), 4), d_prev_f04=round(h(P, L.S04) - h(Pp, L.S04), 4), d_prev_all5=round(h(P, allr) - h(Pp, allr), 4), d_best_f04=round(h(P, L.S04) - h(L.REF, L.S04), 4), d_best_all5=round(h(P, allr) - h(L.REF, allr), 4), folds_vs_prev=[round(h(P, fold == f) - h(Pp, fold == f), 3) for f in range(5)])
    print(json.dumps(o), flush=True); return o


# ---- steps -------------------------------------------------------------------------------------------------------------
def base_hex(w):
    A, P1 = BP._b['A'], BP._b['P1']; h = BP._have; bh = np.exp(w * np.log(P1[:, 2]) + (1 - w) * np.log(A[:, 2])); hh = bh.copy()
    hh[h] = np.exp((1 - 0.28) * np.log(bh[h]) + 0.28 * (-0.2084 + 0.5989 * BP.f6[h])); return hh


def step_ap1(cand):
    opts = [(f'w{w}', {f: base_hex(w) * np.exp(BP.BEST[f][:, 2]) for f in range(5)}) for w in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    return choose(opts, 'ap1')


def step_feats(cand, path, keep=('_h6', '_wh6', '_h6c', '_wh6c', '_w1h6')):
    z = np.load(path); nm = [str(s) for s in z['names']]; F = z['F'][idx].astype(float); cols = [j for j, k in enumerate(nm) if k.endswith(keep)]
    opts = [('none', cand)] + [(nm[j], nested_cor(cand, lg(F[:, j]))) for j in cols]
    return choose(opts, path)


def gload(names):
    R = [np.load(BP.O / 'graph_sweep' / f'{q}.npz') for q in names]; return {f: np.mean([z[f'f{f}'][:, 2] for z in R], 0) for f in range(5)}


OLDG = {f: BP.GRAPH[f][:, 2] for f in range(5)}


def step_graphens(cand):
    opts = [('current', cand)]
    for nm, names in (('s10x3', ['d64_l5_noctx_s10', 'd64_l5_pre47_s10', 'd64_l5_pre48_s10']), ('all5', ['d64_l5_noctx_s10', 'd64_l5_pre47_s10', 'd64_l5_pre48_s10', 'd64_l5_pre47_s3', 'd64_l5_pre48_s3'])):
        g = gload(names); opts.append((nm, {f: cand[f] * np.exp(g[f] - OLDG[f]) for f in range(5)}))
    return choose(opts, 'graphens')


def step_gscale(cand):
    return choose([(f's{s_}', {f: cand[f] * np.exp((s_ - 1) * OLDG[f]) for f in range(5)}) for s_ in (0.8, 0.9, 1.0, 1.1, 1.2)], 'gscale')


def step_quad(cand):
    opts = [('none', cand)]
    for a in (3.0, 30.0, 300.0):
        opts.append((f'quad_a{a}', {f: nested_cor({g: cand[g] for g in range(5)}, lg(cand[f]) ** 2, alpha=a)[f] for f in range(5)}))
    return choose(opts, 'quad')


def step_gnn(cand, name, mode='ens'):
    z = np.load(f'{L.LANE}/preds/{name}.npz'); new = {f: z[f'f{f}'][:, 2] for f in range(5)}
    opts = [('none', cand)]
    for w in (0.25, 0.5, 0.75, 1.0): opts.append((f'mix{w}', {f: cand[f] * np.exp(w * (new[f] - OLDG[f])) for f in range(5)}))
    opts.append(('stack', {f: nested_cor(cand, new[f])[f] for f in range(5)}))
    return choose(opts, name)


def step_h6w(cand):
    A, P1 = BP._b['A'], BP._b['P1']; h = BP._have; bh = np.exp(0.7 * np.log(P1[:, 2]) + 0.3 * np.log(A[:, 2])); d = np.zeros(N); d[h] = (-0.2084 + 0.5989 * BP.f6[h]) - np.log(bh[h])
    return choose([(f'w{w}', {f: cand[f] * np.exp((w - 0.28) * d) for f in range(5)}) for w in (0.0, 0.05, 0.10, 0.16, 0.22, 0.28, 0.34, 0.40)], 'h6w')


def step_ridge(cand, F, label, alphas=(10.0, 100.0, 1000.0)):
    return choose([('none', cand)] + [(f'a{a}', nested_cor(cand, F, alpha=a)) for a in alphas], label)


def imgstats():
    z = np.load('/workspace/clear-ec-research/experiments/eval_harness_20261031/image_stats.npz'); assert (z['idx'] == idx).all()
    return np.c_[np.log(z['mean'] + 1), np.log(z['contrast'] + 1), z['black_frac'], np.log(z['focus'] + 1)]


def mape_fit(cand, Ff, lam, label):
    """per outer fold f: w minimising mean |1 - cand*exp(Z w + b)/y| (smooth abs) on folds != f; Ff = {f: N x k} features"""
    from scipy.optimize import minimize
    out = {}
    for f in range(5):
        X = np.asarray(Ff[f], float).reshape(N, -1).copy(); tr = (fold != f) & good & ok; miss = ~np.isfinite(X); med = np.nanmedian(X[tr], 0); X[miss] = np.take(med, np.where(miss)[1])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Z = (X - mu) / sd; Z = np.c_[Z, np.ones(N)]; k = Z.shape[1]
        lp = lg(cand[f]); ly = lg(np.where(ok, y[:, 2], 1)); Zt, d = Z[tr], (lp - ly)[tr]
        def fg(w):
            r = np.exp(d + Zt @ w); e = r - 1; a = np.sqrt(e * e + 1e-6); pen = lam * np.sum(w[:-1] ** 2)
            return a.mean() + pen, (Zt.T @ (e / a * r)) / len(d) + np.r_[2 * lam * w[:-1], 0]
        w = minimize(fg, np.zeros(k), jac=True, method='L-BFGS-B').x; out[f] = cand[f] * np.exp(Z @ w)
    return out


def unc_feats(cand):
    A, P1 = BP._b['A'], BP._b['P1']; h = BP._have; dap = np.abs(np.log(A[:, 2]) - np.log(P1[:, 2])); dh6 = np.zeros(N)
    bh = np.exp(0.5 * np.log(P1[:, 2]) + 0.5 * np.log(A[:, 2])); dh6[h] = np.abs((-0.2084 + 0.5989 * BP.f6[h]) - np.log(bh[h])); dh6s = np.zeros(N); dh6s[h] = (-0.2084 + 0.5989 * BP.f6[h]) - np.log(bh[h])
    R = [np.load(BP.O / 'graph_sweep' / f'{q}.npz') for q in ['d64_l5_noctx_s10', 'd64_l5_pre47_s10', 'd64_l5_pre48_s10']]
    gs = {f: np.std([z[f'f{f}'][:, 2] for z in R], 0) for f in range(5)}
    return {f: np.c_[dap, dh6, dh6s, gs[f], h.astype(float), imgstats(), lg(cand[f])] for f in range(5)}


def step_mapelin(cand):
    U = unc_feats(cand); opts = [('none', cand)]
    for lam in (1e-4, 1e-3, 1e-2): opts.append((f'unc_l{lam}', mape_fit(cand, U, lam, 'unc')))
    opts.append(('lp_only', mape_fit(cand, {f: lg(cand[f]) for f in range(5)}, 1e-4, 'lp')))
    return choose(opts, 'mapelin')


def step_uncridge(cand):
    U = unc_feats(cand); opts = [('none', cand)]
    for a in (10.0, 100.0, 1000.0): opts.append((f'a{a}', {f: nested_cor(cand, U[f][:, :-1], alpha=a)[f] for f in range(5)}))
    return choose(opts, 'uncridge')


def imgrich():
    z = np.load(f'{L.LANE}/feats/img_rich.npz'); assert (z['idx'] == idx).all(); return z['F'].astype(float)


def step_imgrich(cand):
    G = imgrich(); opts = [('none', cand)] + [(f'r_a{a}', nested_cor(cand, G, alpha=a)) for a in (30.0, 300.0, 3000.0)]
    opts.append(('mape_l1e-3', mape_fit(cand, {f: np.c_[G, lg(cand[f])] for f in range(5)}, 1e-3, 'imgrich')))
    opts.append(('mape_l1e-2', mape_fit(cand, {f: np.c_[G, lg(cand[f])] for f in range(5)}, 1e-2, 'imgrich')))
    return choose(opts, 'imgrich')


def featgroup(path, pick=None):
    z = np.load(path); nm = [str(s) for s in z['names']]; F = z['F'][idx].astype(float); cols = [j for j, k in enumerate(nm) if pick is None or pick(k)]
    G = F[:, cols]; G[~np.isfinite(G)] = np.nan; pos = np.nanmin(G, 0) > 0; G[:, pos] = np.log(G[:, pos])
    keep = (np.mean(np.isnan(G), 0) < 0.5) & (np.nanstd(G, 0) > 1e-9); return G[:, keep]


def step_group(cand, G, label, alphas=(100.0, 1000.0, 10000.0)):
    return choose([('none', cand)] + [(f'a{a}', nested_cor(cand, G, alpha=a)) for a in alphas], label)


XR = '/workspace/clear-ec-research/experiments/'
FAM = {'dinov3_s46': ('lane_gpu3_20261106/runs/dinov3_b16_f0_s46', 'lane_gpu3_20261106/runs/dinov3_b16_f4_s46', 'dinov3_b16'),
       'dinov3_s47': ('lane_gpu2_20261106/runs/dinov3_b16_f0_s47', 'lane_gpu2_20261106/runs/dinov3_b16_f4_s47', 'dinov3_b16'),
       'convnext': ('lane_gpu3_20261106/runs/convnext_b32_f0_s46', 'lane_gpu3_20261106/runs/convnext_b32_f4_s46', 'convnext_b32'),
       'dinov2': ('lane_gpu3_20261106/runs/dinov2_b14_f0_s46', 'lane_gpu3_20261106/runs/dinov2_b14_f4_s46', 'dinov2_b14')}


def fam_hex(d0, d4, arch, epochs=range(16, 21)):
    """HEX of a family member on folds 0 and 4 only (mean of log over epochs), NaN elsewhere (and on a fold whose epochs are
    not all there yet)"""
    pos = {int(i): k for k, i in enumerate(idx)}; out = np.full(N, np.nan)
    for f, d in ((0, d0), (4, d4)):
        if not all(os.path.exists(f'{XR}{d}/runs/fold{f}/{arch}/predictions_epoch_{e:02d}.npz') for e in epochs): continue
        Ls = []
        for e in epochs:
            z = np.load(f'{XR}{d}/runs/fold{f}/{arch}/predictions_epoch_{e:02d}.npz'); p = np.full(N, np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred'][:, 2]; Ls.append(np.log(p))
        P = np.exp(np.mean(Ls, 0)); out[fold == f] = P[fold == f]
    return out


def step_family(cand):
    """folds 0 and 4 only (members exist there): mix family HEX into the base at share c, cand[0] and cand[4] change, choice
    for outer fold 0 is driven by fold 4 rows and vice versa. cand[1..3] untouched."""
    M = {k: fam_hex(*v) for k, v in FAM.items()}; A, P1 = BP._b['A'], BP._b['P1']; lbh = 0.7 * np.log(P1[:, 2]) + 0.3 * np.log(A[:, 2])
    sets = {'all4': list(FAM), 'nodv2': ['dinov3_s46', 'dinov3_s47', 'convnext'], 'convnext': ['convnext'], 'dinov3x2': ['dinov3_s46', 'dinov3_s47']}
    opts = [('none', cand)]
    for sn, ks in sets.items():
        lF = np.mean([np.log(M[k]) for k in ks], 0); d = np.where(np.isfinite(lF), lF - lbh, 0.0)
        for c in (0.1, 0.2, 0.3, 0.4, 0.5):
            new = dict(cand); new[0] = cand[0] * np.exp(c * d); new[4] = cand[4] * np.exp(c * d); opts.append((f'{sn}_c{c}', new))
    return choose(opts, 'family')


CPS = ('lane_gpu3_20261106/runs/cpsam_c512_f0_s46', 'lane_gpu3_20261106/runs/cpsam_c512_f4_s46', 'cpsam_c512')


HCPS = ('lane_hex_20261108/runs/hexcps_c512_f0', 'lane_hex_20261108/runs/hexcps_c512_f4', 'cpsam_c512')


def cps_mix(cand, spec, ep, c):
    """mix a fold-0/4 CNN member's HEX into the HEX base at share c (folds 0 and 4 change, others untouched)"""
    A, P1 = BP._b['A'], BP._b['P1']; lbh = 0.7 * np.log(P1[:, 2]) + 0.3 * np.log(A[:, 2])
    lF = np.log(fam_hex(*spec, epochs=ep)); d = np.where(np.isfinite(lF), lF - lbh, 0.0)
    new = dict(cand); new[0] = cand[0] * np.exp(c * d); new[4] = cand[4] * np.exp(c * d); return new


def choose_cross(options, label):
    """for members that exist on folds 0 and 4 only: the option for fold 0 is the one with lowest HEX on fold 4 rows, the
    stretch for that read fit on folds 1-3 only (fold 0 labels play no part), and the other way round. Folds 1-3 keep the
    first option (they are the same in all options)."""
    out = {f: options[0][1][f] for f in range(5)}; picks = [options[0][0]] * 5
    for f, g in ((0, 4), (4, 0)):
        tr = ~np.isin(fold, [f, g]) & good; m = (fold == g) & ok; sc = []
        for _, c in options:
            Cc = BP.H.copy(); Cc[:, 2] = c[g]; P = NS.apply_stretch(Cc, NS.fit_stretch(Cc, y, tr))[:, 2]; sc.append(float(np.mean(np.abs(P[m] - y[m, 2]) / y[m, 2])))
        j = int(np.argmin(sc)); out[f] = options[j][1][f]; picks[f] = options[j][0]
    return out, picks


def step_cpsam(cand, spec=CPS, name='cpsam'):
    """CPSAM 512 HEX mixed into the HEX base on folds 0 and 4 (like step_family): epochs 8-12 / 10-12 / 12, share c; the pick
    for fold 0 is made on fold 4 rows and vice versa (choose_cross; never on the rows it is applied to). Needs both folds."""
    opts = [('none', cand)]
    for en, ep in (('e8_12', range(8, 13)), ('e10_12', range(10, 13)), ('e12', range(12, 13))):
        for c in (0.1, 0.2, 0.3, 0.4, 0.5): opts.append((f'{en}_c{c}', cps_mix(cand, spec, ep, c)))
    return choose_cross(opts, name)


def step_cps_fix(cand, spec=CPS, c=0.2):
    """fixed share (no choice at all): epochs 8-12, share c"""
    return cps_mix(cand, spec, range(8, 13), c), [f'fixed e8_12 c{c}'] * 5


NOSSL = ('lane_hex_20261108/runs/hexcps_nossl_c512_f0', 'lane_hex_20261108/runs/hexcps_nossl_c512_f4', 'cpsam_c512')
MTNOSSL = ('lane_hex_20261108/runs/mtcps_nossl_c512_f0', 'lane_hex_20261108/runs/mtcps_nossl_c512_f4', 'cpsam_c512')
WIN2X = ('lane_backbone_20261108/runs/recipeA_win2x_f0_s46/raw', 'lane_backbone_20261108/runs/recipeA_win2x_f4_s46/raw', 'cpdino_b8')   # backbone lane 11-10: recipe A on the 2x technician-window crop
CMIX = ('lane_backbone_20261108/runs/cmix_f0_dinov3b16_s46/raw', 'lane_backbone_20261108/runs/cmix_f4_dinov3b16_s46/raw', 'dinov3_b16')   # backbone lane 11-11: C-Mixup DINOv3-B (raw track)
ATTNP = ('lane_backbone_20261108/runs/attnpool_f0_dinov3b16_s46/raw', 'lane_backbone_20261108/runs/attnpool_f4_dinov3b16_s46/raw', 'dinov3_b16')   # backbone lane 11-11: end-to-end attention-pool DINOv3-B (raw track)


def cnn5_hex(root, arch='cpsam_c512', epochs=range(8, 13)):
    """HEX of a CNN member trained on all five folds (root + f'{f}/runs/fold{f}/{arch}/predictions_epoch_XX.npz'), log mean over
    epochs, OOF rows only"""
    pos = {int(i): k for k, i in enumerate(idx)}; out = np.full(N, np.nan)
    for f in range(5):
        Ls = []
        for e in epochs:
            z = np.load(f'{XR}{root.format(f=f) if '{f}' in root else root + str(f)}/runs/fold{f}/{arch}/predictions_epoch_{e:02d}.npz'); p = np.full(N, np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred'][:, 2]; Ls.append(np.log(p))
        out[fold == f] = np.exp(np.mean(Ls, 0))[fold == f]
    return out


def step_cnn5_fix(cand, root='lane_hex_20261108/runs/mtcps_nossl_c512_f', c=0.2, arch='cpsam_c512'):
    """five-fold CNN member mixed into the HEX base at a fixed share c (epochs 8-12), no choice (PI 11-08: no share sweeps)"""
    A, P1 = BP._b['A'], BP._b['P1']; lbh = 0.7 * np.log(P1[:, 2]) + 0.3 * np.log(A[:, 2]); d = np.log(cnn5_hex(root, arch)) - lbh
    return {f: cand[f] * np.exp(c * d) for f in range(5)}, [f'fixed e8_12 c{c}'] * 5


def step_seedmean_fix(cand, roots, c=0.2, arch='dinov3_b16'):
    """seed audit (2026-11-11): five-fold member read with the mean over seeds of the log HEX (each seed = log mean over epochs 8-12),
    mixed into the HEX base at the fixed share c; no seed is picked"""
    A, P1 = BP._b['A'], BP._b['P1']; lbh = 0.7 * np.log(P1[:, 2]) + 0.3 * np.log(A[:, 2])
    d = np.mean([np.log(cnn5_hex(r, arch)) for r in roots], 0) - lbh
    return {f: cand[f] * np.exp(c * d) for f in range(5)}, [f'fixed e8_12 {len(roots)} seeds c{c}'] * 5


def part_hex(roots, arch='cpsam_c512', epochs=range(8, 13)):
    """HEX of a CNN member on the folds in roots ({fold: root}) only, log mean over epochs 8-12, NaN on the other folds"""
    pos = {int(i): k for k, i in enumerate(idx)}; out = np.full(N, np.nan)
    for f, d in roots.items():
        Ls = []
        for e in epochs:
            z = np.load(f'{XR}{d}/runs/fold{f}/{arch}/predictions_epoch_{e:02d}.npz'); p = np.full(N, np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred'][:, 2]; Ls.append(np.log(p))
        out[fold == f] = np.exp(np.mean(Ls, 0))[fold == f]
    return out


def step_part_mix(cand, rootsets, c=0.2, arch='cpsam_c512'):
    """seed audit (2026-11-11): a member that exists on some folds only (the second mtcnn5 seed), read as the mean log over the root
    sets (seeds) and mixed at the fixed share c into cand[f] for the folds it covers (other folds untouched), as cps_mix does"""
    A, P1 = BP._b['A'], BP._b['P1']; lbh = 0.7 * np.log(P1[:, 2]) + 0.3 * np.log(A[:, 2])
    d = np.mean([np.log(part_hex(r, arch)) for r in rootsets], 0); d = np.where(np.isfinite(d), d - lbh, 0.0); new = dict(cand)
    for f in sorted(rootsets[0]): new[f] = cand[f] * np.exp(c * d)
    return new, [f'fixed e8_12 {len(rootsets)} seeds c{c} folds {sorted(rootsets[0])}'] * 5


MT46 = lambda fs: {f: f'lane_hex_20261108/runs/mtcps_nossl_c512_f{f}' for f in fs}
MT47 = lambda fs: {f: f'lane_hex_20261108/runs/mtcps_nossl_c512_f{f}_s47' for f in fs}


CVDINO = {46: ['lane_gpu3_20261106/runs/dinov3_b16_f0_s46', 'lane_cd_20261108/arms/dinov3_b16_f1_s46', 'lane_cd_20261108/arms/dinov3_b16_f2_s46', 'lane_cd_20261108/arms/dinov3_b16_f3_s46', 'lane_gpu3_20261106/runs/dinov3_b16_f4_s46'],
          47: ['lane_gpu2_20261106/runs/dinov3_b16_f0_s47', 'lane_cd_20261108/arms/dinov3_b16_f1_s47', 'lane_cd_20261108/arms/dinov3_b16_f2_s47', 'lane_cd_20261108/arms/dinov3_b16_f3_s47', 'lane_gpu2_20261106/runs/dinov3_b16_f4_s47'],
          49: [f'lane_cv_20261108/arms/dinov3_b16_f{f}_s49' for f in range(5)]}


def step_cvdino_fix(cand, c=0.2):
    """the CV lane's shipped DINOv3 B/16 member (seeds 46, 47, 49; recipe loss; epochs 16-20; log mean over epochs and seeds, as
    in lane_cv_20261108/FREEZE_CANDIDATE.md) mixed into the HEX base at the fixed share c (2026-11-09)"""
    pos = {int(i): k for k, i in enumerate(idx)}; Ls = []
    for sd, roots in CVDINO.items():
        out = np.full(N, np.nan)
        for f in range(5):
            Lf = []
            for e in range(16, 21):
                z = np.load(f'{XR}{roots[f]}/runs/fold{f}/dinov3_b16/predictions_epoch_{e:02d}.npz'); p = np.full(N, np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred'][:, 2]; Lf.append(np.log(p))
            out[fold == f] = np.mean(Lf, 0)[fold == f]
        Ls.append(out)
    A, P1 = BP._b['A'], BP._b['P1']; lbh = 0.7 * np.log(P1[:, 2]) + 0.3 * np.log(A[:, 2]); d = np.mean(Ls, 0) - lbh
    return {f: cand[f] * np.exp(c * d) for f in range(5)}, [f'fixed e16_20 3 seeds c{c}'] * 5


def step_vith_fix(cand, path=f'{XR}lane_cd_20261108/reads/vith16_s46_member_preds.npz', c=0.2):
    """CD lane ViT-H+/16 member (folds 0 and 4; raw e8-11 + final TTA as stored), mixed into the HEX base at the fixed share c on
    folds 0 and 4 only (2026-11-10, coordinator read)"""
    z = np.load(path); hx = np.full(int(z['idx'].max()) + 1, np.nan); hx[z['idx']] = z['pred'][:, 2]; m = hx[idx]
    A, P1 = BP._b['A'], BP._b['P1']; lbh = 0.7 * np.log(P1[:, 2]) + 0.3 * np.log(A[:, 2]); d = np.where(np.isfinite(m) & (m > 0), np.log(np.where(m > 0, m, 1)) - lbh, 0.0)
    print('   vith rows with a value per fold', [int((np.isfinite(m) & (fold == f)).sum()) for f in range(5)], flush=True)
    new = dict(cand); new[0] = cand[0] * np.exp(c * d); new[4] = cand[4] * np.exp(c * d); return new, [f'fixed c{c}'] * 5


def step_cnn5(cand, root='lane_hex_20261108/runs/hexcps_nossl_c512_f', name='cnn5'):
    """five-fold CNN member HEX mixed into the HEX base at share c (all folds), plus a nested ridge stack; usual choose()"""
    A, P1 = BP._b['A'], BP._b['P1']; lbh = 0.7 * np.log(P1[:, 2]) + 0.3 * np.log(A[:, 2]); opts = [('none', cand)]
    for en, ep in (('e8_12', range(8, 13)), ('e10_12', range(10, 13)), ('e12', range(12, 13))):
        lF = np.log(cnn5_hex(root, epochs=ep)); d = lF - lbh
        for c in (0.1, 0.2, 0.3, 0.4, 0.5): opts.append((f'{en}_c{c}', {f: cand[f] * np.exp(c * d) for f in range(5)}))
        opts.append((f'{en}_stack', {f: nested_cor(cand, lF)[f] for f in range(5)}))
    return choose(opts, name)


def step_ridgeev(cand, tag='k3'):
    z = np.load(f'{L.LANE}/feats/ridge_{tag}.npz'); F = z['F'][idx].astype(float); nm = [str(q) for q in z['names']]
    opts = [('none', cand)] + [(nm[j], nested_cor(cand, lg(F[:, j]))) for j in range(4)]
    opts += [(f'all_a{a}', nested_cor(cand, np.c_[lg(F[:, :4]), F[:, 4]], alpha=a)) for a in (10.0, 100.0, 1000.0)]
    return choose(opts, 'ridge_' + tag)


def step_ridgepool(cand, tags=None):
    """ridge evidence pooled over detector graphs: mean of log h6 (s15/s25/s35/dark) and darkness across tags (nanmean)"""
    import glob as _g
    tags = tags or sorted(q.split('ridge_')[-1][:-4] for q in _g.glob(f'{L.LANE}/feats/ridge_*.npz'))
    Fs = np.stack([np.load(f'{L.LANE}/feats/ridge_{t}.npz')['F'][idx].astype(float) for t in tags])
    lF = np.c_[np.nanmean(lg(Fs[:, :, :4].reshape(len(tags), -1)).reshape(len(tags), N, 4), 0), np.nanmean(Fs[:, :, 4], 0)]
    nm = ['p_s15', 'p_s25', 'p_s35', 'p_dark']
    opts = [('none', cand)] + [(nm[j], nested_cor(cand, lF[:, j])) for j in range(4)]
    opts += [(f'pall_a{a}', nested_cor(cand, lF, alpha=a)) for a in (10.0, 100.0, 1000.0)]
    opts += [(f'cat_a{a}', nested_cor(cand, np.c_[lg(Fs[:, :, 0].T), lg(Fs[:, :, 3].T)], alpha=a)) for a in (100.0, 1000.0, 10000.0)]
    print('   ridgepool tags', tags, flush=True)
    return choose(opts, 'ridgepool')


def segfeat(path):
    z = np.load(path); F = np.full((N, z['F'].shape[1]), np.nan); pos = {int(f): r for r, f in enumerate(idx)}
    for f, v in zip(z['frames'], z['F']): F[pos[int(f)]] = v
    return F, [str(q) for q in z['names']]


def step_seg(cand, path=f'{L.LANE}/preds/seg_all.npz'):
    F, nm = segfeat(path); G = np.c_[lg(F[:, [0, 1, 2]] + 1e-3), np.log(F[:, 3] + 1), lg(F[:, 4]), lg(F[:, 5]), F[:, 6]]
    opts = [('none', cand), ('h6', nested_cor(cand, lg(F[:, 0] + 1e-3)))] + [(f'all_a{a}', nested_cor(cand, G, alpha=a)) for a in (10.0, 100.0, 1000.0)]
    if F.shape[1] >= 16:
        hist = F[:, 8:15]; hist = hist / (hist.sum(1, keepdims=True) + 1e-9); GH = np.c_[G, hist, lg(F[:, 15] + 1e-3)]
        opts += [(f'hist_a{a}', nested_cor(cand, hist, alpha=a)) for a in (10.0, 100.0, 1000.0)] + [(f'bh_a{a}', nested_cor(cand, GH, alpha=a)) for a in (10.0, 100.0, 1000.0)]
    return choose(opts, 'seg')


def nested_cor_m(cand, F, m, alpha=10.0):
    """nested_cor fit and applied only on rows m (segmented rows); other rows keep cand"""
    F = np.asarray(F, float).reshape(N, -1); out = {}
    for f in range(5):
        lb = lg(cand[f]); X = np.where(m[:, None], np.c_[F, lb], 0.0); tr = (fold != f) & good & ok & m
        t = lg(np.where(ok, y[:, 2], 1)) - lb; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Z = (X - mu) / sd; A = Z[tr]
        w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (t[tr] - t[tr].mean())); out[f] = np.where(m, cand[f] * np.exp(Z @ w + t[tr].mean()), cand[f])
    return out


def seg_usable(F, nmin=50): return np.isfinite(F[:, :15]).all(1) & (np.nan_to_num(F[:, 3]) >= nmin)


def seg_avg(paths, nmin=50):
    """mean of the segmentation features over the runs usable on each frame (NaN where none is usable)"""
    Fs = [segfeat(p)[0] for p in paths]; U = np.array([seg_usable(F, nmin) for F in Fs]); S_ = np.zeros_like(Fs[0]); W_ = np.zeros_like(Fs[0])
    for F, u in zip(Fs, U):
        v = u[:, None] & np.isfinite(F); S_ += np.where(v, F, 0.0); W_ += v
    print('   seg runs usable', U.sum(1).tolist(), 'both', int(U.all(0).sum()), 'any', int(U.any(0).sum()), flush=True)
    return np.where(W_ > 0, S_ / np.maximum(W_, 1), np.nan)


def step_segm(cand, path=f'{L.LANE}/preds/seg_all.npz', nmin=50, F=None, name='segm'):
    """seg v2 (2026-11-08): correction fit and applied only where >= nmin interior cells were segmented (no imputation)"""
    F = segfeat(path)[0] if F is None else F; m = seg_usable(F, nmin)
    F = F.copy(); F[~np.isfinite(F[:, 15]), 15] = np.nanmedian(F[m, 15])
    G = np.c_[lg(F[:, [0, 1, 2]] + 1e-3), np.log(F[:, 3] + 1), lg(F[:, 4]), lg(F[:, 5]), F[:, 6]]
    hist = F[:, 8:15] / (F[:, 8:15].sum(1, keepdims=True) + 1e-9); GH = np.c_[G, hist, lg(F[:, 15] + 1e-3)]
    print(f'   {name} usable rows', int(m.sum()), 'per fold', [int((m & (fold == f)).sum()) for f in range(5)], flush=True)
    opts = [('none', cand)] + [(f'h6_a{a}', nested_cor_m(cand, lg(F[:, 0] + 1e-3), m, alpha=a)) for a in (10.0, 1000.0)]
    opts += [(f'{nmn}_a{a}', nested_cor_m(cand, X, m, alpha=a)) for nmn, X in (('all', G), ('hist', hist), ('bh', GH)) for a in (100.0, 1000.0, 10000.0)]
    return choose(opts, name)


def step_segb(cand, path=f'{L.LANE}/preds/seg_all.npz'):
    """seg v3 (2026-11-08): segmentation h6 as a direct HEX measurement, calibrated (log-linear, train folds, >=200 cells) and
    blended in log space with weight n/(n+k), n = interior cells; k picked per outer fold by choose()"""
    F, nm = segfeat(path); n = np.nan_to_num(F[:, 3]); m = np.isfinite(F[:, 0]) & (n >= 20); h = lg(np.where(m, F[:, 0], 0.5) + 1e-3)
    opts = [('none', cand)]
    for k in (100, 300, 1000, 3000, 10000):
        out = {}
        for f in range(5):
            tr = (fold != f) & good & ok & m & (n >= 200); b, a = np.polyfit(h[tr], lg(y[tr, 2]), 1)
            w = np.where(m, n / (n + k), 0.0); out[f] = np.exp((1 - w) * lg(cand[f]) + w * (a + b * h))
        opts.append((f'blend_k{k}', out))
    return choose(opts, 'segb')


def step_segg(cand, path=f'{L.LANE}/preds/seg_all.npz', nmin=50):
    """seg v4 (2026-11-08): gradient-boosted trees on the log residual (segmented rows only, fit on folds != f), shrunk by c;
    options c picked per outer fold by choose()"""
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    F, nm = segfeat(path); m = np.isfinite(F[:, :15]).all(1) & (np.nan_to_num(F[:, 3]) >= nmin)
    F = F.copy(); F[~np.isfinite(F[:, 15]), 15] = np.nanmedian(F[m, 15])
    hist = F[:, 8:15] / (F[:, 8:15].sum(1, keepdims=True) + 1e-9)
    X0 = np.c_[lg(F[:, [0, 1, 2]] + 1e-3), np.log(F[:, 3] + 1), lg(F[:, 4]), lg(F[:, 5]), F[:, 6], hist, lg(F[:, 15] + 1e-3)]
    R = {}
    for f in range(5):
        lb = lg(cand[f]); X = np.where(m[:, None], np.c_[X0, lb], 0.0); tr = (fold != f) & good & ok & m; t = lg(np.where(ok, y[:, 2], 1)) - lb
        g = HGB(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=0).fit(X[tr], t[tr] - t[tr].mean())
        R[f] = np.where(m, g.predict(X) + t[tr].mean(), 0.0)
    opts = [('none', cand)] + [(f'gbm_c{c}', {f: cand[f] * np.exp(c * R[f]) for f in range(5)}) for c in (0.25, 0.5, 0.75, 1.0)]
    return choose(opts, 'segg')


def step_segv(cand, path=f'{L.LANE}/preds/seg_vor.npz', nmin=50):
    """seg v5 (2026-11-08): centroid-Delaunay neighbour counts from the same masks (source/seg_vor.py), ridge fit and applied
    only on frames with >= nmin counted cells; alpha picked per outer fold by choose()"""
    V, nm = segfeat(path); m = np.isfinite(V).all(1) & (np.nan_to_num(V[:, 3]) >= nmin)
    X = np.c_[lg(V[:, [0, 1, 2]] + 1e-3), np.log(V[:, 3] + 1), V[:, 4], V[:, 5], lg(V[:, 6]), V[:, 7]]
    print('   segv usable rows', int(m.sum()), flush=True)
    opts = [('none', cand), ('h6_a1000', nested_cor_m(cand, lg(V[:, 0] + 1e-3), m, alpha=1000.0))] + [(f'all_a{a}', nested_cor_m(cand, X, m, alpha=a)) for a in (100.0, 1000.0, 10000.0)]
    return choose(opts, 'segv')


STEPS = {'ap1': step_ap1, 'h6w': step_h6w, 'graphens': step_graphens, 'gscale': step_gscale, 'quad': step_quad,
         'fuse16': lambda c: step_feats(c, f'{L.LANE}/feats/fuse_all16.npz'),
         'prior': lambda c: step_feats(c, f'{L.LANE}/feats/prior_pre3.npz'),
         'imgstats': lambda c: step_ridge(c, imgstats(), 'imgstats'),
         'cdcv': lambda c: step_ridge(c, np.c_[lg(L.REF[:, 0]), lg(L.REF[:, 1])], 'cdcv'),
         'fuseall': lambda c: step_group(c, featgroup(f'{L.LANE}/feats/fuse_all16.npz'), 'fuseall'),
         'legib': lambda c: step_group(c, featgroup(f'{L.LANE}/feats/fuse_all16.npz', lambda k: k.startswith('det') or k == 'spacing' or k.endswith(('_n', '_nb', '_nc'))), 'legib'),
         'priorall': lambda c: step_group(c, featgroup(f'{L.LANE}/feats/prior_pre3.npz'), 'priorall'),
         'imgmape': lambda c: choose([('none', c)] + [(f'l{lam}', mape_fit(c, {f: np.c_[imgstats(), lg(c[f])] for f in range(5)}, lam, 'imgmape')) for lam in (1e-4, 1e-3, 1e-2)], 'imgmape'),
         'imgvig': lambda c: step_ridge(c, imgrich()[:, [5, 6, 7, 8, 9, 12, 17, 18, 19, 21]], 'imgvig'),
         'family': step_family, 'seg': step_seg, 'ridgeev': step_ridgeev, 'ridgeev_k5': lambda c: step_ridgeev(c, 'k5'), 'ridgeev_fall16': lambda c: step_ridgeev(c, 'fall16'),
         'legwin': lambda c: step_group(c, featgroup('/workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep/legible_window.npz'), 'legwin'),
         'mapelin': step_mapelin, 'cpsam': step_cpsam, 'cpsam_fix': step_cps_fix, 'hexcps': lambda c: step_cpsam(c, HCPS, 'hexcps'), 'hexcps_fix': lambda c: step_cps_fix(c, HCPS), 'nossl': lambda c: step_cpsam(c, NOSSL, 'nossl'), 'nossl_fix': lambda c: step_cps_fix(c, NOSSL), 'mtnossl': lambda c: step_cpsam(c, MTNOSSL, 'mtnossl'), 'mtnossl_fix': lambda c: step_cps_fix(c, MTNOSSL), 'mtcnn5': lambda c: step_cnn5(c, 'lane_hex_20261108/runs/mtcps_nossl_c512_f', 'mtcnn5'), 'mtcnn5_fix': step_cnn5_fix, 'cvdino_fix': step_cvdino_fix, 'win2x_fix': lambda c: step_cps_fix(c, WIN2X), 'cmix_fix': lambda c: step_cps_fix(c, CMIX), 'attnp_fix': lambda c: step_cps_fix(c, ATTNP), 'attnp5_fix': lambda c: step_cnn5_fix(c, 'lane_backbone_20261108/runs/attnpool_f{f}_dinov3b16_s46/raw', arch='dinov3_b16'), 'vith_fix': step_vith_fix, 'cvw3_fix': lambda c: step_cnn5_fix(c, 'lane_cv_20261108/arms/dinov3_cvw3_f{f}_s46', arch='dinov3_b16'), 'cvw3s47_fix': lambda c: step_cnn5_fix(c, 'lane_hex_20261108/runs/cvw3_f{f}_s47', arch='dinov3_b16'), 'cvw3m_fix': lambda c: step_seedmean_fix(c, ['lane_cv_20261108/arms/dinov3_cvw3_f{f}_s46', 'lane_hex_20261108/runs/cvw3_f{f}_s47']), 'mt46_04': lambda c: step_part_mix(c, [MT46((0, 4))]), 'mt47_04': lambda c: step_part_mix(c, [MT47((0, 4))]), 'mtm_04': lambda c: step_part_mix(c, [MT46((0, 4)), MT47((0, 4))]), 'mt46_4f': lambda c: step_part_mix(c, [MT46((0, 1, 2, 4))]), 'mt47_4f': lambda c: step_part_mix(c, [MT47((0, 1, 2, 4))]), 'mtm_4f': lambda c: step_part_mix(c, [MT46((0, 1, 2, 4)), MT47((0, 1, 2, 4))]), 'cnn5': step_cnn5, 'segm': step_segm, 'segb': step_segb, 'segg': step_segg, 'segv': step_segv, 'segm2': lambda c: step_segm(c, F=seg_avg([f'{L.LANE}/preds/seg_all.npz', f'{L.LANE}/preds/seg_rot_all.npz']), name='segm2'), 'segr': lambda c: step_segm(c, f'{L.LANE}/preds/seg_rot_all.npz', name='segr'), 'segvig': lambda c: step_segm(c, f'{L.LANE}/preds/seg_vig_all.npz', name='segvig'), 'ridgepool': step_ridgepool, 'imgrich': step_imgrich, 'uncridge': step_uncridge,
         'cdcvimg': lambda c: step_ridge(c, np.c_[lg(L.REF[:, 0]), lg(L.REF[:, 1]), imgstats()], 'cdcvimg')}

if __name__ == '__main__':
    cand = {f: L.HB[f].copy() for f in range(5)}; log = []
    r0 = read('best pipeline', cand, cand)
    for s in sys.argv[1:]:
        new, picks = (step_gnn(cand, s[4:]) if s.startswith('gnn:') else step_ridgeev(cand, s[8:]) if s.startswith('ridgeev:') else STEPS[s](cand)); o = read(s, new, cand); o['picks'] = picks
        force = os.environ.get('CHAIN_FORCE') == '1'; keep = o['d_prev_f04'] < 0 or force; o['kept'] = bool(keep); o['forced'] = force   # CHAIN_FORCE=1: keep every step (seed audit: chain with vs without a step)
        print('   picks', picks, ('kept (forced)' if force else 'kept') if keep else 'dropped', flush=True); log.append(o)
        if keep: cand = new
    P = final(cand); np.savez_compressed(f'{L.LANE}/preds/' + os.environ.get('CHAIN_OUT', 'chain_hex') + '.npz', **{f'c{f}': cand[f] for f in range(5)}, final_hex=P[:, 2], idx=idx, fold=fold, steps=json.dumps(log))
    if os.environ.get('CHAIN_NORESULTS') == '1': sys.exit(0)   # leave-one-out reads (2026-11-11) log to their own files
    with open(f'{L.LANE}/RESULTS.tsv', 'a') as fh:
        for o in log: fh.write(f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime())}Z\tchain step {o['step']}\t0,4|all5\t{o['f04']}|{o['all5']}\tvs prev {o['d_prev_f04']:+.4f}|{o['d_prev_all5']:+.4f}; vs best {o['d_best_f04']:+.4f}|{o['d_best_all5']:+.4f}\t{('kept (forced)' if o.get('forced') else 'kept') if o['kept'] else 'dropped'}; nested picks {o['picks']}\n")
