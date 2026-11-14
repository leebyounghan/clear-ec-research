"""Deploy parameters and test-time apply for the joint nested HEX ridge (HEX lane, 2026-11-11; candidate (b) of the combination
search). Writes preds/deploy_params_joint.json with, per outer fold, the nested fit (alpha picked by inner leave-one-fold-out,
feature medians, mean/sd, weights, intercept, per-fold stretch) and the deploy fit on all labelled rows (alpha by five-fold
leave-one-fold-out on the OOF inputs, as chain_deploy.py does for the chain; stretch on all rows). The nested fits use the
chain convention (fold f's own BEST in the base on every row), so the nested apply reproduces the reported OOF exactly. apply_joint(params, stretch, I) computes HEX per image from the hex_apply inputs.
usage: joint_deploy.py build | joint_deploy.py parity"""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
sys.path.insert(0, LANE + '/source')
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
COLS = ['blend(4)', 'best_graph(2)', 'graphens(1)', 'gnn46(1)', 'imgstats(4)', 'prior(1)', 'ridge_k3(5)', 'ridge_fall16(1)', 'segm(16)', 'cvw3_mean(1)', 'mtcnn5(1)', 'log base(1)']   # the 39-column variant adds attnp5(1) before log base


def lg(v): v = np.asarray(v, float); return np.where(v > 0, np.log(np.where(v > 0, v, 1)), np.nan)


def design(I, w_ap1):
    """columns in joint_fit order (deployable set) plus log base; I = hex_apply inputs (per image)"""
    A, P1 = np.asarray(I['A'], float), np.asarray(I['P1'], float); have = np.asarray(I['have'], bool); f6 = np.nan_to_num(np.asarray(I['f6'], float))
    h6 = np.where(have, -0.2084 + 0.5989 * f6, 0.0)
    bh = np.exp(w_ap1 * np.log(P1) + (1 - w_ap1) * np.log(A)); hh = bh.copy(); hh[have] = np.exp(0.72 * np.log(bh[have]) + 0.28 * h6[have])
    base = hh * np.exp(np.asarray(I['best'], float))
    img = np.asarray(I['img'], float); rk = np.asarray(I['ridge_k3'], float); F = np.asarray(I['seg'], float).copy()
    return base, F, np.c_[np.log(P1), np.log(A), h6, have.astype(float), I['best'], I['graph'], I['g_s10x3'], I['g_hex46'],
                          np.log(img[:, 0] + 1), np.log(img[:, 1] + 1), img[:, 2], np.log(img[:, 3] + 1), lg(I['prior_h6c']), lg(rk[:, :4]), rk[:, 4],
                          lg(I['ridge_fall16_h6_s15'])]


def seg_cols(F, fill15):
    m = np.isfinite(F[:, :15]).all(1) & (np.nan_to_num(F[:, 3]) >= 50); F = F.copy(); F[~np.isfinite(F[:, 15]), 15] = fill15
    S = np.c_[lg(F[:, [0, 1, 2]] + 1e-3), np.log(F[:, 3] + 1), lg(F[:, 4]), lg(F[:, 5]), F[:, 6], F[:, 8:15] / (F[:, 8:15].sum(1, keepdims=True) + 1e-9), lg(F[:, 15] + 1e-3)]
    return np.c_[np.where(m[:, None], np.nan_to_num(S), 0.0), m.astype(float)]


ATTNP = os.environ.get('JOINT_ATTNP') == '1'   # 39-column variant: the backbone attention-pool DINOv3-B column (raw track, mean log over epochs 8-12)
ALL_BLOCKS = ['blend', 'best_graph', 'graphens', 'gnn46', 'imgstats', 'prior', 'ridge_k3', 'ridge_fall16', 'segm', 'cvw3_mean', 'mtcnn5', 'attnp5']
# packaging tiers (D-PKG-002, 2026-11-12): JOINT_DROP = comma list of blocks left out of the 39-column ridge, JOINT_TAG = file suffix
DROP = [q for q in os.environ.get('JOINT_DROP', '').split(',') if q]; assert all(q in ALL_BLOCKS for q in DROP), DROP
BLOCKS = [b for b in ALL_BLOCKS if b not in DROP] if (DROP or os.environ.get('JOINT_TAG')) else (ALL_BLOCKS if ATTNP else ALL_BLOCKS[:-1])
SUF = f"_{os.environ['JOINT_TAG']}" if os.environ.get('JOINT_TAG') else ('_attnp' if ATTNP else '')


def full_X(I, w_ap1, fill15, attnp=None, blocks=None):
    """design matrix of the named blocks (in ALL_BLOCKS order) plus log base; only the inputs of the listed blocks are read"""
    if blocks is None: blocks = BLOCKS if attnp is None else (ALL_BLOCKS if attnp else ALL_BLOCKS[:-1])
    A, P1 = np.asarray(I['A'], float), np.asarray(I['P1'], float); have = np.asarray(I['have'], bool); f6 = np.nan_to_num(np.asarray(I['f6'], float))
    h6 = np.where(have, -0.2084 + 0.5989 * f6, 0.0)
    bh = np.exp(w_ap1 * np.log(P1) + (1 - w_ap1) * np.log(A)); hh = bh.copy(); hh[have] = np.exp(0.72 * np.log(bh[have]) + 0.28 * h6[have])
    base = hh * np.exp(np.asarray(I['best'], float))
    def part(b):
        if b == 'blend': return np.c_[np.log(P1), np.log(A), h6, have.astype(float)]
        if b == 'best_graph': return np.c_[I['best'], I['graph']]
        if b == 'graphens': return np.asarray(I['g_s10x3'], float)[:, None]
        if b == 'gnn46': return np.asarray(I['g_hex46'], float)[:, None]
        if b == 'imgstats': img = np.asarray(I['img'], float); return np.c_[np.log(img[:, 0] + 1), np.log(img[:, 1] + 1), img[:, 2], np.log(img[:, 3] + 1)]
        if b == 'prior': return lg(I['prior_h6c'])[:, None]
        if b == 'ridge_k3': rk = np.asarray(I['ridge_k3'], float); return np.c_[lg(rk[:, :4]), rk[:, 4]]
        if b == 'ridge_fall16': return lg(I['ridge_fall16_h6_s15'])[:, None]
        if b == 'segm': return seg_cols(np.asarray(I['seg'], float).copy(), fill15)
        if b == 'cvw3_mean': return np.log(np.asarray(I['dino_cvw3'], float))[:, None]
        if b == 'mtcnn5': return np.log(np.asarray(I['cps5'], float))[:, None]
        if b == 'attnp5': return np.log(np.asarray(I['attnp5'], float))[:, None]
        raise ValueError(b)
    X = np.c_[(*[part(b) for b in ALL_BLOCKS if b in blocks], np.log(base))]
    return base, X


def apply_joint(p, stretch, I):
    assert p.get('base_kind', 'ap1') == 'ap1', p.get('base_kind')
    blocks = p.get('blocks') or (ALL_BLOCKS if len(p['w']) == 39 else ALL_BLOCKS[:-1])   # files from before 11-12 carry no block list (38 or 39 columns)
    base, X = full_X(I, p['w_ap1'], p['h6_strict_fill'], blocks=blocks); X = np.where(np.isfinite(X), X, np.asarray(p['median']))
    lc = np.log(base) + ((X - np.asarray(p['mean'])) / np.asarray(p['sd'])) @ np.asarray(p['w']) + p['intercept']
    return stretch['c'] * np.exp(stretch['g'] * (lc - stretch['mu']) + stretch['mu'])


def fit(X, t, tr, alpha):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; A = ((X - mu) / sd)[tr]; tm = t[tr].mean()
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (t[tr] - tm)); return dict(mean=mu.tolist(), sd=sd.tolist(), w=w.tolist(), intercept=float(tm))


def outer(f, I, C, Xraw, t_all, wf, BEST_in, fill15):
    """the joint ridge procedure for outer fold f (f = None: no outer fold, i.e. the deploy fit). Folds available = all but f; base on
    every row = the base model fitted on those folds (fold f's own BEST; for f = None the all-rows BEST in-sample, CV lane
    cv_deploy_insample_members G + RID); base kind (ap1 vs best pipeline) and alpha by leave-one-fold-out over the available folds;
    ridge fit on the available folds; stretch on them."""
    fold, ok, good, y, BP = C.fold, C.ok, C.good, C.y[:, 2], C.BP
    avail = [g for g in range(5) if g != f]; best = BP.BEST[f][:, 2] if f is not None else BEST_in; w = wf[f] if f is not None else 0.7
    bases = {'ap1': C.base_hex(w) * np.exp(best), 'hb': BP.H[:, 2] * np.exp(best)}; res = {}
    for bk, base in bases.items():
        X = Xraw.copy(); X[:, -1] = np.log(base); tra = np.isin(fold, avail) & good
        med = np.nanmedian(np.where(np.isfinite(X), X, np.nan)[tra], 0); X = np.where(np.isfinite(X), X, med); t = t_all - np.log(base); inner = []
        for a in ALPHAS:
            e = []
            for g in avail:
                tr = tra & (fold != g) & ok; te = (fold == g) & ok; q = fit(X, t, tr, a)
                p = np.log(base) + ((X - q['mean']) / q['sd']) @ np.asarray(q['w']) + q['intercept']; e.append(np.mean(np.abs(np.exp(p[te]) - y[te]) / y[te]))
            inner.append(float(np.mean(e)))
        a = ALPHAS[int(np.argmin(inner))]; q = fit(X, t, tra & ok, a); q.update(alpha=a, median=med.tolist(), w_ap1=w, base_kind=bk, inner_error=min(inner), h6_strict_fill=fill15, blocks=list(BLOCKS))
        res[bk] = (q, base * np.exp(((X - q['mean']) / q['sd']) @ np.asarray(q['w']) + q['intercept']))
    bk = min(res, key=lambda k: res[k][0]['inner_error']); q, cand = res[bk]
    Cf = BP.H.copy(); Cf[:, 2] = cand; s_ = C.NS.fit_stretch(Cf, C.y, np.isin(fold, avail) & good)[2]
    return q, dict(g=float(s_[0]), c=float(s_[1]), mu=float(s_[2])), cand, {k: v[0]['inner_error'] for k, v in res.items()}


if __name__ == '__main__':
    import hex_apply as HA
    HA.SEEDMEAN = True; I, C = HA.oof_inputs(); fold, ok, good, y = C.fold, C.ok, C.good, C.y[:, 2]; BP = C.BP
    if 'attnp5' in BLOCKS: I['attnp5'] = C.cnn5_hex('lane_backbone_20261108/runs/attnpool_f{f}_dinov3b16_s46/raw', 'dinov3_b16')
    F0 = np.asarray(I['seg'], float); m0 = np.isfinite(F0[:, :15]).all(1) & (np.nan_to_num(F0[:, 3]) >= 50); fill15 = float(np.nanmedian(F0[m0, 15]))
    wf = [float(q[1:]) for q in C.STEPS['ap1']({f: C.L.HB[f].copy() for f in range(5)})[1]]; t_all = lg(np.where(ok, y, 1))
    zi = np.load('/workspace/clear-ec-research/experiments/lane_cv_20261108/deploy/cv_deploy_insample_members.npz'); BEST_in = zi['G'][:, 2] + zi['RID'][:, 2]
    _, Xraw = full_X(I, 0.7, fill15, blocks=BLOCKS)   # columns are the OOF inputs on every row; the last column (log base) is set inside outer()
    if sys.argv[1] == 'build':
        nested, nst, cand, inner = [], [], {}, []
        for f in range(5):
            q, st, cd, ie = outer(f, I, C, Xraw, t_all, wf, BEST_in, fill15); assert q['base_kind'] == 'ap1', (f, q['base_kind']); nested.append(q); nst.append(st); cand[f] = cd; inner.append(ie)
        dep, st, _, ie_dep = outer(None, I, C, Xraw, t_all, wf, BEST_in, fill15); assert dep['base_kind'] == 'ap1', dep['base_kind']
        P = C.final(cand)[:, 2]; E = np.abs(P - y) / np.where(ok, y, 1)
        res = dict(kind='joint nested ridge on log HEX (combination search (b), 2026-11-11)', blocks=list(BLOCKS), dropped=DROP, columns=list(BLOCKS) + ['log base'], alphas=ALPHAS,
                   convention='chain convention everywhere: the base on the fit rows is the base model fitted on those rows (nested: fold f\'s own BEST; deploy: all-rows BEST in-sample = CV lane G + RID); deploy = outer(None), the nested procedure with no outer fold',
                   nested=nested, nested_stretch=nst, deploy=dep, stretch=st, base_inner_errors=dict(nested=inner, deploy=ie_dep),
                   oof_hex_five_folds=round(float(E[ok].mean() * 100), 5), oof_hex_folds04=round(float(E[ok & np.isin(fold, [0, 4])].mean() * 100), 5))
        json.dump(res, open(f'{LANE}/preds/deploy_params_joint{SUF}.json', 'w'), indent=1)
        np.savez_compressed(f'{LANE}/preds/chain_joint{SUF}.npz', **{f'c{f}': cand[f] for f in range(5)}, final_hex=P, idx=C.idx, fold=fold)
        print(json.dumps({k: res[k] for k in ('oof_hex_five_folds', 'oof_hex_folds04', 'stretch')}), 'nested alphas', [q['alpha'] for q in nested], 'deploy alpha', dep['alpha'], 'deploy base errors', ie_dep, flush=True)
    else:
        P = json.load(open(f'{LANE}/preds/deploy_params_joint{SUF}.json')); ref = np.load(f'{LANE}/preds/chain_joint{SUF}.npz')['final_hex']; Hn = np.full(len(fold), np.nan)
        for f in range(5):
            r = fold == f; Hn[r] = apply_joint(P['nested'][f], P['nested_stretch'][f], {k: np.asarray(v)[r] for k, v in I.items()})
        q, st, _, _ = outer(None, I, C, Xraw, t_all, wf, BEST_in, fill15)   # the nested code with a dummy outer fold removed
        dmax = max(float(np.max(np.abs(np.asarray(q[k]) - np.asarray(P['deploy'][k])))) for k in ('mean', 'sd', 'w', 'median')); dmax = max(dmax, abs(q['intercept'] - P['deploy']['intercept']), max(abs(st[k] - P['stretch'][k]) for k in st))
        mape = lambda H, r: round(float((np.abs(H - y) / y * 100)[ok & r].mean()), 5)
        print(json.dumps(dict(nested_five_folds=mape(Hn, np.ones(len(fold), bool)), nested_f04=mape(Hn, np.isin(fold, [0, 4])), max_abs_diff_vs_oof=float(np.nanmax(np.abs(Hn - ref))),
                              outer_none_vs_deploy_params_max_diff=dmax, deploy_alpha=P['deploy']['alpha'], same_alpha=q['alpha'] == P['deploy']['alpha'])), flush=True)
