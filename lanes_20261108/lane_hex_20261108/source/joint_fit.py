"""Joint nested fit of all HEX corrections at once (HEX lane, 2026-11-11, PI via the coordinator), instead of a sequential chain.
One ridge on t = log y - log base with every pool signal as a column (member log predictions, graph-net corrections, image,
prior, ridge-evidence and segmentation features, h6). For outer fold f: alpha is chosen by an inner leave-one-fold-out CV on the
other four folds (alpha grid fixed in advance), the ridge is fit on those four folds, and fold f is predicted; then the usual
per-fold stretch (chain.final). Nothing is chosen on the rows it is scored on. Bases: 'hb' (best pipeline) and 'ap1_nested' (ap1 with its per-fold nested weight).
usage: joint_fit.py OUT.json"""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
sys.path.insert(0, LANE + '/source')
import chain as C

N, fold, y, ok, good, idx = C.N, C.fold, C.y[:, 2], C.ok, C.good, C.idx; lg = C.lg; BP = C.BP
own = lambda d: np.select([fold == f for f in range(5)], [d[f] for f in range(5)])
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
INNER = {}   # inner-CV error of the picked alpha per base and outer fold (for the nested base choice)


def columns():
    A, P1 = BP._b['A'][:, 2], BP._b['P1'][:, 2]; have = np.asarray(BP._have, bool); h6 = np.where(have, -0.2084 + 0.5989 * np.nan_to_num(BP.f6), 0.0)
    G = {}
    G['blend'] = np.c_[np.log(P1), np.log(A), h6, have.astype(float)]
    G['best_graph'] = np.c_[own({f: BP.BEST[f][:, 2] for f in range(5)}), own(C.OLDG)]
    G['graphens'] = own(C.gload(['d64_l5_noctx_s10', 'd64_l5_pre47_s10', 'd64_l5_pre48_s10']))[:, None]
    z = np.load(f'{LANE}/preds/hex_pre4647_s5.npz'); G['gnn4647'] = own({f: z[f'f{f}'][:, 2] for f in range(5)})[:, None]
    z = np.load(f'{LANE}/preds/hex_pre46_s5.npz'); G['gnn46'] = own({f: z[f'f{f}'][:, 2] for f in range(5)})[:, None]
    G['imgstats'] = C.imgstats()
    zp = np.load(f'{LANE}/feats/prior_pre3.npz'); G['prior'] = lg(zp['F'][idx].astype(float)[:, [str(q) for q in zp['names']].index('cm1.0_h6c')])[:, None]
    zk = np.load(f'{LANE}/feats/ridge_k3.npz'); Fk = zk['F'][idx].astype(float); G['ridge_k3'] = np.c_[lg(Fk[:, :4]), Fk[:, 4]]
    zf = np.load(f'{LANE}/feats/ridge_fall16.npz'); G['ridge_fall16'] = lg(zf['F'][idx].astype(float)[:, [str(q) for q in zf['names']].index('h6_s15')])[:, None]
    F = C.segfeat(f'{LANE}/preds/seg_all.npz')[0]; m = C.seg_usable(F, 50); F = F.copy(); F[~np.isfinite(F[:, 15]), 15] = np.nanmedian(F[m, 15])
    S = np.c_[lg(F[:, [0, 1, 2]] + 1e-3), np.log(F[:, 3] + 1), lg(F[:, 4]), lg(F[:, 5]), F[:, 6], F[:, 8:15] / (F[:, 8:15].sum(1, keepdims=True) + 1e-9), lg(F[:, 15] + 1e-3)]
    G['segm'] = np.c_[np.where(m[:, None], np.nan_to_num(S), 0.0), m.astype(float)]
    G['cvw3_mean'] = np.mean([np.log(C.cnn5_hex(r, 'dinov3_b16')) for r in ('lane_cv_20261108/arms/dinov3_cvw3_f{f}_s46', 'lane_hex_20261108/runs/cvw3_f{f}_s47')], 0)[:, None]
    G['mtcnn5'] = np.log(C.cnn5_hex('lane_hex_20261108/runs/mtcps_nossl_c512_f'))[:, None]
    G['attnp5'] = np.log(C.cnn5_hex('lane_backbone_20261108/runs/attnpool_f{f}_dinov3b16_s46/raw', 'dinov3_b16'))[:, None]
    return G


def ridge_fit(X, t, tr, alpha):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Z = (X - mu) / sd; A = Z[tr]; tm = t[tr].mean()
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (t[tr] - tm)); return Z @ w + tm


def run(base, X):
    t_all = lg(np.where(ok, y, 1)); cand = {}; picks = []
    for f in range(5):
        lb = np.log(base[f]); Xf = np.c_[X, lb]; Xf = np.where(np.isfinite(Xf), Xf, np.nan); med = np.nanmedian(Xf[(fold != f) & good], 0); Xf = np.where(np.isfinite(Xf), Xf, med)
        t = t_all - lb; inner = []
        for a in ALPHAS:
            e = []
            for g in range(5):
                if g == f: continue
                tr = (fold != f) & (fold != g) & good & ok; te = (fold == g) & ok; p = ridge_fit(Xf, t, tr, a); e.append(np.mean(np.abs(np.exp(lb[te] + p[te]) - y[te]) / y[te]))
            inner.append(np.mean(e))
        a = ALPHAS[int(np.argmin(inner))]; picks.append(a); INNER.setdefault(id(base), {})[f] = float(min(inner)); cand[f] = base[f] * np.exp(ridge_fit(Xf, t, (fold != f) & good & ok, a))
    P = C.final(cand)[:, 2]; E = np.abs(P - y) / np.where(ok, y, 1)
    return dict(all5=round(float(E[ok].mean() * 100), 5), f04=round(float(E[ok & np.isin(fold, [0, 4])].mean() * 100), 5), folds=[round(float(E[ok & (fold == f)].mean() * 100), 4) for f in range(5)], alpha_picks=picks), cand


if __name__ == '__main__':
    G = columns(); drop = [q for q in os.environ.get('JOINT_DROP', 'gnn46').split(',') if q]; names = [k for k in G if k not in drop]
    X = np.concatenate([G[k] for k in names], 1); print('columns', {k: G[k].shape[1] for k in names}, 'total', X.shape[1], 'dropped', drop, flush=True)
    bases = {'hb': {f: C.L.HB[f].copy() for f in range(5)}, 'ap1_nested': C.STEPS['ap1']({f: C.L.HB[f].copy() for f in range(5)})[0]}   # ap1 with its own per-fold nested weight (0.7; fold 4 0.8)
    out = {}
    for bn, base in bases.items():
        r, cand = run(base, X); out[bn] = r; print(bn, json.dumps(r), flush=True)
        # coefficient stability: standardized ridge weights per outer fold (refit with the picked alpha), sign agreement across folds
        t_all = lg(np.where(ok, y, 1)); W = []
        for f in range(5):
            lb = np.log(base[f]); Xf = np.c_[X, lb]; med = np.nanmedian(np.where(np.isfinite(Xf), Xf, np.nan)[(fold != f) & good], 0); Xf = np.where(np.isfinite(Xf), Xf, med)
            tr = (fold != f) & good & ok; mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-9; A_ = ((Xf - mu) / sd)[tr]; tt = (t_all - lb)[tr]
            W.append(np.linalg.solve(A_.T @ A_ + r['alpha_picks'][f] * np.eye(A_.shape[1]), A_.T @ (tt - tt.mean())))
        W = np.array(W); cn = [f'{k}[{j}]' for k in names for j in range(G[k].shape[1])] + ['log base']
        out[bn]['coef_sign_agree_all5'] = int((np.abs(np.sign(W).sum(0)) == 5).sum()); out[bn]['n_coef'] = int(W.shape[1])
        out[bn]['top_coef_mean_sd'] = sorted([(cn[j], round(float(W[:, j].mean()), 4), round(float(W[:, j].std()), 4)) for j in range(W.shape[1])], key=lambda q: -abs(q[1]))[:12]
        np.savez_compressed(sys.argv[1].replace('.json', f'_{bn}.npz'), **{f'c{f}': cand[f] for f in range(5)}, final_hex=C.final(cand)[:, 2], idx=idx, fold=fold)
    # nested base choice: for outer fold f, the base whose inner-CV error (other four folds) is lower
    cands = {bn: dict(np.load(sys.argv[1].replace('.json', f'_{bn}.npz'))) for bn in bases}; ib = {bn: INNER[id(b)] for bn, b in bases.items()}
    pick = {f: min(bases, key=lambda bn: ib[bn][f]) for f in range(5)}; cand = {f: cands[pick[f]][f'c{f}'] for f in range(5)}
    P = C.final(cand)[:, 2]; E = np.abs(P - y) / np.where(ok, y, 1)
    out['nested_base'] = dict(picks=pick, inner={bn: [round(ib[bn][f] * 100, 4) for f in range(5)] for bn in bases}, all5=round(float(E[ok].mean() * 100), 5), f04=round(float(E[ok & np.isin(fold, [0, 4])].mean() * 100), 5), folds=[round(float(E[ok & (fold == f)].mean() * 100), 4) for f in range(5)])
    print('nested_base', json.dumps(out['nested_base']), flush=True)
    np.savez_compressed(sys.argv[1].replace('.json', '_nestedbase.npz'), **{f'c{f}': cand[f] for f in range(5)}, final_hex=P, idx=idx, fold=fold)
    json.dump(dict(columns={k: G[k].shape[1] for k in names}, results=out, alphas=ALPHAS), open(sys.argv[1], 'w'), indent=1)
