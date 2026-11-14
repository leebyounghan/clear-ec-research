"""CV lane deploy fit (2026-11-10). Every stored parameter the CV column needs at test time, fit on the labelled rows with the OOF recipes:
  - recipe A two-seed scalars (hybrid_oof_20261028.fit_scalars: weighted median of y/p with weights p/y, rows with y > 0) on all rows;
    the Phase-1 scalars are the fixed C.SCALE.
  - simulation-embedding ridge = BEST - GRAPH (CV and HEX columns; the HEX lane reads the HEX one). Strength: leave-one-fold-out over the
    five folds, the inner loop of nested_joint_20261105.py one level up (base of fold q = H exp(GRAPH_q), GRAPH_q = the fold-q graph models).
    Fit: all good rows, target log y - log(H exp(GRAPH_full)), GRAPH_full = the all-rows graph nets' in-sample prediction. The nested recipe
    also fit the ridge on the in-sample residual of its graph models (fold-f graph models on folds != f), so this is the same recipe.
  - CV stack weights: w_all of out/cv_final_20261110.npz (fit on all five folds).
  - CV stretch: stretch_nested.fit_stretch on all good rows of the deploy pre-stretch prediction, built the way the nested stretch saw it:
    graph parts (GRAPH, ridge, both quartets) in-sample from the all-rows models, DINOv3 and frozen-head terms out-of-fold.
Writes deploy/cv_deploy_params.json, deploy/cv_ridge_params.npz, deploy/cv_final_oof.npz (OOF with ids, idx, fold). Stack file: out/$CV_OOF.npz (default cv_final_20261110b).
Also deploy/cv_inputs_oof.npz: per-row OOF member values (each row from the fold model that did not see it) with each fold's parameters, for parity tests.
usage: cv_deploy_fit_20261110.py ids | lam | inputs | fit      (fit needs graph_models/gfull_* from source/qC4.sh)"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import stretch_nested_20261103 as N

X = Path('/workspace/clear-ec-research/experiments'); LANE = X / 'lane_cv_20261108'; DEP = LANE / os.environ.get('CV_DEPLOY_DIR', 'deploy'); GM = LANE / 'graph_models'   # CV_DEPLOY_DIR: write an alternative candidate's params elsewhere
y, fold, good, H = BP.y, BP.fold, BP.good, BP.H; MASK = np.array([0.0, 1.0, 1.0]); LAMS = (300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0)
GR_SETS = {'noctx': 10, 'pre47': 3, 'pre48': 3}; Q_SETS = ('s46', 's47', 's48', 'seeds3')
sys.path.insert(0, str(LANE / 'source')); import cv_apply_20261110 as CA   # TERMS, canon, OOF_NAME (env CV_OOF)


def emb():
    z = np.load(X / 'meta_ceiling_20261103/sweep/sim_pred/sim2_l8_embed.npz'); return [z[k].astype(np.float64) for k in sorted(z.files)]


def solve(Z, ok, tr, base, j, lam):
    q = tr & (y[:, j] > 0) & ok; t = np.log(y[q, j]) - np.log(base[q, j]); mu = Z[q].mean(0); sd = Z[q].std(0) + 1e-6; A = (Z[q] - mu) / sd
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (t - t.mean())); return mu, sd, w, float(t.mean())


def oof():
    """The freeze candidate's stack file: (stack, comps, canonical component names)."""
    o = np.load(LANE / f'out/{CA.OOF_NAME}.npz', allow_pickle=True); cz = np.load(LANE / f'out/{CA.OOF_NAME}_comps.npz', allow_pickle=True)
    return o, cz, [CA.canon(str(n)) for n in o['names']]


def ids():
    o, _, names = oof(); P = o['P']; ok = y[:, 1] > 0; sc = float(np.mean(np.abs(P[ok, 1] - y[ok, 1]) / y[ok, 1]) * 100); DEP.mkdir(exist_ok=True)
    np.savez_compressed(DEP / 'cv_final_oof.npz', source=np.array(CA.OOF_NAME), ids=BP.ids, idx=BP.idx, fold=fold, P=P, cv=P[:, 1], names=np.array(names), w_all=o['w_all'], **{f'w{f}': o[f'w{f}'] for f in range(5)})
    print(json.dumps(dict(file=str(DEP / 'cv_final_oof.npz'), source=CA.OOF_NAME, rows=len(P), cv=round(sc, 4), ids_unique=bool(len(set(BP.ids)) == len(BP.ids)))), flush=True)


def lam():
    EMB = emb(); out = {}
    for j in (1, 2):
        score = {}
        for L in LAMS:
            e = 0.0
            for q in range(5):
                tr = (fold != q) & good; base = H * np.exp(BP.GRAPH[q] * MASK); pr = np.zeros(len(y))
                for Z in EMB:
                    ok = np.isfinite(Z).all(1); Zf = np.where(ok[:, None], Z, 0); mu, sd, w, t0 = solve(Zf, ok, tr, base, j, L); pr += np.where(ok, ((Zf - mu) / sd) @ w + t0, 0.0) / len(EMB)
                m = (fold == q) & (y[:, j] > 0); e += float((np.abs(base[m, j] * np.exp(pr[m]) - y[m, j]) / y[m, j]).sum())
            score[L] = e
        out[BP.K[j]] = dict(lam=min(score, key=score.get), mape_by_lam={str(int(k)): round(v / max(1, int(((y[:, j] > 0)).sum())) * 100, 4) for k, v in score.items()})
    DEP.mkdir(exist_ok=True); (DEP / 'cv_ridge_lambda.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out), flush=True); return out


def graph_full(prefix, sets):
    """Mean over sets of [mean over seeds of the all-rows nets' in-sample prediction] (N x 3, log residual vs H)."""
    per = []
    for s, ns in sets.items():
        per.append(np.mean([np.load(GM / f'{prefix.format(s=s, k=k)}_fullfit_pred.npz')['pred'].astype(np.float64) for k in range(ns)], 0))
    return np.mean(per, 0)


def fit():
    L = json.loads((DEP / 'cv_ridge_lambda.json').read_text()); EMB = emb()
    # recipe A scalars, all rows
    hyb = json.loads((X / 'recipeA_analysis_20261028/hybrid_oof.json').read_text()); A = BP._b['A']; raw = np.empty_like(A)
    for f in range(5): raw[fold == f] = A[fold == f] / np.array(hyb['recipeA_lofo_scalars'][str(f)])
    def wmed(v, w): o = np.argsort(v); c = np.cumsum(w[o]); return float(v[o][np.searchsorted(c, c[-1] / 2.)])
    sA = [wmed(y[y[:, j] > 0, j] / raw[y[:, j] > 0, j], raw[y[:, j] > 0, j] / y[y[:, j] > 0, j]) for j in range(3)]
    # all-rows GRAPH (in-sample) and the ridge on its residual
    G = graph_full('gfull_{s}_s{k}', GR_SETS); base = H * np.exp(G * MASK); R = {'n_seeds': np.array(len(EMB))}; RID = np.zeros_like(H)
    for j in (1, 2):
        col = BP.K[j]; lj = float(L[col]['lam']); R[f'lam_{col}'] = np.array(lj)
        for s, Z in enumerate(EMB):
            ok = np.isfinite(Z).all(1); Zf = np.where(ok[:, None], Z, 0); mu, sd, w, t0 = solve(Zf, ok, good.copy(), base, j, lj)
            R[f'mu_{col}_s{s}'], R[f'sd_{col}_s{s}'], R[f'w_{col}_s{s}'], R[f't0_{col}_s{s}'] = mu, sd, w, np.array(t0)
            RID[:, j] += np.where(ok, ((Zf - mu) / sd) @ w + t0, 0.0) / len(EMB)
    np.savez_compressed(DEP / 'cv_ridge_params.npz', **R)
    # quartets, in-sample from the all-rows nets
    QP = np.mean([np.load(GM / f'full_tta_{s}_tta4_fullfit_pred.npz')['pred'].astype(np.float64) for s in Q_SETS], 0)
    QO = np.mean([np.mean([np.load(GM / f'gord_{s}_tta4_full_s{k}_fullfit_pred.npz')['pred'].astype(np.float64) for k in range(3)], 0) for s in Q_SETS], 0)
    o, cz, names = oof(); w = dict(zip(names, [float(v) for v in o['w_all']])); insample_graph = {'graph:plain': QP[:, 1] - G[:, 1], 'graph:order': QO[:, 1] - G[:, 1]}
    for k, n in enumerate(names):
        if n not in insample_graph: assert all(np.array_equal(cz[f'c{k}_f0'], cz[f'c{k}_f{f}']) for f in range(5)), n   # OOF vectors, same for every fold
    cor = G[:, 1] + RID[:, 1] + sum(w[n] * (insample_graph[n] if n in insample_graph else cz[f'c{k}_f0']) for k, n in enumerate(names))
    C3 = H.copy(); C3[:, 1] = H[:, 1] * np.exp(cor); st = [float(v) for v in N.fit_stretch(C3, y, good)[1]]
    # for comparison only: the stretch fit on the OOF pre-stretch values (not stored as the deploy stretch)
    Poof = np.full(len(y), np.nan)
    for f in range(5):
        wf = [float(v) for v in o[f'w{f}']]; c = BP.BEST[f][:, 1] + sum(wf[k] * cz[f'c{k}_f{f}'] for k in range(len(names))); Poof[fold == f] = (H[:, 1] * np.exp(c))[fold == f]
    C4 = H.copy(); C4[:, 1] = Poof; st_oof = [float(v) for v in N.fit_stretch(C4, y, good)[1]]
    ok = y[:, 1] > 0; oof_cv = float(np.mean(np.abs(o['P'][ok, 1] - y[ok, 1]) / y[ok, 1]) * 100)
    prm = dict(written='2026-11-10', recipe=f'CV lane freeze candidate, out/{CA.OOF_NAME}.npz (five-fold OOF CV {oof_cv:.4f})', components=names,
               recipeA_scalar_cv=sA[1], recipeA_scalars_all=sA, phase1_scalar_cv=float(BP.C.SCALE[1]), phase1_scalars_all=[float(v) for v in BP.C.SCALE],
               weights=w, stretch_cv=st, stretch_cv_fit_on_oof_for_comparison=st_oof, ridge_lambda={k: float(L[k]['lam']) for k in ('CV', 'HEX')},
               graph_ensemble=dict(weights='graph_models/gfull_{noctx,pre47,pre48}_s<k>_s<k>_e{21..25}.pt', seeds=GR_SETS, recipe='d64, 5 layers, 25 epochs, lr 1e-3, wd 0.01, drop 0.1, no ctx, 9 node features; graphs meta_ceiling_20261103/sweep/cell_graphs{,_pre47,_pre48}.npz layout'),
               sim_embedding=dict(networks='meta_ceiling_20261103/sweep/sim_pred/sim2_l8_s{0,1,2}.pt', embed='sim_embed_eval_20261105.embed, mean over the 3 graph sets'),
               order_dino='lane_cv_20261108/arms/dinov3_order_f99_s46 (full data, CD_LOSS=order), epochs 16-20 mean log CV, single view',
               quartets=dict(plain='graph_models/full_tta_{s46,s47,s48,seeds3}_tta4_s{0,1,2}_e{21..25}.pt', order='graph_models/gord_{s46,s47,s48,seeds3}_tta4_full_s<k>_s<k>_e{21..25}.pt'),
               detectors='green_track_20261028/deploy/detectors/sg2pre_s{46,47,48}.pt (md5 06c90eee, c84866ca, 4733d1fc)')
    (DEP / 'cv_deploy_params.json').write_text(json.dumps(prm, indent=1))
    np.savez_compressed(DEP / 'cv_deploy_insample_members.npz', G=G, RID=RID, QP=QP, QO=QO, cor_cv=cor)
    print(json.dumps(dict(recipeA_scalars=sA, stretch_cv=st, stretch_cv_oof=st_oof, lam=prm['ridge_lambda'], weights=w)), flush=True)


def inputs():
    """Per-row OOF inputs (row in fold f takes fold f's model values) plus per-fold parameters; cv_apply.apply_fold reproduces the OOF CV."""
    o, cz, names = oof(); hyb = json.loads((X / 'recipeA_analysis_20261028/hybrid_oof.json').read_text()); A = BP._b['A']; P1 = BP._b['P1']; s1 = float(BP.C.SCALE[1]); n = len(y)
    keys = ['p1_cv', 'a_raw_cv', 'graph_cv', 'ridge_cv', 'fz_std'] + [CA.TERMS[q][0] for q in names]; T = {k: np.full(n, np.nan) for k in keys}; sA = np.zeros(5); st = {}
    for f in range(5):
        r = fold == f; sA[f] = hyb['recipeA_lofo_scalars'][str(f)][1]; G = BP.GRAPH[f][:, 1]; logA = np.log(A[:, 1])
        T['p1_cv'][r] = P1[r, 1] / s1; T['a_raw_cv'][r] = A[r, 1] / sA[f]; T['graph_cv'][r] = G[r]; T['ridge_cv'][r] = (BP.BEST[f][:, 1] - G)[r]; T['fz_std'][r] = 0.0
        for k, q in enumerate(names):
            key, ref = CA.TERMS[q]; T[key][r] = (cz[f'c{k}_f{f}'] + (logA if ref == 'logA' else G if ref == 'graph' else 0.0))[r]
        wf = [float(v) for v in o[f'w{f}']]; c = BP.BEST[f][:, 1] + sum(wf[k] * cz[f'c{k}_f{f}'] for k in range(len(names))); C3 = H.copy(); C3[:, 1] = H[:, 1] * np.exp(c)
        st[f'stretch{f}'] = np.array(N.fit_stretch(C3, y, (fold != f) & good)[1], float)
    np.savez_compressed(DEP / 'cv_inputs_oof.npz', source=np.array(CA.OOF_NAME), ids=BP.ids, idx=BP.idx, fold=fold, cv_oof=o['P'][:, 1], sA=sA, phase1_scalar_cv=np.array(s1), names=np.array(names),
                        note=np.array('fz_* hold log differences against fz_std (fz_std = 0): the stack only ever used lg_arm - lg_std. dino_logcv / orderdino_logcv = (log member - log A_cv) + log A_cv.'),
                        **T, **st, **{f'w{f}': o[f'w{f}'] for f in range(5)})
    Z = np.load(DEP / 'cv_inputs_oof.npz'); pred = np.full(n, np.nan)
    for f in range(5): pred[fold == f] = CA.apply_fold(Z, f)
    print(json.dumps(dict(file=str(DEP / 'cv_inputs_oof.npz'), source=CA.OOF_NAME, max_abs_diff_vs_oof=float(np.nanmax(np.abs(pred - o['P'][:, 1]))))), flush=True)


def insample():
    """Deploy parameters applied to the training rows (graph parts in-sample): an optimistic sanity number, not a score."""
    P = CA.load_params(DEP); D = np.load(DEP / 'cv_deploy_insample_members.npz'); o, cz, names = oof()
    A = BP._b['A']; hyb = json.loads((X / 'recipeA_analysis_20261028/hybrid_oof.json').read_text()); raw = np.empty(len(y))
    for f in range(5): raw[fold == f] = A[fold == f, 1] / hyb['recipeA_lofo_scalars'][str(f)][1]
    a_cv = raw * P['recipeA_scalar_cv']; G = D['G'][:, 1]
    m = dict(p1_cv=BP._b['P1'][:, 1] / P['phase1_scalar_cv'], a_raw_cv=raw, graph_cv=G, qplain_cv=D['QP'][:, 1], qorder_cv=D['QO'][:, 1], fz_std=np.zeros(len(y)))
    for k, q in enumerate(names):
        key, ref = CA.TERMS[q]
        if ref != 'graph': m[key] = cz[f'c{k}_f0'] + (np.log(a_cv) if ref == 'logA' else 0.0)
    E = emb()
    for s, Z in enumerate(E): m[f'emb_s{s}'] = Z
    cv = CA.apply(P, m); ok = y[:, 1] > 0
    print(json.dumps(dict(insample_cv=round(float(np.mean(np.abs(cv[ok] - y[ok, 1]) / y[ok, 1]) * 100), 4), ridge_matches_fit=float(np.max(np.abs(CA.ridge(P, E, 'CV') - D['RID'][:, 1]))))), flush=True)


if __name__ == '__main__':
    {'ids': ids, 'lam': lam, 'inputs': inputs, 'fit': fit, 'insample': insample}[sys.argv[1]]()
