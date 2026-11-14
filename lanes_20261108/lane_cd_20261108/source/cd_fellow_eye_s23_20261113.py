"""Fellow-eye read steps 2-3 (coordinator 22:35 EST 11-13), CPU only. Features per encoder fold e (recipe A fold-e model, token statistics:
lane_gpu2_20261106/frozen_own_f{0,4}, lane_cv_20261108/frozen_own_f{1,2,3}) plus label-free image/mask statistics. Queries in fold e search
rows outside fold e with encoder e (cosine on z-scored, L2-normalised features).
Step 2: identifiability: rank of the old-format gap-1 partner (other fold) among candidates; new-format fellow eye (same fold) among all rows.
Step 3: kNN transfer, strict: for outer fold f, (feature set, k, gate quantile, lambda) chosen on inner folds g != f (queries of fold g, neighbours
outside {f, g}), applied to fold f (neighbours outside f). Two forms: R = residual transfer, log p += lam * mean(log y_nb - log p_nb) (p_nb strict
OOF: second-hand leak, its model trained on the scored fold); Lb = label blend, log p += lam * (mean log y_nb - log p) (uses only training labels
and the row's own strict OOF: no second-hand leak). Writes reads/cd_fellow_eye_s23_20261113.json."""
import sys, re, json, collections, numpy as np
R_ = '/workspace/clear-ec-research/experiments/'; L = R_ + 'lane_cellroute_20261109/'
sys.path[:0] = [R_ + 'meta_ceiling_20261103/source']
import best_pipeline_20261106 as BP
idx = BP.idx.astype(int); Y = np.asarray(BP.y, float); fold = np.asarray(BP.fold).astype(int); ids = np.array([str(i).strip() for i in BP.ids]); n = len(idx)
donors = np.asarray(BP.donors); fmt = np.array(['P2' if re.match(r'^\d{4}-\d{4}', i) else 'P1' for i in ids]); isP1 = fmt == 'P1'
zc = np.load(R_ + 'lane_cd_20261108/reads/cd_tier_t1_20261112.npz'); assert (zc['idx'].astype(int) == idx).all()
zv = np.load(R_ + 'lane_cv_20261108/deploy_t1/cv_final_oof.npz', allow_pickle=True); zh = np.load(L + 'out/corr_hex_e2e_strict_t2_a_chain_convention.npz')
P = np.stack([np.asarray(zc['oof_cd'], float), np.asarray(zv['P'][:, 1], float), np.asarray(zh['P'], float)], 1)
ok = Y > 0; LY = np.log(np.where(ok, Y, np.nan)); LP = np.log(P); RES = LY - LP
def mape(p, j, m=None):
    s = ok[:, j] if m is None else (m & ok[:, j]); return float(np.mean(np.abs(p[s] / Y[s, j] - 1)) * 100)
ser = np.array([int(i.split('-')[0]) if f == 'P1' else -1 for i, f in zip(ids, fmt)]); yr = np.array([i.split('-')[1] if f == 'P1' else '' for i, f in zip(ids, fmt)])
key = {(yr[i], ser[i]): i for i in np.where(isP1)[0]}
nb1 = {i: [key[(yr[i], ser[i] + s)] for s in (-1, 1) if (yr[i], ser[i] + s) in key] for i in np.where(isP1)[0]}
has_adj_train = np.array([any(fold[q] != fold[i] for q in nb1.get(i, [])) for i in range(n)])
sl = np.array([i[:9] if f == 'P2' else '' for i, f in zip(ids, fmt)]); by = collections.defaultdict(list)
for i in np.where(~isP1)[0]: by[sl[i]].append(i)
fe_partner = {}
for v in by.values():
    if len(v) >= 2: fe_partner[v[0]] = v[1]; fe_partner[v[1]] = v[0]
ENC = {0: R_ + 'lane_gpu2_20261106/frozen_own_f0', 4: R_ + 'lane_gpu2_20261106/frozen_own_f4', 1: R_ + 'lane_cv_20261108/frozen_own_f1', 2: R_ + 'lane_cv_20261108/frozen_own_f2', 3: R_ + 'lane_cv_20261108/frozen_own_f3'}
ims = np.load(R_ + 'eval_harness_20261031/image_stats.npz'); assert (ims['idx'].astype(int) == idx).all()
cheap = np.stack([ims[k] for k in ('mean', 'contrast', 'black_frac', 'focus')], 1)
SETS = {'meanstd': ('mean', 'std'), 'cls': ('cls',), 'spectrum': ('spectrum',), 'mix': ('mean', 'std', 'cls', 'skew')}
def norm(X):
    X = np.nan_to_num((X - X.mean(0)) / (X.std(0) + 1e-6)); return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
TOP = 60; NB = {}; SIMS = {}; FE_RANK = {}
for s, blocks in list(SETS.items()) + [('cheap', None)]:
    for e in range(5):
        X = norm(cheap) if blocks is None else norm(np.concatenate([np.load(f'{ENC[e]}/{b}.npy').astype(np.float32) for b in blocks], 1))
        q = np.where(fold == e)[0]; S_ = X[q] @ X.T
        S2 = S_.copy(); S2[:, fold == e] = -9; o = np.argsort(-S2, 1)[:, :TOP]; NB[(s, e)] = o; SIMS[(s, e)] = np.take_along_axis(S2, o, 1)
        S3 = S_.copy(); S3[np.arange(len(q)), q] = -9; o3 = np.argsort(-S3, 1)[:, :20]
        for r, i in enumerate(q):
            if i in fe_partner: FE_RANK[(s, i)] = int(np.where(o3[r] == fe_partner[i])[0][0]) + 1 if fe_partner[i] in o3[r] else 99
    print('features done', s, flush=True)
qpos = {e: {int(i): r for r, i in enumerate(np.where(fold == e)[0])} for e in range(5)}
# step 2
st2 = {}
for s in list(SETS) + ['cheap']:
    ranks = []
    for i in np.where(isP1)[0]:
        pt = [q for q in nb1[i] if fold[q] != fold[i]]
        if not pt: continue
        row = NB[(s, fold[i])][qpos[fold[i]][i]]; rk = [int(np.where(row == q)[0][0]) + 1 for q in pt if q in row]; ranks.append(min(rk) if rk else 999)
    ranks = np.array(ranks); fr = np.array([FE_RANK[(s, i)] for i in fe_partner if (s, i) in FE_RANK])
    st2[s] = dict(old_format_queries=int(len(ranks)), top1=round(float(np.mean(ranks == 1)), 4), top5=round(float(np.mean(ranks <= 5)), 4), top20=round(float(np.mean(ranks <= 20)), 4),
                  new_format_fellow_eye_queries=int(len(fr)), fe_top1=round(float(np.mean(fr == 1)), 4), fe_top5=round(float(np.mean(fr <= 5)), 4), fe_top20=round(float(np.mean(fr <= 20)), 4))
out = {'step2_identifiability': st2, 'chance_top1_old': round(1 / 7163, 5)}
print(json.dumps(st2), flush=True)
# step 3
KS = (1, 3, 10); GQ = (0.0, 0.5, 0.75, 0.9); LAM = np.round(np.arange(0, 1.001, 0.05), 2)
def corr_terms(s, e, excl, k, j, form):
    """for queries of fold e: (top-1 sim, correction direction) using neighbours not in folds 'excl' (list), first k of them"""
    o = NB[(s, e)]; sm = SIMS[(s, e)]; m = ~np.isin(fold[o], list(excl)) & np.isfinite(RES[o, j])
    rank = np.cumsum(m, 1); m &= rank <= k
    s1 = np.where(m, sm, -9).max(1)
    if form == 'R': v = np.where(m, RES[o, j], 0).sum(1) / np.maximum(m.sum(1), 1)
    else:
        q = np.where(fold == e)[0]; v = np.where(m, LY[o, j], 0).sum(1) / np.maximum(m.sum(1), 1) - LP[q, j]
    return s1, v
res3 = {}
for form in ('R', 'Lb'):
    for j, kname in ((0, 'CD'), (1, 'CV'), (2, 'HEX')):
        Pc = P[:, j].copy(); picks = {}
        for f in range(5):
            best = None
            for s in list(SETS) + ['cheap']:
                for k in KS:
                    inner = {g: corr_terms(s, g, [f, g], k, j, form) for g in range(5) if g != f}
                    allS1 = np.concatenate([inner[g][0] for g in inner])
                    for gq in GQ:
                        thr = np.quantile(allS1, gq) if gq > 0 else -9
                        for lam in LAM:
                            num = 0; den = 0
                            for g, (s1, v) in inner.items():
                                q = np.where(fold == g)[0]; okq = ok[q, j]; adj = np.where(s1 >= thr, lam * v, 0.0)
                                num += np.sum(np.abs(P[q[okq], j] * np.exp(adj[okq]) / Y[q[okq], j] - 1)); den += okq.sum()
                            e_ = num / den
                            if best is None or e_ < best[0]: best = (e_, s, k, gq, float(lam), float(thr))
            _, s, k, gq, lam, thr = best; picks[f] = dict(features=s, k=k, gate_quantile=gq, lam=lam)
            s1, v = corr_terms(s, f, [f], k, j, form); q = np.where(fold == f)[0]; Pc[q] = P[q, j] * np.exp(np.where(s1 >= thr, lam * v, 0.0))
        rng = np.random.default_rng(0); okj = ok[:, j]; d = (np.abs(Pc / np.where(okj, Y[:, j], 1) - 1) - np.abs(P[:, j] / np.where(okj, Y[:, j], 1) - 1)) * 100
        u, inv = np.unique(donors[okj], return_inverse=True); sd = np.bincount(inv, d[okj]); cn = np.bincount(inv); bs = []
        for _ in range(2000):
            kk = rng.integers(0, len(u), len(u)); bs.append(sd[kk].sum() / cn[kk].sum())
        res3[f'{form}_{kname}'] = dict(strict=round(mape(Pc, j), 4), base=round(mape(P[:, j], j), 4), delta=round(mape(Pc, j) - mape(P[:, j], j), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                                       per_fold_delta={f: round(mape(Pc, j, fold == f) - mape(P[:, j], j, fold == f), 4) for f in range(5)}, picks=picks,
                                       delta_old_format=round(mape(Pc, j, isP1) - mape(P[:, j], j, isP1), 4), delta_new_format=round(mape(Pc, j, ~isP1) - mape(P[:, j], j, ~isP1), 4),
                                       delta_has_adjacent_in_train=round(mape(Pc, j, has_adj_train) - mape(P[:, j], j, has_adj_train), 4),
                                       delta_old_no_adjacent_in_train=round(mape(Pc, j, isP1 & ~has_adj_train) - mape(P[:, j], j, isP1 & ~has_adj_train), 4))
        print(form, kname, json.dumps({kk: vv for kk, vv in res3[f'{form}_{kname}'].items() if kk != 'picks'}), json.dumps(picks), flush=True)
out['step3_knn_transfer'] = res3
json.dump(out, open(R_ + 'lane_cd_20261108/reads/cd_fellow_eye_s23_20261113.json', 'w'), indent=1); print('saved', flush=True)
