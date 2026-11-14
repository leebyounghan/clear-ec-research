"""PI request (coordinator 22:35 EST 11-13), step 1: size of the prize with the ID as an oracle (not deployable). Candidate = tier p strict OOF
(CD t1 reads/cd_tier_t1_20261112.npz; CV t1 lane_cv deploy_t1/cv_final_oof.npz; HEX t2 + cell-route e2e strict). Old-format IDs 'SSSS-YY':
pairs = same YY, serial gap g. Residual = log(label / strict-OOF prediction). Oracle correction: log p_i += lambda x mean residual of its gap-1
partners that sit in another fold (training side); lambda (grid 0-1.5) fitted per outer fold on rows of the other folds whose partners are also
outside the scored fold. Writes reads/cd_fellow_eye_s1_20261113.json."""
import sys, re, json, collections, numpy as np
R = '/workspace/clear-ec-research/experiments/'; L = R + 'lane_cellroute_20261109/'
sys.path[:0] = [R + 'meta_ceiling_20261103/source']
import best_pipeline_20261106 as BP
idx = BP.idx.astype(int); Y = np.asarray(BP.y, float); fold = np.asarray(BP.fold).astype(int); ids = np.array([str(i).strip() for i in BP.ids]); n = len(idx)
fmt = np.array(['P2' if re.match(r'^\d{4}-\d{4}', i) else 'P1' for i in ids]); isP1 = fmt == 'P1'
def mape(p, y, m=None):
    s = (y > 0) if m is None else (m & (y > 0)); return float(np.mean(np.abs(p[s] / y[s] - 1)) * 100)
zc = np.load(R + 'lane_cd_20261108/reads/cd_tier_t1_20261112.npz'); assert (zc['idx'].astype(int) == idx).all()
zv = np.load(R + 'lane_cv_20261108/deploy_t1/cv_final_oof.npz', allow_pickle=True); assert (zv['idx'].astype(int) == idx).all()
zh = np.load(L + 'out/corr_hex_e2e_strict_t2_a_chain_convention.npz'); assert (zh['idx'].astype(int) == idx).all()
P = np.stack([np.asarray(zc['oof_cd'], float), np.asarray(zv['P'][:, 1], float), np.asarray(zh['P'], float)], 1)
base = [mape(P[:, j], Y[:, j]) for j in range(3)]
ok = Y > 0; LY = np.log(np.where(ok, Y, np.nan)); RES = LY - np.log(P)
ser = np.array([int(i.split('-')[0]) if f == 'P1' else -1 for i, f in zip(ids, fmt)]); yr = np.array([i.split('-')[1] if f == 'P1' else '' for i, f in zip(ids, fmt)])
key = {(yr[i], ser[i]): i for i in np.where(isP1)[0]}
def pr(a, b):
    m = np.isfinite(a) & np.isfinite(b); return round(float(np.corrcoef(a[m], b[m])[0, 1]), 3)
out = {'baseline_strict_CD_CV_HEX': [round(v, 4) for v in base], 'old_format_rows': int(isP1.sum())}
corr = {}
for gap in (1, 2, 3):
    pp = np.array([(i, key[(yr[i], ser[i] + gap)]) for i in np.where(isP1)[0] if (yr[i], ser[i] + gap) in key]); d = fold[pp[:, 0]] != fold[pp[:, 1]]
    corr[f'gap{gap}'] = dict(pairs=int(len(pp)), pairs_diff_fold=int(d.sum()), label_r=[pr(LY[pp[:, 0], j], LY[pp[:, 1], j]) for j in range(3)],
                             residual_r_diff_fold=[pr(RES[pp[d, 0], j], RES[pp[d, 1], j]) for j in range(3)], residual_r_same_fold=[pr(RES[pp[~d, 0], j], RES[pp[~d, 1], j]) for j in range(3)])
out['1a_old_format_pair_correlations'] = corr
sl = np.array([i[:9] if f == 'P2' else '' for i, f in zip(ids, fmt)]); by = collections.defaultdict(list)
for i in np.where(~isP1)[0]: by[sl[i]].append(i)
fe = np.array([v[:2] for v in by.values() if len(v) >= 2]); dfe = fold[fe[:, 0]] != fold[fe[:, 1]]
out['1b_new_format_fellow_eyes'] = dict(pairs=int(len(fe)), pairs_diff_fold=int(dfe.sum()), label_r=[pr(LY[fe[:, 0], j], LY[fe[:, 1], j]) for j in range(3)],
                                        residual_r=[pr(RES[fe[:, 0], j], RES[fe[:, 1], j]) for j in range(3)])
# oracle correction with gap-1 partners in another fold
nb = {i: [key[(yr[i], ser[i] + s)] for s in (-1, 1) if (yr[i], ser[i] + s) in key] for i in np.where(isP1)[0]}
lam_grid = np.round(np.arange(0, 1.501, 0.05), 2); res1 = {}
for j, k in enumerate(('CD', 'CV', 'HEX')):
    Pc = P[:, j].copy(); lams = {}; elig_all = np.zeros(n, bool)
    def corr_term(i, excl):   # mean residual of partners not in folds 'excl'
        v = [RES[q, j] for q in nb.get(i, []) if fold[q] not in excl and np.isfinite(RES[q, j])]; return np.mean(v) if v else np.nan
    for f in range(5):
        # fit rows: folds != f, partners outside their own fold and outside f
        fr = [(i, corr_term(i, {fold[i], f})) for i in np.where(isP1 & (fold != f) & ok[:, j])[0]]; fr = [(i, c) for i, c in fr if np.isfinite(c)]
        ii = np.array([a for a, _ in fr]); cc = np.array([c for _, c in fr]); best = None
        for lam in lam_grid:
            e = np.mean(np.abs(P[ii, j] * np.exp(lam * cc) / Y[ii, j] - 1))
            if best is None or e < best[0]: best = (e, lam)
        lams[f] = float(best[1])
        for i in np.where(isP1 & (fold == f))[0]:
            c = corr_term(i, {f})
            if np.isfinite(c): Pc[i] = P[i, j] * np.exp(best[1] * c); elig_all[i] = True
    res1[k] = dict(lambda_per_fold=lams, strict=round(mape(Pc, Y[:, j]), 4), delta=round(mape(Pc, Y[:, j]) - base[j], 4),
                   per_fold_delta={f: round(mape(Pc, Y[:, j], fold == f) - mape(P[:, j], Y[:, j], fold == f), 4) for f in range(5)},
                   corrected_rows=int(elig_all.sum()), old_format_delta=round(mape(Pc, Y[:, j], isP1) - mape(P[:, j], Y[:, j], isP1), 4),
                   corrected_rows_mape_before_after=[round(mape(P[:, j], Y[:, j], elig_all), 3), round(mape(Pc, Y[:, j], elig_all), 3)])
out['1c_oracle_partner_residual_correction'] = res1
out['note'] = 'oracle: uses the ID to find the partner (not deployable). Partner residual uses its label (training side) and its strict OOF prediction, whose model trained on folds != fold(partner), which include the scored row (second-hand).'
print(json.dumps(out, indent=1)); json.dump(out, open(R + 'lane_cd_20261108/reads/cd_fellow_eye_s1_20261113.json', 'w'), indent=1)
