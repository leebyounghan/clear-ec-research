"""Plan idea I7, direct read (lane GPU 1, 2026-11-07). Correct the best pipeline by the OOF residuals of a frame's nearest neighbours in other folds, in a space whitened by within-pair covariance of source B's true fellow-eye pairs. This reads the gain itself rather than link precision.
Nested: K and the shrink per metric are chosen on the other folds, whose neighbours exclude the held-out fold, and the whitening uses pairs outside the held-out fold. Gate as the plan's corrections: <= -0.01, CI below 0, 4 of 5 folds.
Caveat: it assumes a test frame has its partner in training like an OOF frame does. Unverified."""
import json, re, sys
import numpy as np
sys.path.insert(0, 'experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
E = BP.E; K_ = BP.K
L = 'experiments/lane_gpu1_20261106/features/'
y, fold, idx = BP.y, BP.fold, BP.idx; ids = np.array([str(i).strip() for i in BP.ids]); P0 = BP.REF_BEST
ok = (y > 0); res = np.where(ok, np.log(np.where(ok, y, 1.0)) - np.log(P0), 0.0)
isA = np.array([bool(re.fullmatch(r'\d+-\d\d', s)) for s in ids])
don = {}
for i, s in enumerate(ids):
    m = re.fullmatch(r'(\d{4})[- ](\d+)[- ]?(OD|OS)[A-Z]*', s)
    if m:
        don.setdefault((m[1], m[2]), []).append(i)
pairs = np.array([v[:2] for v in don.values() if len(v) == 2])
KS = (1, 3, 10, 30); ALPHA = np.arange(0, 0.61, 0.05)


def space(X, ex):
    """PCA 128 and within-pair whitening with no rows of the excluded folds."""
    use = ~np.isin(fold, ex)
    mu, sd = X[use].mean(0), X[use].std(0) + 1e-6; Z = (X - mu) / sd
    sub = np.where(use)[0][::3]; _, _, Vt = np.linalg.svd(Z[sub], full_matrices=False); P = Z @ Vt[:128].T
    pr = pairs[use[pairs[:, 0]] & use[pairs[:, 1]]]; d = P[pr[:, 0]] - P[pr[:, 1]]; S = d.T @ d / len(d) + 1e-3 * np.eye(128)
    w, V = np.linalg.eigh(S); Q = (P @ (V / np.sqrt(w))).astype(np.float32); Q -= Q[use].mean(0)
    return Q / np.linalg.norm(Q, axis=1, keepdims=True)


def nb_res(Z, q, c):
    s = Z[q] @ Z[c].T; o = np.argsort(-s, axis=1)[:, :max(KS)]
    return {k: res[c[o[:, :k]]].mean(1) for k in KS}


def run(name):
    f_ = np.load(L + name + '.npz'); X = np.c_[f_['mean'], f_['std']].astype(np.float32)[idx]
    P = np.full_like(P0, np.nan); chosen = []
    for f in range(5):
        Zf = space(X, [f]); q = np.where(fold == f)[0]; c = np.where(fold != f)[0]; test = nb_res(Zf, q, c)
        # choose K and alpha on the other folds: neighbours exclude the held-out fold and the row's own fold
        S = {k: np.zeros((len(ALPHA), 3)) for k in KS}
        for g in range(5):
            if g == f:
                continue
            qq = np.where(fold == g)[0]; cc = np.where((fold != f) & (fold != g))[0]; nb = nb_res(Zf, qq, cc)
            for k in KS:
                for ai, a in enumerate(ALPHA):
                    S[k][ai] += np.nansum(E.ape(P0[qq] * np.exp(a * nb[k]), y[qq]), 0)
        pick = []
        for m in range(3):
            k, ai = min(((k, int(S[k][:, m].argmin())) for k in KS), key=lambda t: S[t[0]][t[1], m])
            P[q, m] = P0[q, m] * np.exp(ALPHA[ai] * test[k][:, m]); pick.append((k, round(float(ALPHA[ai]), 2)))
        chosen.append(pick)
    r = BP.report('donor memory ' + name, P, P0); r['chosen_K_alpha_per_fold'] = chosen
    for tag, m in (('source A', isA), ('source B', ~isA)):
        pb = E.paired_bootstrap(E.ape(P[m], y[m]), E.ape(P0[m], y[m]), np.unique(BP.donors[m], return_inverse=True)[1]); r[tag] = dict(delta=round(pb['delta'], 4), ci=[round(v, 4) for v in pb['ci95']])
    print(json.dumps({k: r[k] for k in ('source A', 'source B', 'chosen_K_alpha_per_fold')}), flush=True)
    return r


out = [run(n) for n in ('cropopt_both_f4', 'tuned_f4')]
json.dump(out, open('experiments/lane_gpu1_20261106/reads/donor_memory.json', 'w'), indent=1)
