"""Cheapest kill test of plan idea I7, donor memory from images (lane GPU 1, 2026-11-07). Can an embedding find a frame's fellow eye? Gate from the ideation draft: kill below 40 % top-1 precision on source A's neighbours.
Features: pooled tokens of three encoders, already extracted. Metric: PCA 128, then whitening by the within-pair covariance from half of source B's true pairs (a linear pair embedding, no GPU). Scored on the other half of B's pairs and on A's serial neighbours, among all 8954 frames."""
import json, re, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
L = '/workspace/clear-ec-research/experiments/lane_gpu1_20261106/features/'
z = np.load('/workspace/clear-ec-research/datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True)
ids = np.array([str(i).strip() for i in z['ids']]); N = len(ids)
donor = {}; eye = {}; aser = {}
for i, s in enumerate(ids):
    m = re.fullmatch(r'(\d{4})[- ](\d+)[- ]?(OD|OS)[A-Z]*', s)
    if m:
        donor.setdefault((m[1], m[2]), []).append(i); continue
    m = re.fullmatch(r'(\d+)-(\d\d)', s)
    if m:
        aser[(m[2], int(m[1]))] = i
pairs = np.array([v[:2] for v in donor.values() if len(v) == 2])
rng = np.random.default_rng(0); pm = rng.permutation(len(pairs)); tr, te = pairs[pm[:len(pm) // 2]], pairs[pm[len(pm) // 2:]]
Atest = {i: [aser[(y, n + d)] for d in (-1, 1) if (y, n + d) in aser] for (y, n), i in aser.items()}
Atest = {i: v for i, v in Atest.items() if v}
print('B pairs', len(pairs), 'A rows with a neighbour', len(Atest), flush=True)


def rank_report(S, q, truth):
    S = S.copy(); np.fill_diagonal(S, -np.inf)
    top = S[q].argmax(1); rk = []
    for k, i in enumerate(q):
        order = S[i]; best = max(order[j] for j in truth[i]); rk.append(int((order > best).sum()) + 1)
    top1 = float(np.mean([top[k] in truth[i] for k, i in enumerate(q)]))
    return dict(top1=round(top1, 4), top10=round(float(np.mean(np.array(rk) <= 10)), 4), median_rank=int(np.median(rk)))


out = {}
for name in ('cropopt_both_f4', 'ssl5_f4', 'tuned_f4', 'ext'):
    f = np.load(L + name + '.npz'); X = np.c_[f['mean'], f['std']].astype(np.float32); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    U, s, Vt = np.linalg.svd(X[rng.choice(N, 3000, replace=False)], full_matrices=False); P = X @ Vt[:128].T
    d = P[tr[:, 0]] - P[tr[:, 1]]; Sw = d.T @ d / len(d) + 1e-3 * np.eye(128)
    w, V = np.linalg.eigh(Sw); W_ = V / np.sqrt(w)
    for tag, Z in (('raw_pca', P), ('within_pair_whitened', P @ W_)):
        Z = Z - Z.mean(0); Z = Z / np.linalg.norm(Z, axis=1, keepdims=True); S = Z @ Z.T
        tb = {int(a): [int(b)] for a, b in te}; tb.update({int(b): [int(a)] for a, b in te}); qb = np.array(sorted(tb))
        qa = np.array(sorted(Atest)); r = dict(B_heldout=rank_report(S, qb, tb), A_neighbours=rank_report(S, qa, Atest))
        out[f'{name} {tag}'] = r; print(name, tag, json.dumps(r), flush=True)
json.dump(out, open('/workspace/clear-ec-research/experiments/lane_gpu1_20261106/reads/donor_link.json', 'w'), indent=1)
