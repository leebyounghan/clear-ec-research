"""Fit the deploy parameters of the cell-route HEX step on all folds, and check the deploy code path against the stored OOF result.
Checks: (1) deploy_hexkonan.features() on 300 random frames from the stored peaks / seg centroids equals feats/cellroute_feats.npz (P2 columns);
(2) per-fold parameters from the same fit function, applied with deploy_hexkonan.correct(), reproduce out/corr_hex_a100.npz on each held-out fold.
Writes out/deploy_hexkonan_params.json (base = lane_hex_20261108/preds/chain_cvw3.npz final_hex; refit if the HEX lane's deployed chain changes)."""
import sys, json
import numpy as np
sys.path[:0] = ['/workspace/clear-ec-research/experiments/lane_cellroute_20261109/source', '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source']
import deploy_hexkonan as D
import best_pipeline_20261106 as BP
import ceiling_meta_cpu_20261103 as C
import legible_window_features_20261105 as LW
ROOT = '/workspace/clear-ec-research/experiments/'; LANE = ROOT + 'lane_cellroute_20261109/'; ALPHA = 100.0
y, fold, good, idx = BP.y, BP.fold, BP.good, BP.idx.astype(int)
z = np.load(LANE + 'feats/cellroute_feats.npz'); NM = list(z['names']); F = z['F'][:, [NM.index(c) for c in D.NAMES]]  # dataset index order
Fr = F[idx]; base = np.load(ROOT + 'lane_hex_20261108/preds/chain_cvw3.npz')['final_hex']


def fit(rows):
    X, m = D.design(Fr, base); ok = y[:, 2] > 0; res = np.log(np.where(ok, y[:, 2], 1.0)) - np.log(base); tr = rows & good & m & ok
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; A = (X[tr] - mu) / sd; w = np.linalg.solve(A.T @ A + ALPHA * np.eye(A.shape[1]), A.T @ (res[tr] - res[tr].mean()))
    q = base * np.exp(((X - mu) / sd) @ w); c = C.wmed(y[tr, 2] / q[tr], q[tr] / y[tr, 2]); c0 = C.wmed(y[tr, 2] / base[tr], base[tr] / y[tr, 2])
    return dict(mu=mu.tolist(), sd=sd.tolist(), w=w.tolist(), c=float(c), c0=float(c0), alpha=ALPHA, names=D.NAMES + ['log_base_hex'], train_rows=int(tr.sum()))


# (1) feature parity from stored inputs
def load(tag):
    out = {}
    for f in range(5):
        zz = np.load(ROOT + f'lane_cv_20261108/out/tta_peaks/{tag}/fold{f}/fold{f}_sg2_peaks.npz', allow_pickle=True); pk, of, ix = zz['peaks'], zz['offsets'], zz['indices']
        for k, i in enumerate(ix): out[int(i)] = pk[of[k]:of[k + 1]]
    return out
PK = {t: load(t) for t in ('seeds3_tta4', 's46_tta4', 's47_tta4', 's48_tta4')}
zs = np.load(LANE + 'feats/seg_centroids.npz'); _p, _o, _i = zs['pts'], zs['offsets'], zs['indices']; SEG = {int(d): _p[_o[k]:_o[k + 1]] for k, d in enumerate(_i)}
rng = np.random.default_rng(0); test = rng.choice(8954, 300, replace=False); md = 0.0; nan_mismatch = 0
from scipy.spatial import cKDTree
for r in test:
    v = D.features(PK['seeds3_tta4'].get(r), PK['s46_tta4'][r], PK['s47_tta4'][r], PK['s48_tta4'][r], SEG.get(r, np.zeros((0, 3))))
    a, b = v, F[r]; fin = np.isfinite(a) & np.isfinite(b); nan_mismatch += int((np.isfinite(a) != np.isfinite(b)).sum()); md = max(md, float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0)
    pk3 = PK['seeds3_tta4'][r].astype(float); hi = pk3[pk3[:, 2] >= 0.5, :2]
    if len(hi) > 20: assert D.windows(hi, 380, 300) == LW.windows(hi, 380, 300), 'window copy differs'
print('feature parity on 300 frames: max abs diff %.2e, NaN-pattern mismatches %d' % (md, nan_mismatch), flush=True)
# (2) fold reproduction
ref = np.load(LANE + 'out/corr_hex_a100.npz')['P'][:, 2]; worst = 0.0
for f in range(5):
    prm = fit(fold != f); q = D.correct(base, Fr, prm); sel = fold == f; worst = max(worst, float(np.max(np.abs(q[sel] / ref[sel] - 1))))
print('fold reproduction of out/corr_hex_a100.npz: max relative diff %.2e' % worst, flush=True)
P = fit(np.ones(len(y), bool)); P['base'] = 'lane_hex_20261108/preds/chain_cvw3.npz final_hex'; P['min_full_cells'] = 50
json.dump(P, open(LANE + 'out/deploy_hexkonan_params.json', 'w'), indent=1); print('deploy params written, train rows', P['train_rows'], flush=True)
