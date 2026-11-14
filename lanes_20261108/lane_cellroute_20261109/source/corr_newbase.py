"""Cell-route lane (2026-11-10): the kept HEX step (P2 Konan features, windows + full frame, ridge alpha 100, >= 50 cells; recipe unchanged) re-read
nested over five folds on a new HEX lane base. usage: corr_newbase.py BASE.npz [OUTNAME]"""
import sys, json
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source'); sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/source')
import best_pipeline_20261106 as BP
import ceiling_meta_cpu_20261103 as C
import deploy_hexkonan as DK
R = '/workspace/clear-ec-research/experiments/'; L = R + 'lane_cellroute_20261109/'
zb = np.load(sys.argv[1]); base = zb['final_hex']
if 'idx' in zb.files: assert (zb['idx'].astype(int) == BP.idx.astype(int)).all() and (zb['fold'] == BP.fold).all(), 'row order differs'
zf = np.load(L + 'feats/cellroute_feats.npz'); NM = list(zf['names']); Fr = zf['F'][:, [NM.index(c) for c in DK.NAMES]][BP.idx.astype(int)]
y, fold, good = BP.y, BP.fold, BP.good; X, m = DK.design(Fr, base); ok = y[:, 2] > 0; res = np.log(np.where(ok, y[:, 2], 1.0)) - np.log(base); P = base.copy()
for f in range(5):
    tr = (fold != f) & good & m & ok; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; A = (X[tr] - mu) / sd; w = np.linalg.solve(A.T @ A + 100.0 * np.eye(A.shape[1]), A.T @ (res[tr] - res[tr].mean()))
    q = base * np.exp(((X - mu) / sd) @ w); c = C.wmed(y[tr, 2] / q[tr], q[tr] / y[tr, 2]); c0 = C.wmed(y[tr, 2] / base[tr], base[tr] / y[tr, 2]); sel = (fold == f) & m; P[sel] = q[sel] * c / c0
ape = lambda p: np.where(ok, np.abs(p - y[:, 2]) / np.where(ok, y[:, 2], 1) * 100, np.nan); a, a0 = ape(P), ape(base); f04 = (fold == 0) | (fold == 4)
o = dict(base=sys.argv[1].split('/')[-1], hex=round(float(np.nanmean(a)), 4), base_hex=round(float(np.nanmean(a0)), 4), delta=round(float(np.nanmean(a) - np.nanmean(a0)), 4),
         folds=[round(float(np.nanmean(a[fold == f]) - np.nanmean(a0[fold == f])), 4) for f in range(5)], f04=round(float(np.nanmean(a[f04]) - np.nanmean(a0[f04])), 4), rows_corrected=int(m.sum()))
print(json.dumps(o), flush=True)
if len(sys.argv) > 2: np.savez_compressed(L + f'out/{sys.argv[2]}.npz', P=P, idx=BP.idx, fold=fold, note='cell-route HEX step on ' + o['base'])
