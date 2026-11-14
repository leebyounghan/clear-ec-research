"""CV lane copy (2026-11-11) for the HEX lane's Vigueras-Guillen segmentation features (seg_vig_all.npz: the 17 CPSAM-route columns + vig_fc,
vig_roi_frac); stack file given as argument 3 (default the freeze candidate out/cv_final_20261110b.npz); the two Vigueras extras are added to the features (log vig_fc, vig_roi_frac). Earlier text: HEX lane's Cellpose-SAM segmentation cell-shape features (read only) as a CV correction on top of the CV lane best
(nested OOF). Ridge on the log CV residual, fit on folds != f, only on frames with >= 50 interior cells (others keep the
stack), features as the HEX lane's 'bh' set plus log stack CV; alpha 100 (HEX lane's pick). The scale is refit MAPE-optimally (weighted median) on the
training rows. Several segmentation runs are averaged where usable.
usage: seg_cv_20261109.py SEGFILE[,SEGFILE...] [ALPHA]"""
import sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import ceiling_meta_cpu_20261103 as C
E = BP.E; y = BP.y; fold = BP.fold; idx = BP.idx; N_ = len(y); good = BP.good
paths = sys.argv[1].split(','); ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0; STACK = sys.argv[3] if len(sys.argv) > 3 else 'cv_final_20261110b'
P0 = np.load(f'/workspace/clear-ec-research/experiments/lane_cv_20261108/out/{STACK}.npz')['P']
def segfeat(p):
    z = np.load(p); F = np.full((N_, z['F'].shape[1]), np.nan); pos = {int(f): r for r, f in enumerate(idx)}
    for f, v in zip(z['frames'], z['F']):
        if int(f) in pos: F[pos[int(f)]] = v
    return F
def usable(F): return np.isfinite(F[:, :15]).all(1) & (np.nan_to_num(F[:, 3]) >= 50)
Fs = [segfeat('/workspace/clear-ec-research/experiments/lane_hex_20261108/preds/' + p) for p in paths]; U = np.array([usable(F) for F in Fs])
S_ = np.zeros_like(Fs[0]); W_ = np.zeros_like(Fs[0])
for F, u in zip(Fs, U):
    v = u[:, None] & np.isfinite(F); S_ += np.where(v, F, 0.0); W_ += v
F = np.where(W_ > 0, S_ / np.maximum(W_, 1), np.nan); m = U.any(0)
F[~np.isfinite(F[:, 15]), 15] = np.nanmedian(F[m, 15])
lg = lambda a: np.log(np.maximum(a, 1e-6))
G = np.c_[lg(F[:, [0, 1, 2]] + 1e-3), np.log(F[:, 3] + 1), lg(F[:, 4]), lg(F[:, 5]), F[:, 6]]
hist = F[:, 8:15] / (F[:, 8:15].sum(1, keepdims=True) + 1e-9); GH = np.c_[G, hist, lg(F[:, 15] + 1e-3)]
if F.shape[1] >= 19:
    for j in (17, 18): F[~np.isfinite(F[:, j]), j] = np.nanmedian(F[m, j])
    GH = np.c_[GH, lg(F[:, 17]), F[:, 18]]
ok = y[:, 1] > 0; ly = np.log(np.where(ok, y[:, 1], 1.0)); res = ly - np.log(P0[:, 1])
mm = m & ok & np.isfinite(res)
print('usable rows', int(m.sum()), 'per fold', [int((m & (fold == f)).sum()) for f in range(5)], flush=True)
print('r(log seg area_cv, log CV label) %.3f | r with stack residual %.3f' % (np.corrcoef(lg(F[mm, 4]), ly[mm])[0, 1], np.corrcoef(lg(F[mm, 4]), res[mm])[0, 1]), flush=True)
P = P0.copy()
for f in range(5):
    X = np.c_[GH, np.log(P0[:, 1])]; X = np.where(m[:, None], X, 0.0); tr = (fold != f) & good & mm
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Z = (X - mu) / sd; A = Z[tr]; t = res
    w = np.linalg.solve(A.T @ A + ALPHA * np.eye(A.shape[1]), A.T @ (t[tr] - t[tr].mean())); d = Z @ w
    q = P0[:, 1] * np.exp(d); c = C.wmed(y[tr, 1] / q[tr], q[tr] / y[tr, 1]); c0 = C.wmed(y[tr, 1] / P0[tr, 1], P0[tr, 1] / y[tr, 1])
    sel = (fold == f) & m; P[sel, 1] = q[sel] * c / c0  # rescale relative to the stack's own optimum on the same rows
a, a0, ar = E.ape(P, y)[:, 1], E.ape(P0, y)[:, 1], E.ape(BP.REF_BEST, y)[:, 1]
print('SEG CV correction (alpha %g): CV %.4f vs stack %.4f (d %.4f) vs best pipeline %.4f (d %.4f) | per fold vs stack %s | folds 0+4 vs stack %.4f' % (ALPHA, np.nanmean(a), np.nanmean(a0), np.nanmean(a) - np.nanmean(a0), np.nanmean(ar), np.nanmean(a) - np.nanmean(ar), [round(float(np.nanmean(a[fold == f]) - np.nanmean(a0[fold == f])), 4) for f in range(5)], np.nanmean(a[np.isin(fold, [0, 4])]) - np.nanmean(a0[np.isin(fold, [0, 4])])), flush=True)
np.savez_compressed(f'/workspace/clear-ec-research/experiments/lane_cv_20261108/out/stack_segvig_{STACK}.npz', P=P)
