"""Residual correlations for the Vigueras segmentation member (HEX lane, 2026-11-10). Report only, nothing is chosen here.
chain residual  = log y - log(main chain final HEX) (preds/chain_mtcnn5fix.npz).
deviation       = log(main chain + segvig step) - log(main chain), the step's nested correction (segm recipe), any keep/drop.
standalone      = nested ridge (alpha 100 fixed, fit on the other folds' usable rows) of log y on the member's segm features alone;
                  its residual is compared with the chain residual and with the same standalone fit on the CPSAM features.
usage: vig_resid.py [seg_vig_all.npz]"""
import sys, json
import numpy as np
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
sys.path.insert(0, LANE + '/source')
import chain as C

N, fold = C.N, C.fold; y = C.y[:, 2]; ok = C.ok & C.good


def gfeat(F):
    m = C.seg_usable(F, 50); F = F.copy(); F[~np.isfinite(F[:, 15]), 15] = np.nanmedian(F[m, 15])
    G = np.c_[C.lg(F[:, [0, 1, 2]] + 1e-3), np.log(F[:, 3] + 1), C.lg(F[:, 4]), C.lg(F[:, 5]), F[:, 6]]
    hist = F[:, 8:15] / (F[:, 8:15].sum(1, keepdims=True) + 1e-9); return np.c_[G, hist, C.lg(F[:, 15] + 1e-3)], m


def standalone(F, alpha=100.0):
    X, m = gfeat(F); t = np.log(np.where(ok, y, 1)); out = np.full(N, np.nan)
    for f in range(5):
        tr = (fold != f) & ok & m; te = (fold == f) & m; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; A = (X[tr] - mu) / sd
        w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (t[tr] - t[tr].mean())); out[te] = (X[te] - mu) / sd @ w + t[tr].mean()
    return out, m


def r(a, b, rows): return round(float(np.corrcoef(a[rows], b[rows])[0, 1]), 4)


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else LANE + '/preds/seg_vig_all.npz'
    z = np.load(LANE + '/preds/chain_mtcnn5fix.npz'); cand = {f: z[f'c{f}'] for f in range(5)}; base = C.final(cand)[:, 2]
    res = np.log(np.where(ok, y, 1)) - np.log(base)
    Fv = C.segfeat(path)[0]; new, picks = C.step_segm(cand, path, name='segvig'); dev = np.log(C.final(new)[:, 2]) - np.log(base)
    sv, mv = standalone(Fv); sc, mc = standalone(C.segfeat(LANE + '/preds/seg_all.npz')[0])
    rows = ok & mv; both = rows & mc; lt = np.log(np.where(ok, y, 1))
    rep = dict(picks=picks, usable_rows=int(mv.sum()), usable_rows_cpsam=int(mc.sum()),
               r_deviation_vs_chain_residual=dict(pooled=r(dev, res, rows), per_fold=[r(dev, res, rows & (fold == f)) for f in range(5)]),
               deviation_sd=round(float(dev[rows].std()), 4), chain_residual_sd=round(float(res[rows].std()), 4),
               r_standalone_residual_vs_chain_residual=r(lt - sv, res, rows),
               r_standalone_residual_vig_vs_cpsam=r(lt - sv, lt - sc, both),
               r_standalone_prediction_vig_vs_cpsam=r(sv, sc, both),
               standalone_hex_mape=dict(vig=round(float(np.mean(np.abs(np.exp(sv[rows]) - y[rows]) / y[rows]) * 100), 3),
                                        cpsam_same_rows=round(float(np.mean(np.abs(np.exp(sc[both]) - y[both]) / y[both]) * 100), 3),
                                        vig_same_rows=round(float(np.mean(np.abs(np.exp(sv[both]) - y[both]) / y[both]) * 100), 3),
                                        chain_same_rows=round(float(np.mean(np.abs(base[rows] - y[rows]) / y[rows]) * 100), 3)))
    print(json.dumps(rep, indent=1), flush=True); json.dump(rep, open(LANE + '/logs/vig_resid.json', 'w'), indent=1)
