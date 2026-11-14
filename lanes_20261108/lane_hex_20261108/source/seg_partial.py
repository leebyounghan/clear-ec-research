"""Partial read of segmentation features on the frames done so far (HEX lane, 2026-11-08): ridge on the log residual of the
chain candidate, fit per outer fold on the other folds' segmented rows only, applied to segmented rows of that fold.
usage: seg_partial.py SEGNPZ(rel LANE) CHAINNPZ(rel LANE)"""
import sys, numpy as np
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'; sys.path.insert(0, f'{LANE}/source')
import chain as C
z = np.load(f'{LANE}/{sys.argv[2]}'); cand = {f: z[f'c{f}'] for f in range(5)}
F, nm = C.segfeat(f'{LANE}/{sys.argv[1]}'); has = np.isfinite(F[:, :8]).all(1) & (F[:, 3] >= 30)
y, fold, good, ok = C.y, C.fold, C.good, C.ok; lg = C.lg
print('segmented rows', int(np.isfinite(F[:, 0]).sum()), 'usable', int(has.sum()), 'usable in folds 0+4', int((has & C.L.S04).sum()))
def cor(G, alpha):
    G = np.asarray(G, float).reshape(len(y), -1); out = {}
    for f in range(5):
        lb = lg(cand[f]); X = np.c_[G, lb]; X[~np.isfinite(X)] = 0; tr = (fold != f) & good & ok & has
        t = lg(np.where(ok, y[:, 2], 1)) - lb; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Z = (X - mu) / sd; A = Z[tr]
        w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (t[tr] - t[tr].mean()))
        o = cand[f].copy(); o[has] = cand[f][has] * np.exp(Z[has] @ w + t[tr].mean()); out[f] = o
    return out
def h(P, r): r = r & ok; return float(np.mean(np.abs(P[r, 2] - y[r, 2]) / y[r, 2]) * 100)
P0 = C.final(cand); rows = {'f04': C.L.S04 & has, 'all5': has}
print('subset HEX before: f04 %.4f all5 %.4f' % (h(P0, rows['f04']), h(P0, rows['all5'])))
n = len(nm); hist = F[:, 8:15]; tot = hist.sum(1, keepdims=True) + 1e-9
SETS = {'h6': lg(F[:, 0] + 1e-3), 'h567': lg(F[:, :3] + 1e-3),
        'base': np.c_[lg(F[:, :3] + 1e-3), np.log(F[:, 3] + 1), lg(F[:, 4]), lg(F[:, 5]), F[:, 6]],
        'strict': lg(F[:, 15] + 1e-3), 'hist': hist / tot,
        'base_hist': np.c_[lg(F[:, :3] + 1e-3), np.log(F[:, 3] + 1), lg(F[:, 4]), lg(F[:, 5]), F[:, 6], hist / tot, lg(F[:, 15] + 1e-3)]}
for k, G in SETS.items():
    for a in (10.0, 100.0, 1000.0):
        P = C.final(cor(G, a))
        print('%-9s a%-6g subset f04 %+.4f all5 %+.4f | whole f04 %+.4f all5 %+.4f' % (k, a, h(P, rows['f04']) - h(P0, rows['f04']), h(P, rows['all5']) - h(P0, rows['all5']),
              h(P, C.L.S04) - h(P0, C.L.S04), h(P, C.allr) - h(P0, C.allr)), flush=True)
