"""mtcnn5 second-seed check inside the joint nested ridge (HEX lane, 2026-11-11, coordinator). Seed 47 exists on some folds only
(0 and 4 first), so the mtcnn5 column is seed 46 everywhere except on those folds' rows, where it is seed 47 ('s47') or the mean
log of seeds 46 and 47 ('mean'); 'none' drops the column. Each variant is the 39-column freeze ridge on the nested ap1 base
(joint_fit.run); reported: five folds, the covered folds, and the mtcnn5 column's standardized weight per outer fold.
usage: joint_mt_read.py FOLDS(e.g. 0,4) OUT.json"""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
sys.path.insert(0, LANE + '/source')
import joint_fit as JF
C = JF.C; fold, y, ok, good = C.fold, C.y[:, 2], C.ok, C.good
FS = [int(q) for q in sys.argv[1].split(',')]
G = JF.columns(); base = C.STEPS['ap1']({f: C.L.HB[f].copy() for f in range(5)})[0]
l46 = G['mtcnn5'][:, 0].copy(); l47 = np.log(C.part_hex(C.MT47(tuple(FS))))
rows = np.isin(fold, FS); assert np.isfinite(l47[rows]).all()
out = {}
for v in ('s46', 's47', 'mean', 'none'):
    col = l46.copy()
    if v == 's47': col[rows] = l47[rows]
    if v == 'mean': col[rows] = 0.5 * (l46[rows] + l47[rows])
    names = [k for k in G if k not in ('gnn4647', 'gnn4647x')]; GG = dict(G); GG['mtcnn5'] = col[:, None]   # the 39-column freeze ridge (attention-pool column included, 11-11 11:40 EST)
    if v == 'none': names = [k for k in names if k != 'mtcnn5']
    X = np.concatenate([GG[k] for k in names], 1); r, cand = JF.run(base, X)
    P = C.final(cand)[:, 2]; E = np.abs(P - y) / np.where(ok, y, 1); r['covered_folds'] = round(float(E[ok & rows].mean() * 100), 5)
    if v != 'none':
        j = sum(GG[k].shape[1] for k in names[:names.index('mtcnn5')]); t_all = JF.lg(np.where(ok, y, 1)); w = []
        for f in range(5):
            lb = np.log(base[f]); Xf = np.c_[X, lb]; med = np.nanmedian(np.where(np.isfinite(Xf), Xf, np.nan)[(fold != f) & good], 0); Xf = np.where(np.isfinite(Xf), Xf, med)
            tr = (fold != f) & good & ok; mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-9; A_ = ((Xf - mu) / sd)[tr]; tt = (t_all - lb)[tr]
            w.append(round(float(np.linalg.solve(A_.T @ A_ + r['alpha_picks'][f] * np.eye(A_.shape[1]), A_.T @ (tt - tt.mean()))[j]), 4))
        r['mtcnn5_weight_per_fold'] = w
    out[v] = r; print(v, json.dumps(r), flush=True)
json.dump(dict(folds=FS, results=out), open(sys.argv[2], 'w'), indent=1)
