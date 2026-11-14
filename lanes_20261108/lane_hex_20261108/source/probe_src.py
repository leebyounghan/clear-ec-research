"""HEX by label source (HEX lane, 2026-11-08). Two ID formats mark two sources ('NNNN-NNNN ODCN/OSCN' vs 'NNNN-YY').
Does a stretch fitted per source (slope and scale, nested: fit on the other folds' rows of that source) lower HEX on top of the
chain? First with the ID-format source (oracle), then with the source predicted from image statistics (nested logistic).
usage: probe_src.py CHAINNPZ(rel LANE)"""
import sys, numpy as np
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'; sys.path.insert(0, f'{LANE}/source')
import chain as C
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
NS, BP, y, fold, good, ok = C.NS, C.BP, C.y, C.fold, C.good, C.ok
z = np.load(f'{LANE}/{sys.argv[1]}'); cand = {f: z[f'c{f}'] for f in range(5)}
ids = [str(q) for q in BP.ids]; src = np.array(['CN' in i for i in ids])
def h(P, r): r = r & ok; return float(np.mean(np.abs(P[r, 2] - y[r, 2]) / y[r, 2]) * 100)
def final_grp(cand, g):
    P = C.L.REF.copy()
    for f in range(5):
        Cc = BP.H.copy(); Cc[:, 2] = cand[f]
        for s in np.unique(g):
            prm = NS.fit_stretch(Cc, y, (fold != f) & good & (g == s)); m = (fold == f) & (g == s); P[m, 2] = NS.apply_stretch(Cc[m], prm)[:, 2]
    return P
def final_scale(cand, g):
    """one slope from all rows, scale per group"""
    P = C.L.REF.copy()
    for f in range(5):
        Cc = BP.H.copy(); Cc[:, 2] = cand[f]; prm = NS.fit_stretch(Cc, y, (fold != f) & good); Q = NS.apply_stretch(Cc, prm)[:, 2]
        for s in np.unique(g):
            tr = (fold != f) & good & ok & (g == s); c = NS.C.wmed(y[tr, 2] / Q[tr], Q[tr] / y[tr, 2])
            m = (fold == f) & (g == s); P[m, 2] = Q[m] * c
    return P
P0 = C.final(cand); S04 = C.L.S04; A = C.allr
print('rows: old-format', int(src.sum()), 'new-format', int((~src).sum()))
for nm, m in (('old', src), ('new', ~src)):
    r = np.log(y[m & ok, 2]) - np.log(P0[m & ok, 2])
    print(f'  {nm}: HEX all5 {h(P0, m):.4f}, mean log residual {r.mean():+.4f}, residual sd {r.std():.4f}')
for nm, fn in (('stretch per source (ID)', final_grp), ('scale per source (ID)', final_scale)):
    P = fn(cand, src.astype(int)); print(f'{nm:32s} f04 {h(P, S04):.4f} ({h(P, S04) - h(P0, S04):+.4f})  all5 {h(P, A):.4f} ({h(P, A) - h(P0, A):+.4f})', flush=True)
# source from the image: nested logistic on 25 image statistics
zi = np.load(f'{LANE}/feats/img_rich.npz'); X = zi['F'] if 'F' in zi.files else np.c_[[zi[k] for k in zi.files if k not in ('idx', 'names')]].T
X = np.asarray(X, float); X[~np.isfinite(X)] = 0; X = (X - X.mean(0)) / (X.std(0) + 1e-9)
ps = np.zeros(len(y))
for f in range(5):
    tr = fold != f; m = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], src[tr]); ps[fold == f] = m.predict_proba(X[fold == f])[:, 1]
print('image source classifier (nested, 25 image stats): AUC %.4f, accuracy %.4f' % (roc_auc_score(src, ps), np.mean((ps > 0.5) == src)))
gi = (ps > 0.5).astype(int)
for nm, fn in (('stretch per source (image)', final_grp), ('scale per source (image)', final_scale)):
    P = fn(cand, gi); print(f'{nm:32s} f04 {h(P, S04):.4f} ({h(P, S04) - h(P0, S04):+.4f})  all5 {h(P, A):.4f} ({h(P, A) - h(P0, A):+.4f})', flush=True)
np.save(f'{LANE}/feats/src_image_prob.npy', ps)
