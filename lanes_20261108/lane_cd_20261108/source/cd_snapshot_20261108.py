"""Snapshot averaging on CD (2026-11-08): stored per-epoch single-view predictions, no GPU.
(1) DINOv3 s46 member: epochs 16-20 vs 8-20 (and fold-1 TTA swapped in where present). (2) Recipe A seeds 46/47: tta4 epoch 20 (current A2) vs
log mean of tta4 e20 with single-view epochs 8-19. Read: nested five-fold three-weight blend P1/A2/F + stretch (cd_read machinery)."""
import numpy as np, json
import cd_read_20261108 as R
from cd_common import *
pos = R.pos
def recipe_epochs(seed, epochs):
    out = np.full(len(yc), np.nan)
    for f, root in C.RECIPE[seed].items():
        d = X / root / 'runs' / f'fold{f}' / 'cpdino_vitb'
        L = []
        for e in epochs:
            z = np.load(d / f'predictions_epoch_{e:02d}.npz'); p = np.full(len(yc), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; L.append(np.log(p))
        m = np.exp(np.mean(L, 0)); out[fold == f] = m[fold == f]
    return out
def fam_epochs(key, epochs):
    fam, seed, tag = R.MEMBERS[key]; out = np.full(len(yc), np.nan)
    for f, r in R.roots(fam, seed, tag).items():
        L = []
        for e in epochs:
            z = np.load(X / r / f'runs/fold{f}/{fam}/predictions_epoch_{e:02d}.npz'); p = np.full(len(yc), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; L.append(np.log(p))
        m = np.exp(np.mean(L, 0)); out[fold == f] = m[fold == f]
    return out
def read(name, P1, A2, F):
    rows = np.isfinite(F) & np.isfinite(A2); out = np.full(len(yc), np.nan)
    Lc = [np.log(P1), np.log(A2), np.log(np.where(np.isfinite(F), F, 1.))]
    Q = {wt: np.exp(sum(w * l for w, l in zip(wt, Lc))) for wt in R.G}
    for f in range(5):
        tr = rows & (fold != f); best = None
        for wt, q in Q.items():
            prm = fit_stretch1(q, tr & good); e = cd(apply1(q, prm), tr)
            if best is None or e < best[0]: best = (e, wt, prm)
        out[fold == f] = apply1(Q[best[1]][fold == f], best[2])
    return rep(name, out, REF, rows=rows)
sc = R.sc
A2_cur = R.A2
singles = {s: recipe_epochs(s, range(8, 20)) for s in (46, 47)}
A2_e8 = np.exp(np.mean([np.log(0.5 * (R.seeds[46][:, 0] + R.seeds[47][:, 0]) * sc)] + [np.log(0.5 * (singles[46] + singles[47]) * sc)], 0))   # half weight on tta4 e20, half on the e8-19 snapshot
A2_snap = 0.5 * (np.exp(np.mean([np.log(R.seeds[46][:, 0])] * 1 + [np.log(singles[46])] * 1, 0)) + np.exp(np.mean([np.log(R.seeds[47][:, 0])] + [np.log(singles[47])], 0))) * sc
print('A2 alone CD: current', round(cd(A2_cur), 4), 'half tta/half snapshot', round(cd(A2_e8), 4), 'per-seed', round(cd(A2_snap), 4), flush=True)
F16 = R.member('dinov3_s46'); F8 = fam_epochs('dinov3_s46', range(8, 21))
print('DINOv3 alone CD: e16-20', round(cd(F16), 4), 'e8-20', round(cd(F8), 4), flush=True)
read('base: A2 tta e20, F e16-20', R.P1, A2_cur, F16)
read('F e8-20', R.P1, A2_cur, F8)
read('A2 snapshot (tta e20 + e8-19), F e16-20', R.P1, A2_snap, F16)
read('A2 snapshot, F e8-20', R.P1, A2_snap, F8)
B_snap = 0.5 * A2_snap + 0.5 * R.P1
rep('B with A2 snapshot, stretch only (no family)', stretch(B_snap), REF)
