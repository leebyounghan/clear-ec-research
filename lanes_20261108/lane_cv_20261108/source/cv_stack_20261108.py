"""CV lane stack (2026-11-08, PI rule: keep what lowers CV). CV-column corrections added to the best pipeline, folds 0 and 4 only (the members exist
only there). Weights cross-fit: each fold's weights are chosen on the other fold's rows (greedy forward, grid), never on the rows reported.
Components: g10 = 10-seed graph members instead of 3-seed (all folds); dino = DINOv3 B/16 two seeds, w (log new - log recipe A TTA) as in read_two_seeds;
frozen heads (lane GPU 2 own encoders): w (lg arm - lg std); lane graph runs given on the command line: w (run CV - shipped graph CV).
usage: cv_stack_20261108.py [graph_run ...]  -> prints the chain and saves out/cv_stack.npz (correction dict, CV column)."""
import sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
X = Path('/workspace/clear-ec-research/experiments'); E = BP.E; y = BP.y; fold = BP.fold; pos = {int(i): n for n, i in enumerate(BP.idx)}; A = BP._b['A']; N_ = len(y)
F2 = (0, 4); rows2 = np.isin(fold, F2)
def zeros(): return {f: np.zeros((N_, 3)) for f in range(5)}
def cvcorr(vec):  # vec: N CV log correction (zero outside its rows) -> correction dict applied for every fold model
    c = zeros()
    for f in range(5): c[f][:, 1] = vec
    return c
comps = {}
G = X / 'meta_ceiling_20261103/sweep/graph_sweep'; g10 = [np.load(G / f'd64_l5_{q}_s10.npz') for q in ('noctx', 'pre47', 'pre48')]
c = zeros()
for f in range(5): c[f][:, 1] = np.mean([z[f'f{f}'][:, 1] for z in g10], 0) - BP.GRAPH[f][:, 1]
comps['g10'] = c
def last5(root, f, fam='dinov3_b16'):
    run = X / root / 'runs' / f'fold{f}' / fam; out = []
    for e in range(16, 21):
        with np.load(run / f'predictions_epoch_{e:02d}.npz', allow_pickle=True) as z: out.append((z['indices'].astype(int), z['pred'].astype(np.float64)))
    return np.array([pos[int(i)] for i in out[0][0]]), np.mean([np.log(p) for _, p in out], 0)
v = np.zeros(N_)
for f in F2:
    r, a = last5(f'lane_gpu3_20261106/runs/dinov3_b16_f{f}_s46', f); r2, b_ = last5(f'lane_gpu2_20261106/runs/dinov3_b16_f{f}_s47', f); assert np.array_equal(r, r2)
    v[r] = (a[:, 1] + b_[:, 1]) / 2 - np.log(A[r, 1])
comps['dino'] = cvcorr(v)
R = X / 'lane_gpu2_20261106/reads/own'
for arm in ('spectrum', 'autocorr', 'cov_sqrt', 'head_quantile', 'cov', 'multiscale', 'winhead', 'all'):
    v = np.zeros(N_)
    for f in F2:
        za, zs = np.load(R / f'{arm}_f{f}.npz'), np.load(R / f'std_f{f}.npz'); assert np.array_equal(za['rows'], zs['rows']) and (fold[za['rows']] == f).all()
        v[za['rows']] = za['lg'][:, 1] - zs['lg'][:, 1]
    comps['fz_' + arm] = cvcorr(v)
LO = X / 'lane_cv_20261108/graph_runs'
for g in sys.argv[1:]:
    runs = g.split('+'); Z = [np.load(LO / f'{q}.npz') for q in runs]; v = np.zeros(N_)
    for f in F2: v[fold == f] = np.mean([z[f'f{f}'][fold == f, 1] for z in Z], 0) - BP.GRAPH[f][fold == f, 1]
    comps['graph_' + g] = cvcorr(v)
def cv_on(cor, mask):
    p = BP.pipe(BP.add(BP.BEST, cor)); a = E.ape(p, y)[:, 1]; return float(np.nanmean(a[mask])), p
GRID = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
ref_cv = {f: float(np.nanmean(E.ape(BP.REF_BEST, y)[fold == f, 1])) for f in F2}
# solo reads (information): each component alone, cross-fit weight
print('reference CV fold0 %.4f fold4 %.4f both %.4f' % (ref_cv[0], ref_cv[4], float(np.nanmean(E.ape(BP.REF_BEST, y)[rows2, 1]))), flush=True)
final = {}; chain = {f: zeros() for f in F2}; picks = {f: [] for f in F2}
for f in F2:
    other = [g for g in F2 if g != f][0]; cur = zeros(); m_o = fold == other
    base_o = cv_on(cur, m_o)[0]
    for nm, cc in comps.items():
        best = (base_o, 0.0)
        for w in GRID[1:]:
            t = BP.add(cur, {ff: w * cc[ff] for ff in range(5)}); s = cv_on(t, m_o)[0]
            if s < best[0] - 1e-6: best = (s, w)
        if best[1] > 0: cur = BP.add(cur, {ff: best[1] * cc[ff] for ff in range(5)}); base_o = best[0]
        picks[f].append((nm, best[1]))
    chain[f] = cur
P = np.full_like(BP.H, np.nan)
for f in F2: P[fold == f] = BP.pipe(BP.add(BP.BEST, chain[f]))[fold == f]
a, a0 = E.ape(P, y), E.ape(BP.REF_BEST, y)
print('picks (weights chosen on the other fold):', json.dumps(picks), flush=True)
print('STACK CV folds0+4 %.4f vs %.4f (d %.4f) | fold0 d %.4f fold4 d %.4f | CD d %.4f HEX d %.4f' % (np.nanmean(a[rows2, 1]), np.nanmean(a0[rows2, 1]), np.nanmean(a[rows2, 1]) - np.nanmean(a0[rows2, 1]), np.nanmean(a[fold == 0, 1]) - ref_cv[0], np.nanmean(a[fold == 4, 1]) - ref_cv[4], np.nanmean(a[rows2, 0]) - np.nanmean(a0[rows2, 0]), np.nanmean(a[rows2, 2]) - np.nanmean(a0[rows2, 2])), flush=True)
for nm, cc in comps.items():  # each alone, cross-fit
    Q = np.full_like(BP.H, np.nan); ws = {}
    for f in F2:
        other = [g for g in F2 if g != f][0]; m_o = fold == other; sc = {w: cv_on({ff: w * cc[ff] for ff in range(5)}, m_o)[0] for w in GRID}; w = min(sc, key=sc.get); ws[f] = w
        Q[fold == f] = BP.pipe(BP.add(BP.BEST, {ff: w * cc[ff] for ff in range(5)}))[fold == f]
    print('  alone %-24s w %s  dCV %.4f' % (nm, ws, np.nanmean(E.ape(Q, y)[rows2, 1]) - np.nanmean(a0[rows2, 1])), flush=True)
np.savez_compressed(X / 'lane_cv_20261108/out/cv_stack.npz', **{f'f{f}': chain[f][f].astype(np.float32) for f in F2}, P=P)
