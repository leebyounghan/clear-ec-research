"""Why does the simulation network's direct HEX prediction flip sign on real frames (2026-11-05)? In held-out simulation r 0.82; on real frames r -0.29 with the labels, while plain hex6 of the same nodes gives +0.47 to +0.51. Per frame summaries of the graph input, on simulation and on real frames: mean confidence, node count, share of six sides among bounded cells, log area SD, unbounded share. For each: correlation with the network output and with the true target (simulated truth or real label). A summary whose relation to the output and the target differs between the worlds is the gap."""
import json, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
E = C.E; b = E.load_base(); y, idx = b['y'], b['idx']; O = C.X / 'meta_ceiling_20261103/sweep'; ly = np.log(np.where(y > 0, y, np.nan))
def summ(X, OF, n):
    S = np.full((n, 7), np.nan)
    for i in range(n):
        a, c = OF[i], OF[i + 1]
        if c - a < 20: continue
        x = X[a:c]; m = x[:, 6] == 0; s = x[:, 4]
        S[i] = (x[:, 0].mean(), np.log(c - a), (s[m] == 6).mean() if m.sum() else np.nan, x[m, 1].std() if m.sum() > 2 else np.nan, x[:, 6].mean(), (x[:, 0] < 0.5).mean(), x[:, 2].std())
    return S
NM = ['mean conf', 'log nodes', 'six-side share', 'log area SD', 'unbounded', 'low-conf share', 'log dist SD']
def cor(a, c):
    s = np.isfinite(a) & np.isfinite(c); return round(float(np.corrcoef(a[s], c[s])[0, 1]), 3)
z = np.load(O / 'sim_graphs_v1.npz'); T = z['T']; Ss = summ(z['X'], z['offs'], len(T)); g = np.load(O / 'cell_graphs.npz'); Sr = summ(g['X'], g['offs'], len(g['offs']) - 1)[idx]; P = np.load(O / 'sim_pred' / 'sim_l5__cell_graphs.npz')['pred'][idx].astype(np.float64)
print('statistic: sim [r with true HEX, true CV, true CD] | real [r with label HEX, CV, CD] [r with net output HEX, CV]; means sim vs real', flush=True)
for j, n in enumerate(NM):
    print(n.ljust(16), [cor(Ss[:, j], T[:, k]) for k in range(3)], '|', [cor(Sr[:, j], ly[:, k]) for k in (2, 1, 0)], [cor(Sr[:, j], P[:, k]) for k in (0, 1)], 'means', round(float(np.nanmean(Ss[:, j])), 3), round(float(np.nanmean(Sr[:, j])), 3), 'SD', round(float(np.nanstd(Ss[:, j])), 3), round(float(np.nanstd(Sr[:, j])), 3), flush=True)
print('real net outputs: HEX vs CV', cor(P[:, 0], P[:, 1]), 'HEX vs CD', cor(P[:, 0], P[:, 2]), 'output exp p5 50 95 HEX', np.round(np.exp(np.nanpercentile(P[:, 0], (5, 50, 95))), 3).tolist(), 'CV', np.round(np.exp(np.nanpercentile(P[:, 1], (5, 50, 95))), 3).tolist(), 'CD', np.round(np.exp(np.nanpercentile(P[:, 2], (5, 50, 95))), 0).tolist(), flush=True)
print('sim targets: HEX vs CV', cor(T[:, 0], T[:, 1]), 'HEX vs CD', cor(T[:, 0], T[:, 2]), 'CV vs CD', cor(T[:, 1], T[:, 2]), '| real labels: HEX vs CV', cor(ly[:, 2], ly[:, 1]), 'HEX vs CD', cor(ly[:, 2], ly[:, 0]), 'CV vs CD', cor(ly[:, 1], ly[:, 0]), flush=True)
