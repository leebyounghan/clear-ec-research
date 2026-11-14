"""Simulation-pretrained embeddings read by ridge, done properly (2026-11-05). First read averaged embeddings across seeds, whose axes do not align, and still gave -0.057 alone and -0.011 on top of the graph ensemble. Here: per seed, embeddings of each detector's graphs from the saved network; for one seed, detectors share axes, so they average; ridge per metric on the log residual, LOFO, per seed; corrections average over seeds. No label touches the embedding. Reads: alone on B + hex6 + stretch; on top of the label-trained graph ensemble; mean of the two corrections; ridge strength reported for all, and chosen in each fold from the other four. Writes sweep/sim_pred/<NAME>_embed.npz. usage: sim_embed_eval_20261105.py NAME. One GPU."""
import json, sys, warnings
from pathlib import Path
import numpy as np
import torch
from torch import nn
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import stretch_nested_20261103 as N
E = C.E; K = list(C.K); b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']; g = E.donors(b['ids']); O = C.X / 'meta_ceiling_20261103/sweep'; NAME = sys.argv[1]; N_ = len(y); FILES = ('cell_graphs.npz', 'cell_graphs_pre47.npz', 'cell_graphs_pre48.npz')
class Net(nn.Module):
    def __init__(self, d, layers):
        super().__init__(); self.emb = nn.Linear(9, d); self.msg = nn.ModuleList(nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d)) for _ in range(layers)); self.upd = nn.ModuleList(nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU()) for _ in range(layers)); self.drop = nn.Dropout(0.1); self.head = nn.Sequential(nn.Linear(2 * d, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 3))
    def pool(self, x, e, frame, nf):
        h = self.emb(x); src = torch.cat([e[:, 0], e[:, 1]]); dst = torch.cat([e[:, 1], e[:, 0]])
        for m, u in zip(self.msg, self.upd):
            ms = m(torch.cat([h[src], h[dst]], 1)); ag = torch.zeros_like(h).index_add_(0, dst, ms); c = torch.zeros(len(h), device=h.device).index_add_(0, dst, torch.ones(len(dst), device=h.device)).clamp_min(1)[:, None]; h = h + self.drop(u(torch.cat([h, ag / c], 1)))
        cnt = torch.zeros(nf, device=h.device).index_add_(0, frame, torch.ones(len(frame), device=h.device)).clamp_min(1)[:, None]; mean = torch.zeros(nf, h.shape[1], device=h.device).index_add_(0, frame, h) / cnt; var = torch.zeros(nf, h.shape[1], device=h.device).index_add_(0, frame, (h - mean[frame]) ** 2) / cnt; return torch.cat([mean, (var + 1e-6).sqrt()], 1)
def norm(X):
    F = X.clone(); F[:, 4] = (F[:, 4] - 6) / 2; F[:, 7:9] = F[:, 7:9] * 2 - 1; return F
def embed(net, gz, d):
    X = norm(torch.from_numpy(gz['X'][:, :9])).cuda(); Ed = torch.from_numpy(gz['E'].astype(np.int64)).cuda(); OF, EO = gz['offs'], gz['eoff']; n = len(OF) - 1; Z = np.full((n, 2 * d), np.nan, np.float32); net.eval()
    with torch.no_grad():
        for s in range(0, n, 256):
            xs, es, fr, keep = [], [], [], []; base = 0
            for i in range(s, min(s + 256, n)):
                a, c = OF[i], OF[i + 1]
                if c == a: continue
                xs.append(X[a:c]); es.append(Ed[EO[i]:EO[i + 1]] - a + base); fr.append(torch.full((c - a,), len(keep), device=X.device)); base += c - a; keep.append(i)
            if keep: Z[keep] = net.pool(torch.cat(xs), torch.cat(es), torch.cat(fr), len(keep)).cpu().numpy()
    return Z
ckpts = sorted((O / 'sim_pred').glob(f'{NAME}_s[0-9]*.pt')); EMB = []
GZ = [np.load(O / f) for f in FILES]
for p in ckpts:
    ck = torch.load(p, map_location='cpu', weights_only=False); a = ck['args']; net = Net(a['d'], a['layers']).cuda(); net.load_state_dict(ck['model']); Zs = [embed(net, gz, a['d'])[idx] for gz in GZ]; EMB.append(np.nanmean(Zs, 0).astype(np.float64)); print('embedded', p.name, flush=True); del net
gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); a_, b_, w_ = -0.2084, 0.5989, 0.28; have = gf['f_active10_valid'] & ~gf['seen_leaked'] & np.isfinite(gf['f_active10']); good = ~gf['seen_leaked']; f6 = gf['f_active10'].astype(float); H = B.copy(); H[have, 2] = np.exp((1 - w_) * np.log(B[have, 2]) + w_ * (a_ + b_ * f6[have]))
R = [np.load(O / 'graph_sweep' / f'{q}.npz') for q in ('d64_l5_noctx_s10', 'd64_l5_pre47_s3', 'd64_l5_pre48_s3')]; GR = {f: np.mean([z[f'f{f}'] for z in R], 0).astype(np.float64) for f in range(5)}; MASK = np.array([0.0, 1.0, 1.0])
def pipe(cor):
    P = np.full_like(H, np.nan)
    for f in range(5):
        C_ = H * np.exp(cor[f] * MASK); P[fold == f] = N.apply_stretch(C_[fold == f], N.fit_stretch(C_, y, (fold != f) & good))
    return P
ZERO = {f: np.zeros_like(H) for f in range(5)}; REF = pipe(ZERO); REFG = pipe(GR)
def ridge(Z, base_of, lam):
    """Correction per fold: ridge on the log residual of base_of(f), fit on the other folds."""
    out = {}
    for f in range(5):
        tr = (fold != f) & good; base = base_of(f); c = np.zeros_like(H); ok = np.isfinite(Z).all(1); Zf = np.where(ok[:, None], Z, 0)
        for j in (1, 2):
            q = tr & (y[:, j] > 0) & ok; t = np.log(y[q, j]) - np.log(base[q, j]); mu = Zf[q].mean(0); sd = Zf[q].std(0) + 1e-6; A = (Zf[q] - mu) / sd; w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (t - t.mean())); c[:, j] = np.where(ok, ((Zf - mu) / sd) @ w + t.mean(), 0)
        out[f] = c
    return out
def avg(L): return {f: np.mean([q[f] for q in L], 0) for f in range(5)}
def add(a, c): return {f: a[f] + c[f] for f in range(5)}
def report(nm, p, ref):
    pb = E.paired_bootstrap(E.ape(p, y), E.ape(ref, y), g); s, s0 = E.official(E.ape(p, y)), E.official(E.ape(ref, y)); per = [round(E.official(E.ape(p[fold == f], y[fold == f]))['mean'] - E.official(E.ape(ref[fold == f], y[fold == f]))['mean'], 3) for f in range(5)]; o = dict(score=round(s['mean'], 3), CV=round(s['CV'], 3), HEX=round(s['HEX'], 3), per_delta={k: round(s[k] - s0[k], 3) for k in K}, delta=round(pb['delta'], 4), ci=[round(v, 4) for v in pb['ci95']], folds=per); print(nm, json.dumps(o), flush=True); return o
rep = {}; Hb = lambda f: H; Gb = lambda f: H * np.exp(GR[f] * MASK); keep = {}
for lam in (300.0, 1000.0, 3000.0, 10000.0, 30000.0):
    A1 = avg([ridge(Z, Hb, lam) for Z in EMB]); rep[f'alone_{int(lam)}'] = report(f'seeds averaged, ridge {int(lam)}, alone', pipe(A1), REF); A2 = avg([ridge(Z, Gb, lam) for Z in EMB]); rep[f'ongraph_{int(lam)}'] = report(f'seeds averaged, ridge {int(lam)}, on graph ensemble', pipe(add(GR, A2)), REFG); M = {f: 0.5 * (A1[f] + GR[f]) for f in range(5)}; rep[f'mean_{int(lam)}'] = report(f'mean of ridge {int(lam)} and graph corrections', pipe(M), REFG); keep[lam] = (A1, A2)
print('single seed, ridge 3000:', [report(f'seed {k}', pipe(ridge(Z, Hb, 3000.0)), REF)['delta'] for k, Z in enumerate(EMB)], flush=True)
np.savez_compressed(O / 'sim_pred' / f'{NAME}_embed.npz', **{f's{k}': Z.astype(np.float16) for k, Z in enumerate(EMB)}); json.dump(rep, open(O / 'sim_pred' / f'{NAME}_embed_eval.json', 'w'), indent=1); print('done', flush=True)
