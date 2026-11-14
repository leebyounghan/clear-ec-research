"""Cell-graph network pretrained on simulated tessellations (2026-11-05). Same architecture as the label-trained graph network; input the same nine node features; target the exact log HEX, log area CV and log CD of the true cells in the windows. Held-out simulated geometries read how much it denoises, next to the plain statistic on the same noisy nodes. Then it reads every real frame: predictions and the pooled embedding, no label touched. usage: sim_graph_train_20261105.py NAME --data sim_graphs_v1.npz --real a.npz,b.npz [--layers 5 --d 64 --epochs 20 --seeds 3]. One GPU."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--data', default='sim_graphs_v1.npz'); ap.add_argument('--real', default='cell_graphs.npz,cell_graphs_pre47.npz,cell_graphs_pre48.npz'); ap.add_argument('--layers', type=int, default=5); ap.add_argument('--d', type=int, default=64); ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--seeds', type=int, default=3); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--drop', type=float, default=0.1); A_ = ap.parse_args()
O = Path('experiments/meta_ceiling_20261103/sweep'); OUT = O / 'sim_pred'; OUT.mkdir(exist_ok=True); torch.set_num_threads(2); assert torch.cuda.device_count() == 1
def norm(X):
    F = X.clone(); F[:, 4] = (F[:, 4] - 6) / 2; F[:, 7:9] = F[:, 7:9] * 2 - 1; return F
z = np.load(O / A_.data); X = norm(torch.from_numpy(z['X'][:, :9])).cuda(); ED = torch.from_numpy(z['E'].astype(np.int64)).cuda(); OFF = z['offs']; EOFF = z['eoff']; T = z['T'].astype(np.float32); fid = z['fid']; NF = len(T); print('sim frames', NF, 'nodes', len(X), flush=True)
geos = np.unique(fid); rg = np.random.default_rng(0); val_g = set(rg.permutation(geos)[:len(geos) // 10].tolist()); is_val = np.array([f in val_g for f in fid]); tr = np.where(~is_val)[0]; va = np.where(is_val)[0]; mu, sd = T[tr].mean(0), T[tr].std(0); Tt = torch.from_numpy((T - mu) / sd).cuda()
class Net(nn.Module):
    def __init__(self, d=A_.d, layers=A_.layers):
        super().__init__(); self.emb = nn.Linear(9, d); self.msg = nn.ModuleList(nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d)) for _ in range(layers)); self.upd = nn.ModuleList(nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU()) for _ in range(layers)); self.drop = nn.Dropout(A_.drop); self.head = nn.Sequential(nn.Linear(2 * d, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 3))
    def pool(self, x, e, frame, nf):
        h = self.emb(x); src = torch.cat([e[:, 0], e[:, 1]]); dst = torch.cat([e[:, 1], e[:, 0]])
        for m, u in zip(self.msg, self.upd):
            ms = m(torch.cat([h[src], h[dst]], 1)); ag = torch.zeros_like(h).index_add_(0, dst, ms); c = torch.zeros(len(h), device=h.device).index_add_(0, dst, torch.ones(len(dst), device=h.device)).clamp_min(1)[:, None]; h = h + self.drop(u(torch.cat([h, ag / c], 1)))
        cnt = torch.zeros(nf, device=h.device).index_add_(0, frame, torch.ones(len(frame), device=h.device)).clamp_min(1)[:, None]; mean = torch.zeros(nf, h.shape[1], device=h.device).index_add_(0, frame, h) / cnt; var = torch.zeros(nf, h.shape[1], device=h.device).index_add_(0, frame, (h - mean[frame]) ** 2) / cnt; return torch.cat([mean, (var + 1e-6).sqrt()], 1)
    def forward(self, x, e, frame, nf): return self.head(self.pool(x, e, frame, nf))
def batch(rows, X_, E_, OF, EO):
    xs, es, fr = [], [], []; base = 0; keep = []
    for k, i in enumerate(rows):
        a, c = OF[i], OF[i + 1]
        if c == a: continue
        xs.append(X_[a:c]); es.append(E_[EO[i]:EO[i + 1]] - a + base); fr.append(torch.full((c - a,), len(keep), device=X_.device)); base += c - a; keep.append(k)
    return torch.cat(xs), torch.cat(es), torch.cat(fr), keep
def predict(net, X_, E_, OF, EO, n):
    net.eval(); P = np.full((n, 3), np.nan, np.float32); Z = np.full((n, 2 * A_.d), np.nan, np.float32)
    with torch.no_grad():
        for s in range(0, n, 256):
            rows = np.arange(s, min(s + 256, n)); x, e, f, keep = batch(rows, X_, E_, OF, EO)
            if keep:
                z_ = net.pool(x, e, f, len(keep)); p = net.head(z_); r = rows[keep]; P[r] = p.cpu().numpy(); Z[r] = z_.cpu().numpy()
    return P, Z
stats = {}
def plain(X_, E_, OF, n):
    """Plain statistics on the same noisy nodes: share of six sides among bounded cells, CV of area, for comparison."""
    x = X_.cpu().numpy(); o = np.full((n, 2), np.nan)
    for i in range(n):
        a, c = OF[i], OF[i + 1]; m = x[a:c, 6] == 0
        if m.sum() >= 8: s = np.round(x[a:c, 4][m] * 2 + 6); ar = np.exp(x[a:c, 1][m]); o[i] = (np.mean(s == 6), ar.std() / ar.mean())
    return o
PL = plain(X, ED, OFF, NF); Rv = {}
for seed in range(A_.seeds):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); net = Net().cuda(); opt = torch.optim.AdamW(net.parameters(), A_.lr, weight_decay=1e-2); bs = 64; steps = A_.epochs * (len(tr) // bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, A_.lr, total_steps=steps, pct_start=0.1); t0 = time.time()
    for ep in range(A_.epochs):
        net.train(); pm = rng.permutation(tr); tot = 0.0
        for s in range(len(pm) // bs):
            rr = pm[s * bs:(s + 1) * bs]; x, e, f, keep = batch(rr, X, ED, OFF, EOFF); p = net(x, e, f, len(keep)); loss = (p - Tt[torch.from_numpy(rr[keep]).cuda()]).abs().mean(); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step(); tot += float(loss)
        if ep % 5 == 4 or ep == A_.epochs - 1:
            P, _ = predict(net, X, ED, OFF, EOFF, NF); P = P[va] * sd + mu; print(f'{A_.name} seed {seed} epoch {ep + 1} loss {tot / (len(pm) // bs):.4f} val r HEX {np.corrcoef(P[:, 0], T[va, 0])[0, 1]:.3f} CV {np.corrcoef(P[:, 1], T[va, 1])[0, 1]:.3f} CD {np.corrcoef(P[:, 2], T[va, 2])[0, 1]:.3f} {time.time() - t0:.0f}s', flush=True)
    torch.save(dict(model=net.state_dict(), mu=mu, sd=sd, args=vars(A_)), OUT / f'{A_.name}_s{seed}.pt')
    for rname in A_.real.split(','):
        g = np.load(O / rname); Xr = norm(torch.from_numpy(g['X'][:, :9])).cuda(); Er = torch.from_numpy(g['E'].astype(np.int64)).cuda(); P, Z = predict(net, Xr, Er, g['offs'], g['eoff'], len(g['offs']) - 1); Rv.setdefault(rname, []).append((P * sd + mu, Z))
        del Xr, Er
mp = lambda a, b: round(float(np.corrcoef(a[np.isfinite(a)], b[np.isfinite(a)])[0, 1]), 3)
print('plain statistics on the same noisy nodes, held-out sim: HEX r', mp(np.log(np.maximum(PL[va, 0], 0.02)), T[va, 0]), 'CV r', mp(np.log(np.maximum(PL[va, 1], 0.02)), T[va, 1]), flush=True)
for rname, v in Rv.items():
    np.savez_compressed(OUT / f"{A_.name}__{rname}", pred=np.mean([q[0] for q in v], 0).astype(np.float32), emb=np.mean([q[1] for q in v], 0).astype(np.float16))
json.dump(dict(args=vars(A_), frames=NF), open(OUT / f'{A_.name}.json', 'w'), indent=1); print('finished', flush=True)
