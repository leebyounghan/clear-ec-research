"""Negative control for the cell-graph network (2026-11-05): --shuffle-target permutes the residual targets among training rows, so graphs carry no information about them. Any gain left on held-out rows would mean a bug or a leak in the pipeline. Otherwise as cell_graph_sweep2.
Sweep of the cell-graph network with edge evidence (2026-11-05). --edge N uses the first N edge features (0 none, 2 border response only, 5 all); --node-extra adds the two border node features; both zero or absent reproduce the geometry-only network. Earlier text follows.
Sweep of the cell-graph network on the legible windows (2026-11-05). First pass, d 64, 3 layers, one seed: all metrics -0.025 (CI -0.040, -0.011), five folds negative, CV -0.028, HEX -0.049. Same LOFO and target as cell_graph_train. Arguments set width, depth, epochs, LR, seeds, dropout, whether context is given and whether messages pass (layers 0 is a per-cell set model, the control for topology). Seeds average in the log-residual space. usage: cell_graph_sweep_20261105.py NAME --d 64 --layers 3 --epochs 25 --lr 1e-3 --seeds 5 [--no-ctx] [--graphs FILE]. One GPU, shared."""
import argparse
import json, sys, time, warnings
import numpy as np
import torch
from torch import nn
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import stretch_nested_20261103 as N
ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--d', type=int, default=64); ap.add_argument('--layers', type=int, default=3); ap.add_argument('--epochs', type=int, default=25); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--no-ctx', action='store_true'); ap.add_argument('--graphs', default='cell_graphs.npz'); ap.add_argument('--wd', type=float, default=1e-2); ap.add_argument('--drop', type=float, default=0.1); ap.add_argument('--shuffle-target', action='store_true'); ap.add_argument('--edge', type=int, default=0); ap.add_argument('--edge-cols', default=''); ap.add_argument('--node-extra', action='store_true'); ap.add_argument('--tokens', default=''); ap.add_argument('--token-dims', type=int, default=32); A_ = ap.parse_args()
E = C.E; K = list(C.K); b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']; g = E.donors(b['ids']); O = C.X / 'meta_ceiling_20261103/sweep'; N_ = len(y); torch.set_num_threads(2)
z = np.load(O / A_.graphs); X = torch.from_numpy(z['X'] if A_.node_extra else z['X'][:, :9]); ED = torch.from_numpy(z['E'].astype(np.int64)); WID = torch.from_numpy(z['wid'].astype(np.int64)); OFF = z['offs']; EOFF = z['eoff']; print('nodes', X.shape, 'edges', ED.shape, flush=True)
gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); a_, b_, w_ = -0.2084, 0.5989, 0.28; have = gf['f_active10_valid'] & ~gf['seen_leaked'] & np.isfinite(gf['f_active10']); good = ~gf['seen_leaked']; f6 = np.nan_to_num(gf['f_active10'].astype(float), nan=float(np.nanmedian(gf['f_active10']))); H = B.copy(); H[have, 2] = np.exp((1 - w_) * np.log(B[have, 2]) + w_ * (a_ + b_ * f6[have]))
REF = np.full_like(H, np.nan)
for f in range(5): REF[fold == f] = N.apply_stretch(H[fold == f], N.fit_stretch(H, y, (fold != f) & good))
lH = np.log(H); CTX = np.c_[(lH - lH.mean(0)) / lH.std(0), (f6 - f6.mean()) / f6.std()].astype(np.float32); T_ = np.log(np.where(y > 0, y, 1.0)) - lH; MASK = (y > 0).astype(np.float32); FR = idx.astype(int)


def norm(X):
    F = X.clone(); F[:, 4] = (F[:, 4] - 6) / 2; F[:, 7:] = F[:, 7:] * 2 - 1; return F


Xn = norm(X)
if A_.tokens:
    with np.load(O / A_.tokens) as tz:
        TK = torch.from_numpy(tz['tokens'][:, :A_.token_dims].astype(np.float32))
    assert len(TK) == len(Xn), (len(TK), len(Xn))
    Xn = torch.cat([Xn, TK], 1); print('with tokens', Xn.shape, flush=True)
Xn = Xn.cuda()
EC = [int(q) for q in A_.edge_cols.split(',')] if A_.edge_cols else list(range(A_.edge))
EA_ = torch.from_numpy(z['EA'][:, EC].astype(np.float32)).cuda() if EC else None
if EA_ is not None: EA_ = (EA_ - EA_.mean(0)) / (EA_.std(0) + 1e-6); print('edge features', EA_.shape, EC, flush=True)
NE = len(EC)
ED_ = ED.cuda(); WID_ = WID.cuda()


class Net(nn.Module):
    def __init__(self, d=A_.d, layers=A_.layers, ctx=4):
        super().__init__(); self.emb = nn.Linear(Xn.shape[1], d); self.msg = nn.ModuleList(nn.Sequential(nn.Linear(2 * d + NE, d), nn.GELU(), nn.Linear(d, d)) for _ in range(layers)); self.upd = nn.ModuleList(nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU()) for _ in range(layers)); self.drop = nn.Dropout(A_.drop); self.head = nn.Sequential(nn.Linear(2 * d + ctx, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 3)); nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, e, ea, frame, nf, ctx):
        h = self.emb(x); src = torch.cat([e[:, 0], e[:, 1]]); dst = torch.cat([e[:, 1], e[:, 0]]); ea2 = torch.cat([ea, ea]) if ea is not None else None
        for m, u in zip(self.msg, self.upd):
            ms = m(torch.cat([h[src], h[dst]] + ([ea2] if ea2 is not None else []), 1)); ag = torch.zeros_like(h).index_add_(0, dst, ms); c = torch.zeros(len(h), device=h.device).index_add_(0, dst, torch.ones(len(dst), device=h.device)).clamp_min(1)[:, None]; h = h + self.drop(u(torch.cat([h, ag / c], 1)))
        cnt = torch.zeros(nf, device=h.device).index_add_(0, frame, torch.ones(len(frame), device=h.device)).clamp_min(1)[:, None]; mean = torch.zeros(nf, h.shape[1], device=h.device).index_add_(0, frame, h) / cnt; var = torch.zeros(nf, h.shape[1], device=h.device).index_add_(0, frame, (h - mean[frame]) ** 2) / cnt; return self.head(torch.cat([mean, (var + 1e-6).sqrt(), ctx], 1))


def batch(rows):
    xs, es, fr, ee = [], [], [], []; base = 0
    for k, r in enumerate(rows):
        i = FR[r]; a, c = OFF[i], OFF[i + 1]
        if c == a: continue
        ea, ec = EOFF[i], EOFF[i + 1]; xs.append(Xn[a:c]); es.append(ED_[ea:ec] - a + base); ee.append(EA_[ea:ec] if EA_ is not None else None); fr.append(torch.full((c - a,), k, device=Xn.device)); base += c - a
    return (torch.cat(xs), torch.cat(es), torch.cat(ee) if EA_ is not None else None, torch.cat(fr)) if xs else None


CTX = CTX * (0.0 if A_.no_ctx else 1.0)
CTXt = torch.from_numpy(CTX).cuda(); Tt = torch.from_numpy(T_.astype(np.float32)).cuda(); Mt = torch.from_numpy(MASK).cuda()


def predict(net, rows):
    net.eval(); out = np.zeros((N_, 3))
    with torch.no_grad():
        for s in range(0, len(rows), 128):
            rr = rows[s:s + 128]; bb = batch(rr)
            if bb is None: continue
            p = net(*bb, len(rr), CTXt[torch.from_numpy(rr).cuda()]); out[rr] = p.cpu().numpy()
    return out


def run(f, epochs=A_.epochs, lr=A_.lr, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); tr = np.where((fold != f) & good & (np.diff(OFF)[FR] > 0))[0]; te = np.where(fold == f)[0]; shuffled = rng.permutation(tr) if A_.shuffle_target else tr; TT = Tt.clone(); MM = Mt.clone(); TT[torch.from_numpy(tr).cuda()] = Tt[torch.from_numpy(shuffled).cuda()]; MM[torch.from_numpy(tr).cuda()] = Mt[torch.from_numpy(shuffled).cuda()]; net = Net().cuda(); opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=A_.wd); steps = epochs * (len(tr) // 64); sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1); snaps = np.zeros((N_, 3)); ns = 0; t0 = time.time(); allrows = np.arange(N_)
    for ep in range(epochs):
        net.train(); pm = rng.permutation(tr); tot = 0.0
        for s in range(len(pm) // 64):
            rr = pm[s * 64:(s + 1) * 64]; bb = batch(rr); ri = torch.from_numpy(rr).cuda(); p = net(*bb, len(rr), CTXt[ri]); loss = ((p - TT[ri]).abs() * MM[ri]).sum() / MM[ri].sum(); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step(); tot += float(loss)
        if ep >= epochs - 5:
            snaps += predict(net, allrows); ns += 1
        if ep == epochs - 1: print(f'{A_.name} fold {f} seed {seed} loss {tot / (len(pm) // 64):.4f} {time.time() - t0:.0f}s', flush=True)
    return snaps / ns


def pipe(P_all, ms=(0, 1, 2)):
    P = np.full_like(H, np.nan)
    for f in range(5):
        tr = (fold != f) & good; C_ = H.copy()
        for j in ms: C_[:, j] = H[:, j] * np.exp(P_all[f][:, j])
        P[fold == f] = N.apply_stretch(C_[fold == f], N.fit_stretch(C_, y, tr))
    return P


def report(nm, p):
    pb = E.paired_bootstrap(E.ape(p, y), E.ape(REF, y), g); s, s0 = E.official(E.ape(p, y)), E.official(E.ape(REF, y)); per = [round(E.official(E.ape(p[fold == f], y[fold == f]))['mean'] - E.official(E.ape(REF[fold == f], y[fold == f]))['mean'], 3) for f in range(5)]; o = dict(score=round(s['mean'], 3), per_delta={k: round(s[k] - s0[k], 3) for k in K}, delta=round(pb['delta'], 4), ci=[round(v, 4) for v in pb['ci95']], folds=per); print(nm, json.dumps(o), flush=True); return o


pred = {f: np.mean([run(f, seed=q) for q in range(A_.seeds)], 0) for f in range(5)}; rep = dict(config=vars(A_))
for nm, ms in dict(CD=(0,), CV=(1,), HEX=(2,), all3=(0, 1, 2), CV_HEX=(1, 2)).items(): rep[nm] = report(A_.name + ' ' + nm, pipe(pred, ms))
(O / 'graph_sweep').mkdir(exist_ok=True); np.savez_compressed(O / 'graph_sweep' / f'{A_.name}.npz', **{f'f{f}': pred[f].astype(np.float32) for f in pred}); json.dump(rep, open(O / 'graph_sweep' / f'{A_.name}.json', 'w'), indent=1); print('done', flush=True)
