"""Multi-detector cell-graph network (HEX lane, 2026-11-08): --graphs takes several graph files (one per detector or fused
set); in training each frame of a batch uses the graph of a randomly drawn detector, at prediction the outputs are averaged over
every detector that has a graph for the frame. One network learns from all detectors' errors at once. Otherwise as below.
Cell-graph network for the HEX lane (2026-11-08), from meta_ceiling cell_graph_sweep3 (same net, batching, LOFO, target
log y - log H, snapshot mean of the last 5 epochs). Changes: --lw loss weights per metric (0,0,1 = HEX only), --dev, --graphs
may point anywhere, output to lane preds/. Reads HEX three ways against the best pipeline: swap (BEST - old graph + new),
ens (old and new graph averaged), stack (nested ridge on log HB and the new HEX output).
usage: gnn_hex.py NAME --graphs PATH [--lw 0,0,1] [--seeds 5] [--layers 5] [--d 64] [--epochs 25] [--dev cuda] [--folds 0,1,2,3,4]"""
import argparse, json, sys, time, warnings
import numpy as np
import torch
from torch import nn
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source')
import hexlib as L
BP = L.BP; N_ = L.N; y, fold, idx, good = L.y, L.fold, L.idx, L.good
ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--graphs', required=True); ap.add_argument('--lw', default='1,1,1'); ap.add_argument('--d', type=int, default=64); ap.add_argument('--layers', type=int, default=5); ap.add_argument('--epochs', type=int, default=25); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--seed0', type=int, default=0); ap.add_argument('--wd', type=float, default=1e-2); ap.add_argument('--drop', type=float, default=0.1); ap.add_argument('--ctx', action='store_true'); ap.add_argument('--dev', default='cuda'); ap.add_argument('--threads', type=int, default=4); ap.add_argument('--folds', default='0,1,2,3,4'); ap.add_argument('--bs', type=int, default=64); ap.add_argument('--nfeat', type=int, default=9); ap.add_argument('--nolog', action='store_true'); A_ = ap.parse_args()
torch.set_num_threads(A_.threads); DEV = A_.dev; FOLDS = [int(q) for q in A_.folds.split(',')]; LW = torch.tensor([float(q) for q in A_.lw.split(',')])
GR = []
for gp in A_.graphs.split(','):
    z = np.load(gp); X = torch.from_numpy(z['X'][:, :A_.nfeat].astype(np.float32)); X[:, 4] = (X[:, 4] - 6) / 2
    if X.shape[1] >= 9: X[:, 7:9] = X[:, 7:9] * 2 - 1
    GR.append((X.to(A_.dev), torch.from_numpy(z['E'].astype(np.int64)).to(A_.dev), z['offs'], z['eoff'])); print(gp, 'nodes', tuple(X.shape), flush=True)
NG = len(GR); HAS = np.stack([np.diff(g[2]) > 0 for g in GR], 1)   # frame x graph
H = BP.H; lH = np.log(H); f6 = np.nan_to_num(BP.f6, nan=float(np.nanmedian(BP.f6)))
CTX = np.c_[(lH - lH.mean(0)) / lH.std(0), (f6 - f6.mean()) / f6.std()].astype(np.float32) * (1.0 if A_.ctx else 0.0)
T_ = np.log(np.where(y > 0, y, 1.0)) - lH; MASK = (y > 0).astype(np.float32); FR = idx.astype(int)
NF = GR[0][0].shape[1]


class Net(nn.Module):
    def __init__(self, d=A_.d, layers=A_.layers, ctx=4):
        super().__init__(); self.emb = nn.Linear(NF, d); self.msg = nn.ModuleList(nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d)) for _ in range(layers)); self.upd = nn.ModuleList(nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU()) for _ in range(layers)); self.drop = nn.Dropout(A_.drop); self.head = nn.Sequential(nn.Linear(2 * d + ctx, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 3)); nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, e, frame, nf, ctx):
        h = self.emb(x); src = torch.cat([e[:, 0], e[:, 1]]); dst = torch.cat([e[:, 1], e[:, 0]])
        c = torch.zeros(len(h), device=h.device).index_add_(0, dst, torch.ones(len(dst), device=h.device)).clamp_min(1)[:, None]
        for m, u in zip(self.msg, self.upd):
            ms = m(torch.cat([h[src], h[dst]], 1)); ag = torch.zeros_like(h).index_add_(0, dst, ms); h = h + self.drop(u(torch.cat([h, ag / c], 1)))
        cnt = torch.zeros(nf, device=h.device).index_add_(0, frame, torch.ones(len(frame), device=h.device)).clamp_min(1)[:, None]; mean = torch.zeros(nf, h.shape[1], device=h.device).index_add_(0, frame, h) / cnt; var = torch.zeros(nf, h.shape[1], device=h.device).index_add_(0, frame, (h - mean[frame]) ** 2) / cnt; return self.head(torch.cat([mean, (var + 1e-6).sqrt(), ctx], 1))


def batch(rows, gsel):
    xs, es, fr = [], [], []; base = 0
    for k, (r, gi) in enumerate(zip(rows, gsel)):
        if gi < 0: continue
        Xn, ED_, OFF, EOFF = GR[gi]; i = FR[r]; a, c = OFF[i], OFF[i + 1]
        if c == a: continue
        ea, ec = EOFF[i], EOFF[i + 1]; xs.append(Xn[a:c]); es.append(ED_[ea:ec] - a + base); fr.append(torch.full((c - a,), k, device=DEV)); base += c - a
    return (torch.cat(xs), torch.cat(es), torch.cat(fr)) if xs else None


def draw(rows, rng):
    h = HAS[FR[rows]]; u = rng.random(h.shape) * h; g = u.argmax(1); g[~h.any(1)] = -1; return g


CTXt = torch.from_numpy(CTX).to(DEV); Tt = torch.from_numpy(T_.astype(np.float32)).to(DEV); Mt = (torch.from_numpy(MASK) * LW[None]).to(DEV)


def predict(net, rows):
    net.eval(); out = np.zeros((N_, 3)); cnt = np.zeros(N_)
    with torch.no_grad():
        for gi in range(NG):
            for s in range(0, len(rows), 128):
                rr = rows[s:s + 128]; rr = rr[HAS[FR[rr], gi]]
                if not len(rr): continue
                bb = batch(rr, np.full(len(rr), gi)); out[rr] += net(*bb, len(rr), CTXt[torch.from_numpy(rr).to(DEV)]).cpu().numpy(); cnt[rr] += 1
    return out / np.maximum(cnt, 1)[:, None]


DUTYF = '/workspace/clear-ec-research/experiments/lane_hex_20261108/duty_gpu1.txt'; _D = [1.0]


def _thr(step, t1):
    """GPU duty cap (see gnn_hex.py): sleep after each update so this job is busy about D of the time."""
    if step % 50 == 0:
        try: _D[0] = min(1.0, max(0.05, float(open(DUTYF).read().strip())))
        except Exception: _D[0] = 1.0
    if _D[0] < 1.0:
        if DEV.startswith('cuda'): torch.cuda.synchronize()
        time.sleep((time.time() - t1) * (1.0 / _D[0] - 1.0))


def run(f, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); tr = np.where((fold != f) & good & HAS[FR].any(1))[0]; net = Net().to(DEV); opt = torch.optim.AdamW(net.parameters(), A_.lr, weight_decay=A_.wd); bs = A_.bs; steps = A_.epochs * (len(tr) // bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, A_.lr, total_steps=steps, pct_start=0.1); snaps = np.zeros((N_, 3)); ns = 0; t0 = time.time(); allrows = np.arange(N_)
    for ep in range(A_.epochs):
        net.train(); pm = rng.permutation(tr); tot = 0.0
        for s in range(len(pm) // bs):
            _t1 = time.time(); rr = pm[s * bs:(s + 1) * bs]; bb = batch(rr, draw(rr, rng)); ri = torch.from_numpy(rr).to(DEV); p = net(*bb, len(rr), CTXt[ri]); loss = ((p - Tt[ri]).abs() * Mt[ri]).sum() / Mt[ri].sum(); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step(); _thr(s, _t1); tot += float(loss)
        if ep == 0: print(f'{A_.name} fold {f} seed {seed} epoch 0 {time.time() - t0:.0f}s', flush=True)
        if ep >= A_.epochs - 5: snaps += predict(net, allrows); ns += 1
        if ep == 0: print('epoch 0 incl.', flush=True)
    print(f'{A_.name} fold {f} seed {seed} loss {tot / (len(pm) // bs):.4f} {time.time() - t0:.0f}s', flush=True)
    return snaps / ns


pred = {}
for f in FOLDS:
    pred[f] = np.mean([run(f, seed=A_.seed0 + q) for q in range(A_.seeds)], 0)
    np.savez_compressed(f'{L.LANE}/preds/{A_.name}.npz', **{f'f{k}': v.astype(np.float32) for k, v in pred.items()})
if len(FOLDS) == 5:
    new = {f: pred[f][:, 2] for f in range(5)}; old = {f: BP.GRAPH[f][:, 2] for f in range(5)}
    L.evaluate(A_.name + ' swap', {f: new[f] - old[f] for f in range(5)}, log=not A_.nolog, note='graph HEX swapped into best pipeline')
    L.evaluate(A_.name + ' ens', {f: 0.5 * (new[f] - old[f]) for f in range(5)}, log=not A_.nolog, note='old and new graph HEX averaged')
    L.evaluate(A_.name + ' stack', L.nested({f: new[f] for f in range(5)}), log=not A_.nolog, note='nested ridge on log HB + new graph HEX')
print('done', flush=True)
