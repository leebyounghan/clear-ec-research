"""Cell-graph network for the HEX lane (2026-11-08), from meta_ceiling cell_graph_sweep3 (same net, batching, LOFO, target
log y - log H, snapshot mean of the last 5 epochs). Changes: --lw loss weights per metric (0,0,1 = HEX only), --dev, --graphs
may point anywhere, output to lane preds/. Reads HEX three ways against the best pipeline: swap (BEST - old graph + new),
ens (old and new graph averaged), stack (nested ridge on log HB and the new HEX output).
usage: gnn_hex.py NAME --graphs PATH [--lw 0,0,1] [--seeds 5] [--layers 5] [--d 64] [--epochs 25] [--dev cuda] [--folds 0,1,2,3,4 | full]
--folds full (2026-11-10): trains on all good rows and saves each seed's last-5-epoch weights to deploy/graph_models/
(NAME_s{seed}_e{epoch}.pt) plus the in-sample prediction NAME_fullfit_pred.npz; inference: source/gnn_hex_infer.py."""
import argparse, json, sys, time, warnings
import numpy as np
import torch
from torch import nn
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source')
import hexlib as L
BP = L.BP; N_ = L.N; y, fold, idx, good = L.y, L.fold, L.idx, L.good
ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--graphs', required=True); ap.add_argument('--lw', default='1,1,1'); ap.add_argument('--d', type=int, default=64); ap.add_argument('--layers', type=int, default=5); ap.add_argument('--epochs', type=int, default=25); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--seed0', type=int, default=0); ap.add_argument('--wd', type=float, default=1e-2); ap.add_argument('--drop', type=float, default=0.1); ap.add_argument('--ctx', action='store_true'); ap.add_argument('--ctxfile', default=''); ap.add_argument('--dev', default='cuda'); ap.add_argument('--threads', type=int, default=4); ap.add_argument('--folds', default='0,1,2,3,4'); ap.add_argument('--bs', type=int, default=64); ap.add_argument('--nfeat', type=int, default=9); ap.add_argument('--nolog', action='store_true'); A_ = ap.parse_args()
torch.set_num_threads(A_.threads); DEV = A_.dev; FOLDS = [-1] if A_.folds == 'full' else [int(q) for q in A_.folds.split(',')]; LW = torch.tensor([float(q) for q in A_.lw.split(',')])
z = np.load(A_.graphs); X = torch.from_numpy(z['X'][:, :A_.nfeat].astype(np.float32)); ED = torch.from_numpy(z['E'].astype(np.int64)); OFF = z['offs']; EOFF = z['eoff']; print('nodes', tuple(X.shape), 'edges', tuple(ED.shape), flush=True)
H = BP.H; lH = np.log(H); f6 = np.nan_to_num(BP.f6, nan=float(np.nanmedian(BP.f6)))
CTX = np.c_[(lH - lH.mean(0)) / lH.std(0), (f6 - f6.mean()) / f6.std()].astype(np.float32) * (1.0 if A_.ctx else 0.0)
if A_.ctxfile:
    _z = np.load(A_.ctxfile); assert (_z['idx'] == idx).all(); _G = (_z['F'] if 'F' in _z.files else np.stack([_z[k] for k in _z.files if k != 'idx'], 1)).astype(np.float64)
    _G = np.where(np.isfinite(_G), _G, np.nanmedian(_G, 0)); _G = (_G - _G.mean(0)) / (_G.std(0) + 1e-9); CTX = np.c_[CTX, np.clip(_G, -5, 5)].astype(np.float32); print('ctxfile', A_.ctxfile, _G.shape, flush=True)
T_ = np.log(np.where(y > 0, y, 1.0)) - lH; MASK = (y > 0).astype(np.float32); FR = idx.astype(int)
F = X.clone(); F[:, 4] = (F[:, 4] - 6) / 2
if F.shape[1] >= 9: F[:, 7:9] = F[:, 7:9] * 2 - 1
Xn = F.to(DEV); ED_ = ED.to(DEV)


class Net(nn.Module):
    def __init__(self, d=A_.d, layers=A_.layers, ctx=CTX.shape[1]):
        super().__init__(); self.emb = nn.Linear(Xn.shape[1], d); self.msg = nn.ModuleList(nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d)) for _ in range(layers)); self.upd = nn.ModuleList(nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU()) for _ in range(layers)); self.drop = nn.Dropout(A_.drop); self.head = nn.Sequential(nn.Linear(2 * d + ctx, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 3)); nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, e, frame, nf, ctx):
        h = self.emb(x); src = torch.cat([e[:, 0], e[:, 1]]); dst = torch.cat([e[:, 1], e[:, 0]])
        c = torch.zeros(len(h), device=h.device).index_add_(0, dst, torch.ones(len(dst), device=h.device)).clamp_min(1)[:, None]
        for m, u in zip(self.msg, self.upd):
            ms = m(torch.cat([h[src], h[dst]], 1)); ag = torch.zeros_like(h).index_add_(0, dst, ms); h = h + self.drop(u(torch.cat([h, ag / c], 1)))
        cnt = torch.zeros(nf, device=h.device).index_add_(0, frame, torch.ones(len(frame), device=h.device)).clamp_min(1)[:, None]; mean = torch.zeros(nf, h.shape[1], device=h.device).index_add_(0, frame, h) / cnt; var = torch.zeros(nf, h.shape[1], device=h.device).index_add_(0, frame, (h - mean[frame]) ** 2) / cnt; return self.head(torch.cat([mean, (var + 1e-6).sqrt(), ctx], 1))


def batch(rows):
    xs, es, fr = [], [], []; base = 0
    for k, r in enumerate(rows):
        i = FR[r]; a, c = OFF[i], OFF[i + 1]
        if c == a: continue
        ea, ec = EOFF[i], EOFF[i + 1]; xs.append(Xn[a:c]); es.append(ED_[ea:ec] - a + base); fr.append(torch.full((c - a,), k, device=DEV)); base += c - a
    return (torch.cat(xs), torch.cat(es), torch.cat(fr)) if xs else None


CTXt = torch.from_numpy(CTX).to(DEV); Tt = torch.from_numpy(T_.astype(np.float32)).to(DEV); Mt = (torch.from_numpy(MASK) * LW[None]).to(DEV)


def predict(net, rows):
    net.eval(); out = np.zeros((N_, 3))
    with torch.no_grad():
        for s in range(0, len(rows), 128):
            rr = rows[s:s + 128]; bb = batch(rr)
            if bb is None: continue
            out[rr] = net(*bb, len(rr), CTXt[torch.from_numpy(rr).to(DEV)]).cpu().numpy()
    return out


DUTYF = '/workspace/clear-ec-research/experiments/lane_hex_20261108/duty_gpu1.txt'; _D = [1.0]


def _thr(step, t1):
    """GPU duty cap for sharing a GPU: after each update sleep so that this job is busy about D of the time (D read from
    duty_gpu1.txt every 50 updates, default 1 = no cap)."""
    if step % 50 == 0:
        try: _D[0] = min(1.0, max(0.05, float(open(DUTYF).read().strip())))
        except Exception: _D[0] = 1.0
    if _D[0] < 1.0:
        if DEV.startswith('cuda'): torch.cuda.synchronize()
        time.sleep((time.time() - t1) * (1.0 / _D[0] - 1.0))


def run(f, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); tr = np.where((fold != f) & good & (np.diff(OFF)[FR] > 0))[0]; net = Net().to(DEV); opt = torch.optim.AdamW(net.parameters(), A_.lr, weight_decay=A_.wd); bs = A_.bs; steps = A_.epochs * (len(tr) // bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, A_.lr, total_steps=steps, pct_start=0.1); snaps = np.zeros((N_, 3)); ns = 0; t0 = time.time(); allrows = np.arange(N_)
    for ep in range(A_.epochs):
        net.train(); pm = rng.permutation(tr); tot = 0.0
        for s in range(len(pm) // bs):
            _t1 = time.time(); rr = pm[s * bs:(s + 1) * bs]; bb = batch(rr); ri = torch.from_numpy(rr).to(DEV); p = net(*bb, len(rr), CTXt[ri]); loss = ((p - Tt[ri]).abs() * Mt[ri]).sum() / Mt[ri].sum(); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step(); _thr(s, _t1); tot += float(loss)
        if ep == 0: print(f'{A_.name} fold {f} seed {seed} epoch 0 {time.time() - t0:.0f}s', flush=True)
        if ep >= A_.epochs - 5:
            snaps += predict(net, allrows); ns += 1
            if f < 0:
                SAVE.mkdir(parents=True, exist_ok=True)
                torch.save(dict(model=net.state_dict(), d=A_.d, layers=A_.layers, nfeat=A_.nfeat, drop=A_.drop, ctx=bool(A_.ctx or A_.ctxfile), lw=A_.lw, epoch=ep + 1, seed=seed, graphs=A_.graphs), SAVE / f'{A_.name}_s{seed}_e{ep + 1}.pt')
    print(f'{A_.name} fold {f} seed {seed} loss {tot / (len(pm) // bs):.4f} {time.time() - t0:.0f}s', flush=True)
    return snaps / ns


from pathlib import Path
SAVE = Path(L.LANE) / 'deploy' / 'graph_models'
if FOLDS == [-1]:
    assert not (A_.ctx or A_.ctxfile), 'full mode: context inputs are not wired for unseen images'
    full = np.mean([run(-1, seed=A_.seed0 + q) for q in range(A_.seeds)], 0); np.savez_compressed(SAVE / f'{A_.name}_fullfit_pred.npz', idx=idx, pred=full.astype(np.float32))
    print('full-data models saved in', SAVE, flush=True); sys.exit(0)
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
