"""Inference for the full-data cell-graph networks (HEX lane, 2026-11-10). Works for both weight layouts:
gnn_hex.py --folds full (deploy/graph_models/NAME_s{seed}_e{ep}.pt) and graph_train_full.py (same layout).
Output per image: the network's 3-column log correction (CD, CV, HEX; the HEX chain uses column 2), mean over all saved seeds
and epochs (the OOF read used the mean of the last 5 epochs' predictions per seed, then the mean over seeds).
Graph file: the cell-graph format (X nodes, E edges, offs, eoff; row i = image i of the file's own order).
usage: gnn_hex_infer.py NAME GRAPHS.npz OUT.npz [--models DIR] [--check]   (--check: GRAPHS is a training graph file; compares
with NAME_fullfit_pred.npz, which must match)"""
import sys, glob, argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn

LANE = Path('/workspace/clear-ec-research/experiments/lane_hex_20261108')


class Net(nn.Module):
    """copy of gnn_hex.Net / cell_graph_sweep3 Net (geometry-only, no edge features)"""
    def __init__(self, nfeat, d, layers, drop, ctx):
        super().__init__(); self.emb = nn.Linear(nfeat, d); self.msg = nn.ModuleList(nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d)) for _ in range(layers))
        self.upd = nn.ModuleList(nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU()) for _ in range(layers)); self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(nn.Linear(2 * d + ctx, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 3))

    def forward(self, x, e, frame, nf, ctx):
        h = self.emb(x); src = torch.cat([e[:, 0], e[:, 1]]); dst = torch.cat([e[:, 1], e[:, 0]])
        c = torch.zeros(len(h), device=h.device).index_add_(0, dst, torch.ones(len(dst), device=h.device)).clamp_min(1)[:, None]
        for m, u in zip(self.msg, self.upd):
            ms = m(torch.cat([h[src], h[dst]], 1)); ag = torch.zeros_like(h).index_add_(0, dst, ms); h = h + self.drop(u(torch.cat([h, ag / c], 1)))
        cnt = torch.zeros(nf, device=h.device).index_add_(0, frame, torch.ones(len(frame), device=h.device)).clamp_min(1)[:, None]
        mean = torch.zeros(nf, h.shape[1], device=h.device).index_add_(0, frame, h) / cnt
        var = torch.zeros(nf, h.shape[1], device=h.device).index_add_(0, frame, (h - mean[frame]) ** 2) / cnt
        return self.head(torch.cat([mean, (var + 1e-6).sqrt(), ctx], 1))


def load_graphs(path, nfeat, dev):
    z = np.load(path); X = torch.from_numpy(z['X'][:, :nfeat].astype(np.float32)); F = X.clone(); F[:, 4] = (F[:, 4] - 6) / 2
    if F.shape[1] >= 9: F[:, 7:9] = F[:, 7:9] * 2 - 1
    return F.to(dev), torch.from_numpy(z['E'].astype(np.int64)).to(dev), z['offs'], z['eoff']


def predict(name, graphs, models=LANE / 'deploy' / 'graph_models', dev='cuda' if torch.cuda.is_available() else 'cpu'):
    files = sorted(glob.glob(str(Path(models) / f'{name}_s*_e*.pt'))); assert files, f'no weights for {name} in {models}'
    ck0 = torch.load(files[0], map_location='cpu'); nfeat = ck0.get('nfeat', 9); ctx = 3 + 1 if ck0.get('ctx') else 4
    Xn, ED, OFF, EOFF = load_graphs(graphs, nfeat, dev); M = len(OFF) - 1; out = np.zeros((M, 3)); n = 0
    for fpath in files:
        ck = torch.load(fpath, map_location='cpu'); net = Net(nfeat, ck['d'], ck['layers'], ck.get('drop', 0.1), ctx).to(dev); net.load_state_dict(ck['model']); net.eval(); P = np.zeros((M, 3))
        with torch.no_grad():
            for s in range(0, M, 128):
                rr = np.arange(s, min(M, s + 128)); xs, es, fr = [], [], []; base = 0; keep = []
                for k, i in enumerate(rr):
                    a, c = OFF[i], OFF[i + 1]
                    if c == a: continue
                    ea, ec = EOFF[i], EOFF[i + 1]; xs.append(Xn[a:c]); es.append(ED[ea:ec] - a + base); fr.append(torch.full((c - a,), k, device=dev)); base += c - a; keep.append(k)
                if not xs: continue
                p = net(torch.cat(xs), torch.cat(es), torch.cat(fr), len(rr), torch.zeros(len(rr), ctx, device=dev)).cpu().numpy(); P[rr] = p   # as the trainers: images without a graph get the network's output for an empty graph (only an all-empty batch stays 0)
        out += P; n += 1
    print(f'{name}: {n} weight files, {M} images', flush=True); return out / n


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('graphs'); ap.add_argument('out'); ap.add_argument('--models', default=str(LANE / 'deploy' / 'graph_models')); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
    P = predict(a.name, a.graphs, a.models); np.savez_compressed(a.out, pred=P.astype(np.float32))
    if a.check:
        sys.path.insert(0, str(LANE / 'source')); import hexlib as L
        ref = np.load(Path(a.models) / f'{a.name}_fullfit_pred.npz')['pred']; Q = P[L.idx.astype(int)]
        print('check vs fullfit_pred: max |diff|', float(np.abs(Q - ref).max()), flush=True)
