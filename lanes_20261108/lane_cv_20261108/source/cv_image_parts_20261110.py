"""Per-image, label-free parts of the CV column (CV lane, 2026-11-10). Imports nothing that reads labels, folds or OOF files; every constant is
frozen here or read from a weights / basis file. Feeds cv_apply_20261110.apply (keys in brackets).

Peaks: (n, 3) arrays x, y, confidence on the 1296 x 972 frame, from SG.peaks2(heat, support, 0.2) (green_sg2_specialist_20261028), as stored.
  single view  = the identity-flip heat map of one pre-only detector (tta_infer_20261108: fl == 0)
  tta4         = mean heat map over the 4 exact flips of one detector; seeds3_tta4 = mean over the 3 detectors x 4 flips
  Detectors at test time: green_track_20261028/deploy/detectors/sg2pre_s{46,47,48}.pt.
Graphs: graphs_from_peaks(peaks, kind) = cell_graph_prep3_20261105 / graph_prep_sp_20261108 (alpha 0): the 3 most legible 380 x 300 windows
  (stride 16, by count of peaks >= 0.5), nodes = peaks >= 0.35 in the window, Voronoi edges, 9 node features. kind 'prep3' keeps peak order (GRAPH
  ensemble and simulation embeddings), kind 'sp' sorts nodes by confidence (TTA quartets). Node order does not change the outputs beyond float rounding.
  GRAPH ensemble graphs: single view of s46 -> set 'noctx', s47 -> 'pre47', s48 -> 'pre48'.  Quartet graphs: s46_tta4, s47_tta4, s48_tta4, seeds3_tta4.
Graph networks: GraphNets(pattern).out(graphs) -> (3,) mean over checkpoints of the network output (log residual vs H; CV is index 1). An image
  with no graph gets the network's empty-graph output, as the training-time predict() gave such rows.
    GRAPH  [graph_cv]  = mean over sets noctx, pre47, pre48 of GraphNets(f'gfull_{set}_s*_s*_e*.pt').out(graphs of that set)[1]
    plain  [qplain_cv] = mean over 4 quartet sets of GraphNets(f'full_tta_{set}_tta4_s*_e*.pt').out(...)[1]
    order  [qorder_cv] = mean over 4 quartet sets of GraphNets(f'gord_{set}_tta4_full_s*_s*_e*.pt').out(...)[1]
Simulation embeddings [emb_s0..emb_s2]: SimNets().embed({'noctx': g46, 'pre47': g47, 'pre48': g48}) -> list over network seeds of (128,) vectors,
  mean over the graph sets that have a graph, NaN if none (sim_embed_eval_20261105.embed).
Token blocks + frozen heads [fz_std, fz_spectrum, fz_cov_sqrt, fz_autocorr]: token_blocks(t) from the full-data recipe A seed-46 encoder's patch
  tokens t (124*164, 768), one unflipped view, bf16 forward as in training (token_stats_cv_20261108.fw), then FrozenHeads().logcv(blocks).
usage: python cv_image_parts_20261110.py check   (parity against stored training-row arrays; needs the pod paths)"""
import glob, math, os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.spatial import Delaunay, Voronoi, cKDTree

XP = Path('/workspace/clear-ec-research/experiments'); LANE = XP / 'lane_cv_20261108'; GM = LANE / 'graph_models'
FH, FW = 972, 1296; WW, WH = 380, 300; NODE_T = 0.35
QSETS = ('s46', 's47', 's48', 'seeds3'); GSETS = ('noctx', 'pre47', 'pre48')


# ---------------- graphs from peaks ----------------
def windows(hi, w=WW, h=WH, k=3):
    """legible_window_features_20261105.windows: top windows by peak count, stride 16, greedy non-overlap."""
    G = np.zeros((FH // 16 + 2, FW // 16 + 2)); ix = np.minimum((hi[:, 1] / 16).astype(int), G.shape[0] - 1); iy = np.minimum((hi[:, 0] / 16).astype(int), G.shape[1] - 1); np.add.at(G, (ix, iy), 1)
    I = np.zeros((G.shape[0] + 1, G.shape[1] + 1)); I[1:, 1:] = G.cumsum(0).cumsum(1)
    nh, nw = h // 16, w // 16; S = I[nh:, nw:] - I[:-nh, nw:] - I[nh:, :-nw] + I[:-nh, :-nw]; ymax = (FH - h) // 16; xmax = (FW - w) // 16; S = S[:ymax + 1, :xmax + 1].copy(); out = []
    for _ in range(k):
        q = int(np.argmax(S)); y0, x0 = np.unravel_index(q, S.shape); out.append((int(x0) * 16, int(y0) * 16))
        a0, a1 = max(0, y0 - nh // 2), min(S.shape[0], y0 + nh // 2 + 1); b0, b1 = max(0, x0 - nw // 2), min(S.shape[1], x0 + nw // 2 + 1); S[a0:a1, b0:b1] = -1
    return out


def graph(q, x0, y0):
    """cell_graph_prep3_20261105.graph with K = 3 (window-relative positions)."""
    p = q[:, :2].astype(np.float64); c = q[:, 2]; n = len(p)
    if n < 12: return None
    try:
        vor = Voronoi(p); tri = Delaunay(p)
    except Exception:
        return None
    a = np.full(n, np.nan); s = np.zeros(n); ub = np.zeros(n)
    for i in range(n):
        g = vor.regions[vor.point_region[i]]
        if -1 in g or not g: ub[i] = 1; s[i] = max(len(g) - 1, 3); continue
        v = vor.vertices[g]; s[i] = len(g)
        if (tri.find_simplex(v) < 0).any(): ub[i] = 1
        x, z = v[:, 0], v[:, 1]; a[i] = 0.5 * abs(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1)))
    E = np.array(vor.ridge_points); d = np.linalg.norm(p[E[:, 0]] - p[E[:, 1]], axis=1); cnt = np.bincount(E.ravel(), minlength=n).astype(float); sm = np.bincount(E[:, 0], d, n) + np.bincount(E[:, 1], d, n); dm = sm / np.maximum(cnt, 1)
    cn = (np.bincount(E[:, 0], c[E[:, 1]], n) + np.bincount(E[:, 1], c[E[:, 0]], n)) / np.maximum(cnt, 1); nn_ = cKDTree(p).query(p, k=2)[0][:, 1]
    med_a = np.nanmedian(a[ub == 0]) if (ub == 0).sum() > 3 else np.nanmedian(a)
    F = np.c_[c, np.nan_to_num(np.log(np.maximum(a, 1) / max(med_a, 1)), nan=0.0, posinf=0.0, neginf=0.0).clip(-2, 2), np.log(dm / np.median(dm)).clip(-2, 2), np.log(nn_ / np.median(nn_)).clip(-2, 2), s.clip(3, 9), cn, ub, (p[:, 0] - x0) / WW, (p[:, 1] - y0) / WH]
    return F.astype(np.float32), E.astype(np.int32)


def graphs_from_peaks(pk, kind='prep3'):
    """List of (X (n, 9), E (m, 2)) for the image's windows; [] if the image has too few peaks."""
    pk = np.asarray(pk, np.float64)
    if pk.ndim != 2 or len(pk) == 0: return []
    hi = pk[pk[:, 2] >= 0.5, :2]; q = pk[pk[:, 2] >= NODE_T]
    if kind == 'sp': q = q[np.argsort(-q[:, 2])]
    if len(hi) < 10 or len(q) < 16: return []
    out = []
    for x0, y0 in windows(hi):
        m = (q[:, 0] >= x0) & (q[:, 0] < x0 + WW) & (q[:, 1] >= y0) & (q[:, 1] < y0 + WH); g = graph(q[m], x0, y0)
        if g is not None: out.append(g)
    return out


def _pack(graphs):
    X = np.concatenate([g[0] for g in graphs]); E, base = [], 0
    for g in graphs: E.append(g[1] + base); base += len(g[0])
    X = torch.from_numpy(X[:, :9].astype(np.float32)).clone(); X[:, 4] = (X[:, 4] - 6) / 2; X[:, 7:9] = X[:, 7:9] * 2 - 1
    return X, torch.from_numpy(np.concatenate(E).astype(np.int64))


# ---------------- graph networks (graph_train_cv_20261108.Net, no edge features, no context) ----------------
class GraphNet(nn.Module):
    def __init__(self, d=64, layers=5, ctx=4):
        super().__init__(); self.emb = nn.Linear(9, d); self.msg = nn.ModuleList(nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d)) for _ in range(layers))
        self.upd = nn.ModuleList(nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU()) for _ in range(layers)); self.drop = nn.Dropout(0.1)
        self.head = nn.Sequential(nn.Linear(2 * d + ctx, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 3)); self.d = d

    def pooled(self, x, e):
        h = self.emb(x); src = torch.cat([e[:, 0], e[:, 1]]); dst = torch.cat([e[:, 1], e[:, 0]])
        for m, u in zip(self.msg, self.upd):
            ms = m(torch.cat([h[src], h[dst]], 1)); ag = torch.zeros_like(h).index_add_(0, dst, ms); c = torch.zeros(len(h)).index_add_(0, dst, torch.ones(len(dst))).clamp_min(1)[:, None]; h = h + self.drop(u(torch.cat([h, ag / c], 1)))
        mean = h.mean(0, keepdim=True); var = ((h - mean) ** 2).mean(0, keepdim=True); return torch.cat([mean, (var + 1e-6).sqrt()], 1)

    def forward_image(self, graphs):
        if graphs: p = self.pooled(*_pack(graphs))
        else: p = torch.cat([torch.zeros(1, self.d), torch.full((1, self.d), math.sqrt(1e-6))], 1)   # empty-graph output, as predict() gave
        return self.head(torch.cat([p, torch.zeros(1, 4)], 1))[0]


class GraphNets:
    def __init__(self, pattern):
        self.paths = sorted(glob.glob(str(GM / pattern))); assert self.paths, pattern; self.nets = []
        for p in self.paths:
            ck = torch.load(p, map_location='cpu', weights_only=False); assert ck.get('no_ctx', True), p
            n = GraphNet(ck['d'], ck['layers']); n.load_state_dict(ck['model'], strict=True); n.eval(); self.nets.append(n)

    @torch.no_grad()
    def out(self, graphs):
        return np.mean([n.forward_image(graphs).double().numpy() for n in self.nets], 0)


# ---------------- simulation embeddings (sim_embed_eval_20261105) ----------------
class SimNet(nn.Module):
    def __init__(self, d, layers):
        super().__init__(); self.emb = nn.Linear(9, d); self.msg = nn.ModuleList(nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d)) for _ in range(layers))
        self.upd = nn.ModuleList(nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU()) for _ in range(layers)); self.drop = nn.Dropout(0.1)
        self.head = nn.Sequential(nn.Linear(2 * d, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 3))

    def pooled(self, x, e):
        h = self.emb(x); src = torch.cat([e[:, 0], e[:, 1]]); dst = torch.cat([e[:, 1], e[:, 0]])
        for m, u in zip(self.msg, self.upd):
            ms = m(torch.cat([h[src], h[dst]], 1)); ag = torch.zeros_like(h).index_add_(0, dst, ms); c = torch.zeros(len(h)).index_add_(0, dst, torch.ones(len(dst))).clamp_min(1)[:, None]; h = h + self.drop(u(torch.cat([h, ag / c], 1)))
        mean = h.mean(0, keepdim=True); var = ((h - mean) ** 2).mean(0, keepdim=True); return torch.cat([mean, (var + 1e-6).sqrt()], 1)[0]


class SimNets:
    def __init__(self, name='sim2_l8'):
        self.nets = []
        for p in sorted((XP / 'meta_ceiling_20261103/sweep/sim_pred').glob(f'{name}_s[0-9]*.pt')):
            ck = torch.load(p, map_location='cpu', weights_only=False); a = ck['args']; n = SimNet(a['d'], a['layers']); n.load_state_dict(ck['model']); n.eval(); self.nets.append(n)

    @torch.no_grad()
    def embed(self, graphs_by_set):
        out = []
        for n in self.nets:
            Z = [n.pooled(*_pack(g)).double().numpy() for g in graphs_by_set.values() if g]
            out.append(np.mean(Z, 0) if Z else np.full(2 * n.emb.out_features, np.nan))
        return out


# ---------------- token blocks (token_stats_cv_20261108.stats, the four blocks the heads read) ----------------
HT, WT = 124, 164; NTOK = HT * WT; K = 64; LAGS = (1, 2, 3, 4, 5, 6, 8, 10); PAD = 10; NSPEC = 24


def _geometry(dev):
    fy = torch.fft.fftfreq(HT, device=dev); fx = torch.fft.rfftfreq(WT, device=dev); r = torch.sqrt(fy[:, None] ** 2 + fx[None] ** 2)
    edges = torch.logspace(np.log10(1 / 62), np.log10(0.5), NSPEC + 1, device=dev); sb = torch.bucketize(r.flatten(), edges) - 1; ok = (sb >= 0) & (sb < NSPEC)
    hp, wp = HT + PAD, WT + PAD; dy = torch.arange(hp, device=dev); dy = torch.minimum(dy, hp - dy); dx = torch.arange(wp, device=dev); dx = torch.minimum(dx, wp - dx)
    rr = torch.sqrt(dy[:, None].float() ** 2 + dx[None].float() ** 2); ov = ((HT - dy[:, None]).clamp(min=1) * (WT - dx[None]).clamp(min=1)).float()
    keep = (dy[:, None] <= PAD) & (dx[None] <= PAD); return sb, ok, [((rr.round() == l) & keep) for l in LAGS], ov


@torch.inference_mode()
def token_blocks(t, dev='cpu', pca=XP / 'lane_gpu2_20261106/frozen_s46/pca.npz'):
    """t: (B, 124*164, 768) or (124*164, 768) float patch tokens. Returns float32 arrays std (B, 768), spectrum (B, 1536), cov_sqrt (B, 2080), autocorr (B, 6144)."""
    t = torch.as_tensor(t, dtype=torch.float32, device=dev); t = t[None] if t.ndim == 2 else t; B = len(t)
    z = np.load(pca); PM = torch.from_numpy(z['mean']).float().to(dev); PP = torch.from_numpy(z['basis']).float().to(dev); sb, ok, masks, ov = _geometry(dev)
    m = t.mean(1); d = t - m[:, None]; v = d.pow(2).mean(1); sd = (v + 1e-12).sqrt()
    Z = (t - PM) @ PP; Zc = Z - Z.mean(1, keepdim=True); C = Zc.transpose(1, 2) @ Zc / NTOK; iu = torch.triu_indices(K, K, device=dev); cov = C[:, iu[0], iu[1]]
    P = torch.fft.rfft2(Zc.reshape(B, HT, WT, K).permute(0, 3, 1, 2)).abs().pow(2).flatten(2)
    S = torch.zeros(B, K, NSPEC, device=dev); S.index_add_(2, sb[ok], P[:, :, ok]); spectrum = (S / S.sum(2, keepdim=True).clamp(min=1e-9)).flatten(1)
    g = d.reshape(B, HT, WT, 768).permute(0, 3, 1, 2)
    A = torch.fft.irfft2(torch.fft.rfft2(g, s=(HT + PAD, WT + PAD)).abs().pow(2), s=(HT + PAD, WT + PAD)) / ov; A0 = A[:, :, 0, 0].clamp(min=1e-9)
    autocorr = torch.stack([A[:, :, q].mean(-1) / A0 for q in masks], 2).flatten(1)
    return dict(std=sd.float().cpu().numpy(), spectrum=spectrum.float().cpu().numpy(), cov_sqrt=cov_sqrt(cov.float().cpu().numpy()), autocorr=autocorr.float().cpu().numpy())


def cov_sqrt(cov):
    """frozen_read_20261106.block('cov_sqrt'): symmetric square root of the 64 x 64 PCA-token covariance, upper triangle."""
    c = np.atleast_2d(cov); iu = np.triu_indices(K); M = torch.zeros(len(c), K, K, dtype=torch.float64); v = torch.from_numpy(np.asarray(c)).double(); M[:, iu[0], iu[1]] = v; M[:, iu[1], iu[0]] = v
    w, V = torch.linalg.eigh(M); R = (V * w.clamp(min=0).sqrt()[:, None]) @ V.transpose(1, 2); return R[:, iu[0], iu[1]].float().numpy()


# ---------------- frozen heads (frozen_read_20261106.BlockHead) ----------------
class BlockHead(nn.Module):
    def __init__(s, blocks, width, out):
        super().__init__(); s.ln = nn.ModuleList([nn.LayerNorm(d) for n, d in blocks]); D = sum(d for _, d in blocks); s.net = nn.Sequential(nn.Linear(D, width), nn.GELU(), nn.Linear(width, out))

    def forward(s, xs): return s.net(torch.cat([l(x) for l, x in zip(s.ln, xs)], 1))


class FrozenHeads:
    ARMS = ('std', 'spectrum', 'cov_sqrt', 'autocorr')

    def __init__(self, d=LANE / 'frozen_heads_full'):
        self.h = {}
        for arm in self.ARMS:
            L = []
            for p in sorted(Path(d).glob(f'{arm}_s[0-9]_e*.pt')):
                ck = torch.load(p, map_location='cpu', weights_only=False); n = BlockHead(ck['shapes'], 128, 3); n.load_state_dict(ck['model']); n.eval(); L.append((n, ck['blocks'], ck['mu'].double(), ck['sigma'].double()))
            assert len(L) == 15, (arm, len(L)); self.h[arm] = L

    @torch.no_grad()
    def logcv(self, blocks):
        """blocks: dict of (B, d) arrays. Returns {'fz_std': (B,), ...}: mean over 3 seeds x epochs 16-20 of the log CV prediction."""
        out = {}
        for arm, L in self.h.items():
            v = [((n([torch.as_tensor(np.atleast_2d(blocks[b]), dtype=torch.float32) for b in bl]).double() * sg + mu)[:, 1]).numpy() for n, bl, mu, sg in L]
            out[f'fz_{arm}'] = np.mean(v, 0)
        return out


# ---------------- parity check on training rows ----------------
def _stored_graphs(z, r):
    a, c = z['offs'][r], z['offs'][r + 1]
    if c == a: return None
    return z['X'][a:c], z['E'][z['eoff'][r]:z['eoff'][r + 1]] - a


def check(n=24):
    import json
    rng = np.random.default_rng(0); out = {}
    idx = np.load(XP / 'recipeA_analysis_20261028/hybrid_oof.npz')['indices'].astype(int); pos = {int(i): k for k, i in enumerate(idx)}; rows = np.sort(rng.choice(len(idx), n, replace=False))
    def peaks(d, r):
        for f in range(5):
            z = np.load(f'{d}/fold{f}/fold{f}_sg2_peaks.npz', allow_pickle=True); ix = list(z['indices'].astype(int))
            if int(idx[r]) in ix: k = ix.index(int(idx[r])); return z['peaks'][z['offsets'][k]:z['offsets'][k + 1]]
        return None
    # graphs: rebuilt from stored peaks vs stored graph files (cache-row order in the graph files)
    gfiles = {'noctx': ('meta_ceiling_20261103/sweep/cell_graphs.npz', 'green_track_20261028/external_srep/preonly', 'prep3'),
              'pre47': ('meta_ceiling_20261103/sweep/cell_graphs_pre47.npz', 'green_track_20261028/external_srep/s47_preonly', 'prep3'),
              's46q': ('lane_cv_20261108/graphs/tta_s46_tta4.npz', 'lane_cv_20261108/out/tta_peaks/s46_tta4', 'sp')}
    G = {}
    for key, (gf, pd, kind) in gfiles.items():
        z = np.load(XP / gf); worst = 0.0; same_n = 0
        for r in rows:
            g = graphs_from_peaks(peaks(XP / pd, r), kind); st = _stored_graphs(z, int(idx[r]))
            if not g and st is None: same_n += 1; continue
            if not g or st is None: continue
            X = np.concatenate([q[0] for q in g])
            if X.shape == st[0].shape: same_n += 1; worst = max(worst, float(np.abs(X - st[0][:, :X.shape[1]]).max()))
            G.setdefault(key, {})[int(r)] = g
        out[f'graph_{key}'] = dict(rows=int(n), same_node_count=same_n, max_abs_feature_diff=worst)
    # networks: per-image forward vs the all-rows nets' stored in-sample predictions (training-time batched predict)
    for key, pat, fp in (('s46q', 'full_tta_s46_tta4_s*_e*.pt', 'full_tta_s46_tta4_fullfit_pred.npz'), ('noctx', 'gfull_noctx_s0_s0_e*.pt', 'gfull_noctx_s0_fullfit_pred.npz')):
        if not (GM / fp).exists(): out[f'net_{key}'] = 'not built yet'; continue
        nets = GraphNets(pat); ref = np.load(GM / fp)['pred']; d = [float(np.abs(nets.out(G.get(key, {}).get(int(r), [])) - ref[r]).max()) for r in rows]
        out[f'net_{key}'] = dict(rows=len(d), max_abs_diff=max(d))
    # simulation embeddings vs stored sim2_l8_embed.npz (fold-detector graphs of the three sets)
    S = SimNets(); E = np.load(XP / 'meta_ceiling_20261103/sweep/sim_pred/sim2_l8_embed.npz'); zs = {k: np.load(XP / f) for k, f in (('noctx', 'meta_ceiling_20261103/sweep/cell_graphs.npz'), ('pre47', 'meta_ceiling_20261103/sweep/cell_graphs_pre47.npz'), ('pre48', 'meta_ceiling_20261103/sweep/cell_graphs_pre48.npz'))}
    worst = 0.0
    for r in rows:
        gb = {}
        for k, z in zs.items():
            st = _stored_graphs(z, int(idx[r])); gb[k] = [(st[0], st[1])] if st is not None else []
        emb = S.embed(gb)
        for s, e in enumerate(emb):
            ref = E[f's{s}'][r].astype(np.float64)
            if np.isfinite(ref).all(): worst = max(worst, float(np.abs(e - ref).max()))
    out['sim_embed'] = dict(rows=int(n), max_abs_diff_vs_float16_store=worst)
    # frozen heads on stored full-data blocks (frozen_own_f99) vs a batched forward is the same code path; report the values for the record
    FHs = FrozenHeads(); D = LANE / 'frozen_own_f99'; bl = {'std': np.load(D / 'std.npy', mmap_mode='r')[rows], 'spectrum': np.load(D / 'spectrum.npy', mmap_mode='r')[rows], 'autocorr': np.load(D / 'autocorr.npy', mmap_mode='r')[rows]}
    bl['cov_sqrt'] = cov_sqrt(np.load(D / 'cov.npy', mmap_mode='r')[rows]); stored_cs = D / 'cov_sqrt.npy'
    if stored_cs.exists(): out['cov_sqrt_vs_stored'] = float(np.abs(bl['cov_sqrt'] - np.load(stored_cs, mmap_mode='r')[rows]).max())
    lv = FHs.logcv(bl); out['frozen_heads'] = {k: [round(float(x), 4) for x in v[:3]] for k, v in lv.items()}
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'check': check()
    else: print(__doc__)
