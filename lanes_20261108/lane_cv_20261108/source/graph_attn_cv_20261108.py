"""CV lane copy (2026-11-08): FOLDS env limits folds, outputs to lane_cv_20261108/graph_runs; ROI read uses 380x300 windows. Otherwise unchanged.
CV-focused cell graph (2026-11-06). CV is a second moment, driven by a few large or small cells, while the network pools only mean and std. --tail adds max and min pooling over nodes; --metrics limits the loss to chosen metrics (1 = CV). Everything else as the plain graph network. Earlier text follows.
Window attention over the cell graphs (2026-11-05). The label comes from one region, and the legible windows only approximate it (IoU about 0.57). Here each window is pooled alone, a small attention picks among the windows of a frame, and the image label trains it: a soft ROI finder, read on the 25 expert boxes it never saw. --init NAME takes the encoder from a simulation-pretrained network; --freeze keeps it fixed, so only attention and head learn from labels. Without --attention the pooling is the usual one over all nodes, the reference on the same graphs. LOFO, target and pipeline as cell_graph_sweep. usage: cell_graph_cvtail_20261106.py NAME --graphs FILE [--attention] [--layers 5 --init sim2_l8 --freeze --seeds 3]. One GPU."""
import argparse, json, re, sys, time, warnings
import numpy as np
import torch
from torch import nn
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import stretch_nested_20261103 as N
ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--d', type=int, default=64); ap.add_argument('--layers', type=int, default=5); ap.add_argument('--epochs', type=int, default=25); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--seeds', type=int, default=3); ap.add_argument('--graphs', default='cell_graphs_k8_xy.npz'); ap.add_argument('--wd', type=float, default=1e-2); ap.add_argument('--drop', type=float, default=0.1); ap.add_argument('--attention', action='store_true'); ap.add_argument('--init', default=''); ap.add_argument('--freeze', action='store_true'); ap.add_argument('--tail', action='store_true'); ap.add_argument('--metrics', default='0,1,2'); A_ = ap.parse_args()
E = C.E; K = list(C.K); b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']; g = E.donors(b['ids']); O = C.X / 'meta_ceiling_20261103/sweep'; N_ = len(y); torch.set_num_threads(2); W_, H_ = 380, 300
z = np.load(O / A_.graphs); X = torch.from_numpy(z['X'][:, :9]); ED_ = torch.from_numpy(z['E'].astype(np.int64)).cuda(); WID_ = torch.from_numpy(z['wid'].astype(np.int64)).cuda(); OFF = z['offs']; EOFF = z['eoff']; print('nodes', X.shape, 'windows per frame', round(float(np.mean([len(np.unique(z['wid'][OFF[i]:OFF[i + 1]])) for i in range(0, 8954, 50) if OFF[i + 1] > OFF[i]])), 2), flush=True)
gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); a_, b_, w_ = -0.2084, 0.5989, 0.28; have = gf['f_active10_valid'] & ~gf['seen_leaked'] & np.isfinite(gf['f_active10']); good = ~gf['seen_leaked']; f6 = np.nan_to_num(gf['f_active10'].astype(float), nan=float(np.nanmedian(gf['f_active10']))); H = B.copy(); H[have, 2] = np.exp((1 - w_) * np.log(B[have, 2]) + w_ * (a_ + b_ * f6[have]))
REF = np.full_like(H, np.nan)
for f in range(5): REF[fold == f] = N.apply_stretch(H[fold == f], N.fit_stretch(H, y, (fold != f) & good))
T_ = np.log(np.where(y > 0, y, 1.0)) - np.log(H); MASK = (y > 0).astype(np.float32); LM = np.zeros(3, np.float32); LM[[int(q) for q in A_.metrics.split(',')]] = 1.0; MASK = MASK * LM; FR = idx.astype(int)
def norm(X):
    F = X.clone(); F[:, 4] = (F[:, 4] - 6) / 2; F[:, 7:9] = F[:, 7:9] * 2 - 1; return F
Xn = norm(X).cuda(); Tt = torch.from_numpy(T_.astype(np.float32)).cuda(); Mt = torch.from_numpy(MASK).cuda()
class Net(nn.Module):
    def __init__(self, d=A_.d, layers=A_.layers):
        super().__init__(); self.emb = nn.Linear(9, d); self.msg = nn.ModuleList(nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d)) for _ in range(layers)); self.upd = nn.ModuleList(nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU()) for _ in range(layers)); self.drop = nn.Dropout(0.0 if A_.freeze else A_.drop); self.attn = nn.Sequential(nn.Linear(2 * d, 64), nn.GELU(), nn.Linear(64, 1)); self.out = nn.Sequential(nn.Linear((4 if A_.tail else 2) * d, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 3)); nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias); nn.init.zeros_(self.attn[-1].weight); nn.init.zeros_(self.attn[-1].bias)
    def enc(self, x, e):
        h = self.emb(x); src = torch.cat([e[:, 0], e[:, 1]]); dst = torch.cat([e[:, 1], e[:, 0]])
        for m, u in zip(self.msg, self.upd):
            ms = m(torch.cat([h[src], h[dst]], 1)); ag = torch.zeros_like(h).index_add_(0, dst, ms); c = torch.zeros(len(h), device=h.device).index_add_(0, dst, torch.ones(len(dst), device=h.device)).clamp_min(1)[:, None]; h = h + self.drop(u(torch.cat([h, ag / c], 1)))
        return h
    def pool(self, h, g_, n):
        cnt = torch.zeros(n, device=h.device).index_add_(0, g_, torch.ones(len(g_), device=h.device)).clamp_min(1)[:, None]; mean = torch.zeros(n, h.shape[1], device=h.device).index_add_(0, g_, h) / cnt; var = torch.zeros(n, h.shape[1], device=h.device).index_add_(0, g_, (h - mean[g_]) ** 2) / cnt; return torch.cat([mean, (var + 1e-6).sqrt()], 1)
    def forward(self, x, e, frame, wid, nf):
        with torch.set_grad_enabled(self.training and not A_.freeze): h = self.enc(x, e)
        if not A_.attention:
            z_ = self.pool(h, frame, nf)
            if A_.tail:
                ix = frame[:, None].expand(-1, h.shape[1]); mx = torch.full((nf, h.shape[1]), -1e9, device=h.device).scatter_reduce(0, ix, h, 'amax'); mn = torch.full((nf, h.shape[1]), 1e9, device=h.device).scatter_reduce(0, ix, h, 'amin'); mx = mx.masked_fill(mx < -1e8, 0.0); mn = mn.masked_fill(mn > 1e8, 0.0); z_ = torch.cat([z_, mx, mn], 1)
            return self.out(z_), None
        key = frame * 100 + wid; uq, inv = torch.unique(key, return_inverse=True); zw = self.pool(h, inv, len(uq)); fw = uq // 100; a = self.attn(zw)[:, 0]; ex = (a - a.detach().max()).exp(); tot = torch.zeros(nf, device=h.device).index_add_(0, fw, ex).clamp_min(1e-9); wt = ex / tot[fw]; zf = torch.zeros(nf, zw.shape[1], device=h.device).index_add_(0, fw, zw * wt[:, None]); return self.out(zf), (fw, uq % 100, wt)
def batch(rows):
    xs, es, fr, ws = [], [], [], []; base = 0
    for k, r in enumerate(rows):
        i = FR[r]; a, c = OFF[i], OFF[i + 1]
        if c == a: continue
        ea, ec = EOFF[i], EOFF[i + 1]; xs.append(Xn[a:c]); es.append(ED_[ea:ec] - a + base); ws.append(WID_[a:c]); fr.append(torch.full((c - a,), k, device=Xn.device)); base += c - a
    return (torch.cat(xs), torch.cat(es), torch.cat(fr), torch.cat(ws)) if xs else None
def predict(net, rows, ATT=None):
    net.eval(); out = np.zeros((N_, 3))
    with torch.no_grad():
        for s in range(0, len(rows), 128):
            rr = rows[s:s + 128]; bb = batch(rr)
            if bb is None: continue
            p, at = net(*bb, len(rr)); out[rr] = p.cpu().numpy()
            if at is not None and ATT is not None:
                fw, wd, wt = (q.cpu().numpy() for q in at)
                for u, v, q in zip(fw, wd, wt): ATT.setdefault(int(rr[u]), {}); ATT[int(rr[u])][int(v)] = ATT[int(rr[u])].get(int(v), 0.0) + float(q)
    return out
ATT = {}
def run(f, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); tr = np.where((fold != f) & good & (np.diff(OFF)[FR] > 0))[0]; net = Net().cuda()
    if A_.init:
        ck = torch.load(O / 'sim_pred' / f'{A_.init}_s{seed % 3}.pt', map_location='cpu', weights_only=False); net.load_state_dict({k: v for k, v in ck['model'].items() if k.split('.')[0] in ('emb', 'msg', 'upd')}, strict=False)
    if A_.freeze:
        for m in (net.emb, net.msg, net.upd):
            for q in m.parameters(): q.requires_grad = False
    opt = torch.optim.AdamW([q for q in net.parameters() if q.requires_grad], A_.lr, weight_decay=A_.wd); steps = A_.epochs * (len(tr) // 64); sch = torch.optim.lr_scheduler.OneCycleLR(opt, A_.lr, total_steps=steps, pct_start=0.1); snaps = np.zeros((N_, 3)); ns = 0; t0 = time.time(); allrows = np.arange(N_)
    for ep in range(A_.epochs):
        net.train(); pm = rng.permutation(tr); tot = 0.0
        for s in range(len(pm) // 64):
            rr = pm[s * 64:(s + 1) * 64]; bb = batch(rr); ri = torch.from_numpy(rr).cuda(); p, _ = net(*bb, len(rr)); loss = ((p - Tt[ri]).abs() * Mt[ri]).sum() / Mt[ri].sum(); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step(); tot += float(loss)
        if ep >= A_.epochs - 5:
            snaps += predict(net, allrows); ns += 1
        if ep == A_.epochs - 1: print(f'{A_.name} fold {f} seed {seed} loss {tot / (len(pm) // 64):.4f} {time.time() - t0:.0f}s', flush=True)
    predict(net, np.where(fold == f)[0], ATT); return snaps / ns
import os; FOLDS = [int(c) for c in os.environ.get("FOLDS", "01234")]; pred = {f: np.mean([run(f, seed=q) for q in range(A_.seeds)], 0) for f in FOLDS}; rep = dict(config=vars(A_))
def pipe(P_all, ms):
    P = np.full_like(H, np.nan)
    for f in range(5):
        tr = (fold != f) & good; C_ = H.copy()
        for j in ms: C_[:, j] = H[:, j] * np.exp(P_all[f][:, j])
        P[fold == f] = N.apply_stretch(C_[fold == f], N.fit_stretch(C_, y, tr))
    return P
def report(nm, p):
    pb = E.paired_bootstrap(E.ape(p, y), E.ape(REF, y), g); s, s0 = E.official(E.ape(p, y)), E.official(E.ape(REF, y)); per = [round(E.official(E.ape(p[fold == f], y[fold == f]))['mean'] - E.official(E.ape(REF[fold == f], y[fold == f]))['mean'], 3) for f in range(5)]; o = dict(score=round(s['mean'], 3), per_delta={k: round(s[k] - s0[k], 3) for k in K}, delta=round(pb['delta'], 4), ci=[round(v, 4) for v in pb['ci95']], folds=per); print(nm, json.dumps(o), flush=True); return o
if len(FOLDS) == 5:
    for nm, ms in dict(CV=(1,), HEX=(2,), CV_HEX=(1, 2)).items(): rep[nm] = report(A_.name + ' ' + nm, pipe(pred, ms))
if A_.attention and 'xy' in z.files:
    ids = [re.sub(r'\s+', ' ', i.strip()) for i in b['ids']]; pos = {n: i for i, n in enumerate(ids)}; ann = json.load(open(C.X.parent / 'datasets/clear_ec_official/green_eval/annotations.json')); xy = z['xy']; Xr = z['X'][:, 7:9]; wid = z['wid']; rows = []; rng = np.random.default_rng(0)
    def iou(w, bx):
        ix = max(0, min(w[0] + W_, bx[2]) - max(w[0], bx[0])); iy = max(0, min(w[1] + H_, bx[3]) - max(w[1], bx[1])); it = ix * iy; return it / (W_ * H_ + (bx[2] - bx[0]) * (bx[3] - bx[1]) - it)
    for n, e in ann.items():
        rw = pos[n]; at = ATT.get(rw); i = FR[rw]; a, c = OFF[i], OFF[i + 1]
        if not at or c == a: continue
        ws = {}
        for v in np.unique(wid[a:c]):
            q = a + int(np.argmax(wid[a:c] == v)); ws[int(v)] = (float(xy[q, 0] - Xr[q, 0] * W_), float(xy[q, 1] - Xr[q, 1] * H_))
        ious = {v: iou(p, e['roi_box_mha']) for v, p in ws.items()}; top = max(at, key=at.get); best = max(ious, key=ious.get); rk = sorted(at, key=at.get, reverse=True).index(best); rows.append((ious[top], ious.get(0, np.nan), float(np.mean(list(ious.values()))), ious[best], rk, max(at.values()) / sum(at.values()), len(ws)))
    R = np.array(rows); rep['roi'] = dict(frames=len(R), iou_top_attention=round(float(R[:, 0].mean()), 3), iou_first_legible=round(float(np.nanmean(R[:, 1])), 3), iou_mean_window=round(float(R[:, 2].mean()), 3), iou_best_window=round(float(R[:, 3].mean()), 3), rank_of_best_window=round(float(R[:, 4].mean()), 2), top_weight=round(float(R[:, 5].mean()), 3), windows=round(float(R[:, 6].mean()), 1)); print('expert boxes', json.dumps(rep['roi']), flush=True); ww = [max(v.values()) / sum(v.values()) for v in ATT.values()]; rep['top_weight_all_frames'] = round(float(np.mean(ww)), 3); print('top attention share, all frames', rep['top_weight_all_frames'], flush=True)
LO = C.X / 'lane_cv_20261108/graph_runs'; LO.mkdir(exist_ok=True); np.savez_compressed(LO / f'{A_.name}.npz', **{f'f{f}': pred[f].astype(np.float32) for f in pred}); json.dump(rep, open(LO / f'{A_.name}.json', 'w'), indent=1); print('done', flush=True)
