"""Can an image alone find its partner, and does a neighbour's residual help? (2026-11-04). Label-free inputs: all-data SSL5 token
statistics (retrieval/ssl_tokens.npz) and raw descriptors (retrieval/raw_desc.npz). Known pairs supervise only the matcher.
  rank    median rank of the partner among all other frames: whitened PCA of token statistics, then a matcher trained per fold
          (InfoNCE on known pairs with neither end in the held fold; pairs = new-format fellow eyes, old-format adjacent IDs).
  knn     OOF correction p' = B exp(beta s), s = similarity-weighted mean of the OOF log residual of the k nearest gallery rows,
          gallery = other folds only, beta fit on the other folds. A control permutes the gallery residuals. Also on old-format rows
          alone, and by top similarity quantile (would a confident match pay?).
Prize if the true neighbour were known: 9.039 -> 8.449 on old-format rows with one (id_sequence_probe). CPU, threads = clamp(16 - load1 - 1, 2, 12)."""
import json, os, re, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
OUT = '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/retrieval'
torch.set_num_threads(int(min(12, max(2, 16 - os.getloadavg()[0] - 1))))
K = list(E.K)
b = E.load_base(); ids = [re.sub(r'\s+', ' ', i.strip()) for i in b['ids']]; y, B, fold = b['y'], b['B'], b['fold']; N = len(ids)
r = np.log(np.where(y > 0, y, np.nan)) - np.log(B)
old, new = {}, {}
for n, i in enumerate(ids):
    m = re.fullmatch(r'(\d{4})-(\d{2})', i)
    if m: old[(int(m.group(2)), int(m.group(1)))] = n; continue
    m = re.match(r'(\d{4})-(\d{4})[ -]*([O0][DS])', i)
    if m: new[(int(m.group(1)), int(m.group(2)), m.group(3).replace('0', 'O'))] = n
new_pairs = np.array([(new[k], new[(k[0], k[1], 'OS')]) for k in new if k[2] == 'OD' and (k[0], k[1], 'OS') in new])
old_pairs = np.array([(old[k], old[(k[0], k[1] + 1)]) for k in old if (k[0], k[1] + 1) in old])
oldrows = np.array(sorted(old.values()))
print('pairs new', len(new_pairs), 'old', len(old_pairs), flush=True)
z = np.load(f'{OUT}/ssl_tokens.npz'); pos = {int(i): n for n, i in enumerate(z['idx'])}; order = np.array([pos[int(i)] for i in b['idx']])
Mn, Sd = z['mean'][order].astype(np.float32), z['std'][order].astype(np.float32); Co = z['coarse'][order].astype(np.float32)
d = np.load(f'{OUT}/raw_desc.npz'); pos = {int(i): n for n, i in enumerate(d['idx'])}; o2 = np.array([pos[int(i)] for i in b['idx']]); H = d['H'][o2].astype(np.float32)
def pca(X, q):
    X = torch.from_numpy(X); X = X - X.mean(0)
    U, S, _ = torch.pca_lowrank(X, q=q, center=False, niter=4)
    return (U * (X.shape[0] - 1) ** 0.5).numpy()                 # whitened
def unit(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
P_ms = pca(np.c_[Mn, Sd], 512); P_co = pca(Co, 256); P_h = pca(H, 32)
print('pca done', flush=True); del Co
def ranks(Q, pr, rng, n=600):
    pr = np.r_[pr, pr[:, ::-1]]; pr = pr[rng.permutation(len(pr))[:n]]
    s = Q[pr[:, 0]] @ Q.T; s[np.arange(len(pr)), pr[:, 0]] = -9
    rk = (s > s[np.arange(len(pr)), pr[:, 1]][:, None]).sum(1) + 1
    return dict(n=len(pr), median=int(np.median(rk)), top1=round(float((rk == 1).mean()), 4), top10=round(float((rk <= 10).mean()), 4), top100=round(float((rk <= 100).mean()), 4))
rng = np.random.default_rng(0); rep = dict(threads=torch.get_num_threads(), rank={}, knn={})
for nm, Q in dict(ms64=P_ms[:, :64], ms256=P_ms[:, :256], ms512=P_ms, coarse256=P_co, ms256_coarse256=np.c_[P_ms[:, :256], P_co], ms256_hist=np.c_[P_ms[:, :256], P_h]).items():
    Q = unit(Q); rep['rank'][nm] = dict(new=ranks(Q, new_pairs, rng), old=ranks(Q, old_pairs, rng)); print(nm, json.dumps(rep['rank'][nm]), flush=True)
X = np.c_[P_ms[:, :256], P_co[:, :128], P_h].astype(np.float32); Xt = torch.from_numpy(X)
class Net(torch.nn.Module):
    def __init__(self, n):
        super().__init__(); self.f = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(n, 512), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(512, 128)); self.t = torch.nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))
    def forward(self, x): return F.normalize(self.f(x), dim=1)
def train(pairs, seed, epochs=150):
    torch.manual_seed(seed); net = Net(X.shape[1]); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2); P = torch.from_numpy(pairs); bs = 512
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * (len(P) // bs))
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(P))
        for s in range(len(P) // bs):
            q = P[perm[s:s + bs] if False else perm[s * bs:(s + 1) * bs]]
            a, c = net(Xt[q[:, 0]]), net(Xt[q[:, 1]]); lg = a @ c.T * net.t.exp().clamp(max=50); t = torch.arange(len(q))
            loss = (F.cross_entropy(lg, t) + F.cross_entropy(lg.T, t)) / 2
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad(): return net(Xt).numpy(), float(loss)
all_pairs = np.r_[new_pairs, old_pairs]
embed = {}
for f in range(5):
    tr = all_pairs[(fold[all_pairs[:, 0]] != f) & (fold[all_pairs[:, 1]] != f)]
    E_f, l = train(tr, 100 + f); embed[f] = E_f
    te_new = new_pairs[(fold[new_pairs[:, 0]] == f) & (fold[new_pairs[:, 1]] == f)]; te_old = old_pairs[(fold[old_pairs[:, 0]] == f) | (fold[old_pairs[:, 1]] == f)]
    rep['rank'][f'matcher_f{f}'] = dict(train_pairs=len(tr), loss=round(l, 3), new=ranks(E_f, te_new, rng), old=ranks(E_f, te_old, rng)); print(f'matcher f{f}', json.dumps(rep['rank'][f'matcher_f{f}']), flush=True)
def knn(Q_of, k, tau=0.1, perm=None, name=''):
    """Q_of(f): unit embedding for fold f. Returns s (N, 3) and the top similarity per row."""
    s = np.full((N, 3), np.nan); top = np.zeros(N)
    R = r if perm is None else r[perm]
    for f in range(5):
        Q = Q_of(f); q = np.where(fold == f)[0]; g = np.where(fold != f)[0]
        S = torch.from_numpy(Q[q]) @ torch.from_numpy(Q[g]).T
        v, ix = S.topk(k, 1); w = F.softmax(v / tau, 1).numpy(); ix = g[ix.numpy()]; top[q] = v[:, 0].numpy()
        for j in range(3):
            x = R[ix, j]; ok = np.isfinite(x); s[q, j] = np.nansum(w * np.where(ok, x, 0), 1) / np.maximum((w * ok).sum(1), 1e-6)
    return s, top
def apply(s, rows=None):
    p = B.copy(); betas = []
    for f in range(5):
        tr = fold != f; te = fold == f
        for j in range(3):
            ok = tr & np.isfinite(r[:, j]) & np.isfinite(s[:, j]); be = float((s[ok, j] * r[ok, j]).sum() / max((s[ok, j] ** 2).sum(), 1e-9)); p[te, j] = B[te, j] * np.exp(be * s[te, j]); betas.append(be)
    return p, betas
def score(p, rows):
    a, c = E.official(E.ape(B[rows], y[rows])), E.official(E.ape(p[rows], y[rows]))
    return {k: round(c[k] - a[k], 4) for k in K + ['mean']}
for nm, Q_of in dict(whiten=lambda f: unit(np.c_[P_ms[:, :256], P_h]).astype(np.float32), matcher=lambda f: embed[f]).items():
    for k in (1, 3, 10, 30, 100):
        s, top = knn(Q_of, k); p, be = apply(s)
        perm = np.random.default_rng(k).permutation(N); s0, _ = knn(Q_of, k, perm=perm); p0, _ = apply(s0)
        allr = np.arange(N); e = dict(all=score(p, allr), old=score(p, oldrows), control_all=score(p0, allr), beta_mean=[round(float(np.mean(be[j::3])), 3) for j in range(3)], folds=[round(score(p, np.where(fold == f)[0])['mean'], 3) for f in range(5)])
        qs = np.quantile(top, [.5, .9, .99]); e['by_top_sim'] = {f'q{int(q * 100)}': score(p, np.where(top >= t)[0])['mean'] for q, t in zip((.5, .9, .99), qs)}
        rep['knn'][f'{nm}_k{k}'] = e; print(nm, k, json.dumps(e), flush=True)
json.dump(rep, open(f'{OUT}/matcher_report.json', 'w'), indent=1); np.savez_compressed(f'{OUT}/matcher_embed.npz', **{f'f{f}': embed[f].astype(np.float16) for f in embed})
print('done', flush=True)
