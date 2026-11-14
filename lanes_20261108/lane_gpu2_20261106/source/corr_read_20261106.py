"""F1b, lane GPU 2 (2026-11-06): does a token statistic add to the best pipeline? Ridge on whitened PCs of the block against the honest residual of BEST (log y - log H - BEST from the row's own fold).
For fold f: fit on the other folds' rows. Number of PCs and penalty chosen per metric by four inner folds on those rows. Training rows of fold f get inner out-of-fold values, so the stretch in best_pipeline is fit on honest corrections. Read with best_pipeline, correction rule.
PCA is fit once per block on all rows, without labels. Usage: corr_read_20261106.py [block,block,...]"""
import json, os, sys, time
from pathlib import Path
os.chdir('/workspace/clear-ec-research')
import numpy as np
import torch
X = Path('experiments'); sys.path.insert(0, str(X / 'meta_ceiling_20261103/source'))
import best_pipeline_20261106 as BP
LANE = X / 'lane_gpu2_20261106'; OUT = LANE / 'frozen_s46'; READS = LANE / 'reads/v2/corr'; READS.mkdir(parents=True, exist_ok=True)
E = BP.E; y, fold = BP.y, BP.fold; good = BP.good & (y > 0).all(1); K_ = (16, 64, 256); A_ = (1e1, 1e2, 1e3, 1e4)
BLOCKS = ['std', 'mean', 'skew', 'kurt', 'quant', 'cov', 'cov_sqrt', 'autocorr', 'spectrum', 'multiscale', 'localhet', 'winstat', 'legible_pred', 'legible_std', 'cls']
honest = np.log(y.clip(1e-9)) - np.log(BP.H)
for f in range(5): honest[fold == f] -= BP.BEST[f][fold == f]


def pcs(name, kmax=256):
    a = np.load(OUT / f'{name}.npy', mmap_mode='r'); a = torch.from_numpy(np.asarray(a)).double().cuda(); a = a - a.mean(0); a = a / (a.std(0).mean() + 1e-9)
    if a.shape[1] <= 2500:
        w, V = torch.linalg.eigh(a.T @ a); V = V.flip(1)[:, :min(kmax, a.shape[1])]; Z = a @ V
    else:
        U, S, _ = torch.pca_lowrank(a, q=min(kmax, a.shape[1]), center=False, niter=4); Z = U * S
    Z = Z / (Z.std(0) + 1e-9); return Z.float()


def ridge(Z, r, tr, te, alpha):
    Zt, rt = Z[tr], r[tr]; G = Zt.T @ Zt + alpha * torch.eye(Z.shape[1], device=Z.device); return Z[te] @ torch.linalg.solve(G, Zt.T @ rt)


def select(Z, r, rows, rng):
    """Per metric, best (k, alpha) by inner-fold absolute log error."""
    perm = rng.permutation(len(rows)); parts = [rows[perm[i::4]] for i in range(4)]; best = np.full((len(K_), len(A_), 3), np.nan)
    for a_, k in enumerate(K_):
        for b_, al in enumerate(A_):
            er = torch.zeros(3, device=Z.device)
            for p in range(4):
                te = parts[p]; tr = np.concatenate([parts[q] for q in range(4) if q != p]); pr = ridge(Z[:, :k], r, torch.from_numpy(tr).cuda(), torch.from_numpy(te).cuda(), al); er += (r[te] - pr).abs().sum(0)
            best[a_, b_] = er.cpu().numpy()
    base = r[rows].abs().sum(0).cpu().numpy(); out = []
    for j in range(3):
        c = best[:, :, j]; i = np.unravel_index(np.nanargmin(c), c.shape); out.append((K_[i[0]], A_[i[1]], float(np.nanmin(c) / base[j])))
    return out


def run(name):
    t0 = time.time(); Z = pcs(name); r = torch.from_numpy(honest).float().cuda(); N = len(y); cor = {f: np.zeros((N, 3)) for f in range(5)}; chosen = {}
    for f in range(5):
        tr = np.where((fold != f) & good)[0]; rng = np.random.default_rng(f); sel = select(Z, r, tr, rng); chosen[f] = sel; rest = np.where(~((fold != f) & good))[0]
        for j, (k, al, _) in enumerate(sel):
            if al is None: continue
            pr = ridge(Z[:, :k], r[:, [j]], torch.from_numpy(tr).cuda(), torch.from_numpy(rest).cuda(), al); cor[f][rest, j] = pr[:, 0].cpu().numpy()
            perm = rng.permutation(len(tr))
            for p in range(4):
                te = tr[perm[p::4]]; t2 = tr[np.setdiff1d(perm, perm[p::4], assume_unique=False)]; cor[f][te, j] = ridge(Z[:, :k], r[:, [j]], torch.from_numpy(t2).cuda(), torch.from_numpy(te).cuda(), al)[:, 0].cpu().numpy()
    BP.save(READS / f'{name}.npz', cor); o = BP.report(name, BP.pipe(BP.add(BP.BEST, cor)), BP.REF_BEST); o['chosen'] = {f: [(c[0], c[1], round(c[2], 4)) for c in v] for f, v in chosen.items()}; o['sec'] = round(time.time() - t0, 1)
    (READS / 'corr.jsonl').open('a').write(json.dumps(o) + '\n'); del Z; torch.cuda.empty_cache(); return o


if __name__ == '__main__':
    for n in (sys.argv[1].split(',') if len(sys.argv) > 1 else BLOCKS): run(n)
