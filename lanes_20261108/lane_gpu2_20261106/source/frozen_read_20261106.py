"""Frozen read of pooling statistics and head forms, lane GPU 2 (2026-11-06). Gate in ../protocol.json.
Features: frozen_s46/<block>.npy, one row per frame in hybrid-OOF order, each row from the encoder that held it out. A head trained on the other folds' rows is read on the held-out fold.
Each seed: mean APE over epochs 16-20. Arm: mean over seeds. Paired donor bootstrap against the reference on the same rows.
Usage: frozen_read_20261106.py F1 [arm,arm,...] [--folds 0,4] [--seeds 0,1,2] | F2 [head,...] [--folds ...] | show"""
import json, math, os, sys, time
from pathlib import Path
os.chdir('/workspace/clear-ec-research')
import numpy as np
import torch
from torch import nn
X = Path('experiments'); sys.path.insert(0, str(X / 'eval_harness_20261031/source'))
import eval_oof_20261031 as E
LANE = X / 'lane_gpu2_20261106'; OUT = LANE / 'frozen_s46'; READS = LANE / 'reads/v2'; READS.mkdir(exist_ok=True, parents=True)
K = ['CD', 'CV', 'HEX']; bb = E.load_base(); y = bb['y']; fold = bb['fold']; gp = E.donors(bb['ids']); good = (y > 0).all(1)
ARMS = dict(std=['std'], mean_std=['std', 'mean'], dist=['std', 'skew', 'kurt'], quant=['std', 'quant'], cov=['std', 'cov'], cov_sqrt=['std', 'cov_sqrt'], autocorr=['std', 'autocorr'], spectrum=['std', 'spectrum'], multiscale=['std', 'multiscale'], localhet=['std', 'localhet'], winhead=['std', 'winstat', 'legible_pred', 'legible_std'], cls=['std', 'cls'], all=['std', 'mean', 'skew', 'kurt', 'quant', 'cov_sqrt', 'autocorr', 'spectrum', 'multiscale', 'localhet', 'winstat', 'legible_pred', 'legible_std', 'cls'])
Q_ = 64


def block(k):
    if k == 'cov_sqrt':
        p = OUT / 'cov_sqrt.npy'
        if not p.exists():
            c = np.load(OUT / 'cov.npy', mmap_mode='r'); iu = np.triu_indices(Q_); out = np.zeros(c.shape, np.float32); dev = 'cuda'
            for s in range(0, len(c), 512):
                M = torch.zeros(min(512, len(c) - s), Q_, Q_, device=dev, dtype=torch.float64); v = torch.from_numpy(np.asarray(c[s:s + 512])).to(dev).double(); M[:, iu[0], iu[1]] = v; M[:, iu[1], iu[0]] = v
                w, V = torch.linalg.eigh(M); R = (V * w.clamp(min=0).sqrt()[:, None]) @ V.transpose(1, 2); out[s:s + 512] = R[:, iu[0], iu[1]].float().cpu().numpy()
            np.save(p, out)
        return np.load(p, mmap_mode='r')
    return np.load(OUT / f'{k}.npy', mmap_mode='r')


SMALL = ('winstat', 'legible_pred'); SLOG = ('skew', 'kurt')


def features(names, tr):
    """Raw scale per block, its own LayerNorm in the head. skew, kurtosis signed log. Small blocks of unlike units z-scored on training rows, no LayerNorm."""
    out = []
    for n in names:
        a = np.asarray(block(n)).astype(np.float32)
        if n in SLOG: a = np.sign(a) * np.log1p(np.abs(a))
        if n in SMALL: a = np.clip((a - a[tr].mean(0)) / (a[tr].std(0) + 1e-6), -8, 8)
        out.append((n, torch.from_numpy(a).cuda()))
    return out


class BlockHead(nn.Module):
    def __init__(s, blocks, width, out):
        super().__init__(); s.ln = nn.ModuleList([nn.Identity() if n in SMALL else nn.LayerNorm(d) for n, d in blocks]); D = sum(d for _, d in blocks); s.net = nn.Sequential(nn.Linear(D, width), nn.GELU(), nn.Linear(width, out))
    def forward(s, xs): return s.net(torch.cat([l(x) for l, x in zip(s.ln, xs)], 1))


def ratio(step, updates, sched=10):
    if step < updates: return (step + 1) / updates
    phase = min(1., (step - updates) / max(1, updates * (sched - 1) - 1)); return .1 + .9 * .5 * (1 + math.cos(math.pi * phase))


def recipe_loss(z, zt, sigma):
    return 0.5 * (z - zt).abs().mean() + 2.5 * torch.expm1((z - zt) * sigma).abs().mean()


class Dist:
    """Distribution over standardized log target, decode by the MAPE optimum: the median of the predicted distribution weighted by 1/value."""
    def __init__(s, kind, sigma, mu):
        s.kind = kind; s.sigma, s.mu = sigma, mu; s.nb = 64 if kind == 'hlg' else 19; s.out = 3 * s.nb
        if kind == 'hlg':
            s.e = torch.linspace(-4.5, 4.5, s.nb + 1).cuda(); s.c = (s.e[1:] + s.e[:-1]) / 2; s.sm = 1.5 * float(s.e[1] - s.e[0])
        else:
            s.tau = torch.linspace(.05, .95, s.nb).cuda()
    def loss(s, o, zt):
        o = o.reshape(len(o), 3, s.nb)
        if s.kind == 'hlg':
            cdf = torch.special.ndtr((s.e[None, None] - zt[:, :, None]) / s.sm); p = cdf[:, :, 1:] - cdf[:, :, :-1]; p = p / p.sum(2, keepdim=True).clamp(min=1e-6)
            return -(p * torch.log_softmax(o, 2)).sum(2).mean()
        o = o.sort(2).values if False else o
        d = zt[:, :, None] - o; return (torch.maximum(s.tau * d, (s.tau - 1) * d)).mean()
    def decode(s, o):
        """Returns standardized log prediction, MAPE-optimal under the predicted distribution."""
        o = o.reshape(len(o), 3, s.nb)
        if s.kind == 'hlg':
            z = s.c[None, None].expand(o.shape); p = torch.softmax(o, 2)
        else:
            z = o.sort(2).values; p = torch.ones_like(z) / s.nb
        w = p * torch.exp(-z * s.sigma[:, None]); w = w / w.sum(2, keepdim=True); i = (w.cumsum(2) >= .5).float().argmax(2, keepdim=True); return z.gather(2, i).squeeze(2)


def fit(Xa, tr, va, seed, kind='recipe', width=128, epochs=20):
    ly = torch.from_numpy(np.log(y[tr])).float().cuda(); mu, sigma = ly.mean(0), ly.std(0, unbiased=False); zt_all = (ly - mu) / sigma; dist = Dist(kind, sigma, mu) if kind in ('hlg', 'quantile') else None; shapes = [(n, a.shape[1]) for n, a in Xa]
    torch.manual_seed(seed); head = BlockHead(shapes, width, dist.out if dist else 3).cuda(); nn.init.normal_(head.net[-1].weight, std=1e-3); nn.init.zeros_(head.net[-1].bias)
    if kind == 'quantile':
        with torch.no_grad(): head.net[-1].bias.copy_(torch.special.ndtri(dist.tau).repeat(3))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4, fused=True); t_ = torch.from_numpy(tr).cuda(); v_ = torch.from_numpy(va).cuda(); Xt = [a[t_] for _, a in Xa]; Xv = [a[v_] for _, a in Xa]; updates = math.ceil(len(tr) / 16); step = 0; keep = []
    for ep in range(epochs):
        order = torch.from_numpy(np.random.default_rng(seed * 1000 + ep).permutation(len(tr))).cuda()
        for i in range(0, len(tr), 16):
            b_ = order[i:i + 16]; r = ratio(step, updates)
            for g in opt.param_groups: g['lr'] = 1e-3 * r
            o = head([a[b_] for a in Xt]); loss = dist.loss(o, zt_all[b_]) if dist else recipe_loss(o, zt_all[b_], sigma)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); step += 1
        if ep >= epochs - 5:
            with torch.inference_mode():
                o = head(Xv); z = dist.decode(o) if dist else o
                keep.append((z * sigma + mu).double().cpu().numpy())
    return keep


def read_arm(name, feats, folds, seeds, kind='recipe', width=128):
    """Returns {fold: dict(e=rows x 3 APE mean over seeds and epochs, lg=mean log prediction)}."""
    res = {}
    for f in folds:
        tr = np.where((fold != f) & good)[0]; va = np.where(fold == f)[0]; Xa = features(feats, tr); E_ = []; L_ = []; t0 = time.time()
        for s in seeds:
            k = fit(Xa, tr, va, s, kind, width); E_.append(np.mean([E.ape(np.exp(q), y[va]) for q in k], 0)); L_.append(np.mean(k, 0))
        res[f] = dict(rows=va, e=np.mean(E_, 0), lg=np.mean(L_, 0), per_seed=[round(E.official(q)['mean'], 4) for q in E_], D=sum(a.shape[1] for _, a in Xa), sec=round(time.time() - t0, 1)); del Xa; torch.cuda.empty_cache()
        np.savez_compressed(READS / f'{name}_f{f}.npz', rows=va, e=res[f]['e'], lg=res[f]['lg'], per_seed=np.array(res[f]['per_seed']))
        print(json.dumps(dict(arm=name, fold=f, score=E.official(res[f]['e']), per_seed=res[f]['per_seed'], D=res[f]['D'], sec=res[f]['sec'])), flush=True)
    return res


def load_arm(name, folds):
    out = {}
    for f in folds:
        z = np.load(READS / f'{name}_f{f}.npz'); out[f] = dict(rows=z['rows'], e=z['e'], lg=z['lg'], per_seed=z['per_seed'].tolist())
    return out


def compare(name, a, ref, folds, added=True):
    e1 = np.concatenate([a[f]['e'] for f in folds]); e0 = np.concatenate([ref[f]['e'] for f in folds]); g = np.concatenate([gp[a[f]['rows']] for f in folds]); pb = E.paired_bootstrap(e1, e0, g); s1, s0 = E.official(e1), E.official(e0)
    o = dict(arm=name, folds=folds, delta=round(pb['delta'], 4), ci95=[round(v, 4) for v in pb['ci95']], per={k: round(s1[k] - s0[k], 4) for k in K}, by_fold={f: round(E.official(a[f]['e'])['mean'] - E.official(ref[f]['e'])['mean'], 4) for f in folds}, score=round(s1['mean'], 4), ref=round(s0['mean'], 4))
    if added:
        import best_pipeline_20261106 as BP
        assert np.array_equal(BP.idx, bb['idx']); rows = np.concatenate([a[f]['rows'] for f in folds]); d = np.concatenate([a[f]['lg'] - ref[f]['lg'] for f in folds]); o['added'] = {}
        for w in (0.125, 0.25, 0.5):
            c = np.zeros_like(BP.H); c[rows] = w * d; P = BP.pipe(BP.add(BP.BEST, {ff: c for ff in range(5)})); eb, er = E.ape(P[rows], y[rows]), E.ape(BP.REF_BEST[rows], y[rows]); q = E.paired_bootstrap(eb, er, gp[rows]); s1_, s0_ = E.official(eb), E.official(er)
            o['added'][str(w)] = dict(delta=round(q['delta'], 4), ci95=[round(v, 4) for v in q['ci95']], per={k: round(s1_[k] - s0_[k], 4) for k in K})
    print(json.dumps(o), flush=True); return o


def opt(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    sys.path.insert(0, str(X / 'meta_ceiling_20261103/source')); cmd = sys.argv[1]; folds = [int(v) for v in opt('--folds', '0,4').split(',')]; seeds = [int(v) for v in opt('--seeds', '0,1,2').split(',')]; pick = sys.argv[2].split(',') if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None; results = []
    if cmd == 'F1':
        ref_ok = all((READS / f'std_f{f}.npz').exists() for f in folds); ref = load_arm('std', folds) if ref_ok else read_arm('std', ARMS['std'], folds, seeds)
        for arm in pick or [k for k in ARMS if k != 'std']:
            a = read_arm(arm, ARMS[arm], folds, seeds); results.append(compare(arm, a, ref, folds))
            (READS / f'F1_{"_".join(map(str, folds))}.jsonl').open('a').write(json.dumps(results[-1]) + '\n')
    elif cmd == 'F2':
        ref = load_arm('std', folds)
        for kind in pick or ['hlg', 'quantile']:
            a = read_arm(f'head_{kind}', ['std'], folds, seeds, kind); results.append(compare(f'head_{kind}', a, ref, folds))
            (READS / f'F2_{"_".join(map(str, folds))}.jsonl').open('a').write(json.dumps(results[-1]) + '\n')


if __name__ == '__main__':
    main()
