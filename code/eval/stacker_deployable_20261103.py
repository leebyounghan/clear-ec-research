"""Deployable residual stacker check (2026-11-03). The Grand Challenge container sees one image per case, so features
derived from the case ID (year, eye, ID format) are unavailable at test time. Feature sets here use only what the
submission-2 container can compute: recipe A and Phase-1 predictions, four label-free image statistics, the hex6
detector outputs (n_det, f_active10) and hex6-corrected HEX. Nested LOFO shrink as in the follow-up script.
Also: the affine-in-log stretch alone and before the stacker, and a version without Phase-1 predictions as inputs."""
import json, os, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import ceiling_meta_followup_20261103 as Fu
E = C.E; K = C.K; r4 = C.r4

def run(Xf, b, base, workers, S=(0.25, 0.5, 0.75, 1.0)):
    y, fold = b['y'], b['fold']
    t = np.log(np.where(y > 0, y, np.nan)) - np.log(base)
    tasks, keys = [], []
    for f in range(5):
        for j in range(3):
            tasks.append((Xf, t, fold != f, fold == f, j)); keys.append(('o', f, None, j))
            for h in range(5):
                if h != f:
                    tasks.append((Xf, t, (fold != f) & (fold != h), fold == h, j)); keys.append(('i', f, h, j))
    with get_context('fork').Pool(workers) as pool:
        preds = pool.map(Fu.fit_predict, tasks, chunksize=1)
    outer = np.full_like(base, np.nan); inner = {}
    for kk, pr in zip(keys, preds):
        if kk[0] == 'o': outer[fold == kk[1], kk[3]] = pr
        else: inner.setdefault((kk[1], kk[3]), np.full(len(base), np.nan))[fold == kk[2]] = pr
    p = base.copy(); ch = {}
    for f in range(5):
        for j in range(3):
            tr = (fold != f) & (y[:, j] > 0); th = inner[(f, j)]
            cand = {s: np.mean(np.abs(base[tr, j] * np.exp(s * th[tr]) - y[tr, j]) / y[tr, j]) for s in (0.0,) + S}
            s = min(cand, key=cand.get); ch[f'{f}_{K[j]}'] = s
            p[fold == f, j] = base[fold == f, j] * np.exp(s * outer[fold == f, j])
    return p, ch

def stretch(b, base):
    y, fold = b['y'], b['fold']; newp = np.full_like(base, np.nan)
    for f in range(5):
        for j in range(3):
            ok = (fold != f) & (y[:, j] > 0); lb = np.log(base[ok, j]); mu = lb.mean(); best = None
            for g in np.round(np.arange(0.85, 1.36, 0.025), 3):
                q = np.exp(g * (lb - mu) + mu); c = C.wmed(y[ok, j] / q, q / y[ok, j]); m = np.mean(np.abs(c * q - y[ok, j]) / y[ok, j])
                if best is None or m < best[0]: best = (m, g, c)
            _, g, c = best; te = fold == f
            newp[te, j] = c * np.exp(g * (np.log(base[te, j]) - mu) + mu)
    return newp

def main():
    b = E.load_base(); st = E.image_stats(b); y, B, fold = b['y'], b['B'], b['fold']
    gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True)
    workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1)))
    def g(name):
        v = gf[name].astype(float).copy()
        if name + '_valid' in gf.files: v[~gf[name + '_valid']] = np.nan
        v[gf['seen_leaked']] = np.nan; return v
    la, lp = np.log(b['A']), np.log(b['P1'])
    img = np.c_[st['mean'], st['contrast'], st['black_frac'], st['focus']]
    det = np.c_[g('n_det'), g('f_active10'), g('hex_B_hex6_lofo')]
    hexB = B.copy(); v = gf['hex_B_hex6_lofo_valid'] & ~gf['seen_leaked']; hexB[v, 2] = gf['hex_B_hex6_lofo'][v]
    sets = {'deploy_A_P1_img_det': np.c_[la, lp, la - lp, img, det], 'deploy_A_img_det_noP1': np.c_[la, img, det],
            'deploy_A_P1_img': np.c_[la, lp, la - lp, img]}
    grp = E.donors(b['ids']); out = {}
    def rep(name, p, ref=B):
        r = dict(score={k: r4(v) for k, v in C.mape(p, y).items() if k != 'n'}, paired_vs_B=E.paired_bootstrap(E.ape(p, y), E.ape(ref, y), grp),
                 per_fold={f: r4(C.mape(p[fold == f], y[fold == f])['mean'] - C.mape(ref[fold == f], y[fold == f])['mean']) for f in range(5)})
        out[name] = r; print(name, r['score'], round(r['paired_vs_B']['delta'], 4), r['paired_vs_B']['ci95'], r['per_fold'], flush=True)
    rep('hex6_only', hexB)
    rep('stretch_B', stretch(b, B))
    rep('stretch_hex6', stretch(b, hexB))
    for name, Xf in sets.items():
        p, ch = run(Xf, b, B, workers); rep(name, p); out[name]['shrink'] = ch
    p, ch = run(sets['deploy_A_P1_img_det'], b, stretch(b, hexB), workers); rep('stretch_hex6_then_stacker', p); out['stretch_hex6_then_stacker']['shrink'] = ch
    (C.OUT / 'stacker_deployable.json').write_text(json.dumps(out, indent=1, default=float))

if __name__ == '__main__':
    main()
