"""Map renderer geometry parameters (relax, size_sd, jitter) to the CV/HEX they produce, on 500x500 canvases at the
median spacing; complete cells away from the border. Grid x 3 repeats, forked pool sized to the free CPU quota.
Writes synth_calibration.json: per grid point mean/sd of CV and HEX, plus the real label quantiles to match."""
import json, os, sys, itertools
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_endothelium_20261103 as S
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
C = 500

def one(args):
    relax, size_sd, jitter, rep = args
    rng = np.random.default_rng(hash((relax, size_sd, jitter, rep)) % 2**32)
    a = S.spacing_px(2800); pad = 60
    n = int((C + 2 * pad) ** 2 / (np.sqrt(3) / 2 * a ** 2))
    p = rng.uniform(-pad, C + pad, (n, 2)); w = rng.lognormal(0, size_sd, n); w = (w / w.mean() - 1) * a ** 2 * 0.5
    for _ in range(relax):
        lab = S.power_labels(p, w, step=2, h=C, wd=C); c, cnt = S.lloyd(lab, n, step=2); p = np.where((cnt > 0)[:, None], c, p)
    p = p + rng.normal(0, jitter * a, p.shape)
    lab = S.power_labels(p, w, h=C, wd=C)
    area = np.bincount(lab.ravel(), minlength=n)
    edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    ok = area > 0; ok[edge] = False
    nb = S.neighbours(lab); ids = np.where(ok)[0]
    return relax, size_sd, jitter, float(area[ids].std() / area[ids].mean()), float(np.mean([len(nb.get(i, ())) == 6 for i in ids]))

if __name__ == '__main__':
    import eval_oof_20261031 as E
    y = E.load_base()['y']
    grid = list(itertools.product([0, 1, 2, 3, 4, 6, 8, 12, 16, 24], [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4], [0.0, 0.05, 0.1], range(3)))
    workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1)))
    with get_context('fork').Pool(workers) as pool:
        res = list(pool.imap_unordered(one, grid, chunksize=4))
    tab = {}
    for r, s_, j, cv, hx in res:
        tab.setdefault(f'{r}_{s_}_{j}', []).append((cv, hx))
    out = dict(workers=workers, grid={k: dict(cv=float(np.mean([v[0] for v in vs])), hex=float(np.mean([v[1] for v in vs])),
                                               cv_sd=float(np.std([v[0] for v in vs])), hex_sd=float(np.std([v[1] for v in vs]))) for k, vs in tab.items()},
               real_quantiles={k: [float(np.quantile(y[y[:, j] > 0, j], q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)] for j, k in enumerate(('CD', 'CV', 'HEX'))})
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'synth_calibration.json'), 'w'), indent=1)
    g = out['grid']; print(json.dumps(out['real_quantiles']))
    for k in sorted(g, key=lambda k: (g[k]['hex'])):
        if k.endswith('_0.0'): print(k, round(g[k]['cv'], 3), round(g[k]['hex'], 3))
