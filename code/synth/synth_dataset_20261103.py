"""Synthetic endothelium training set for route B (2026-11-03): no human labels.

Each image copies the CD/CV/HEX of a random real training row: CD sets the cell spacing, (CV, HEX) pick the
renderer geometry from the calibration grid (nearest of the grid means, one of the best three at random), and the
appearance is randomised (framed style with the real 48 % rate). Saved per image: PNG (native 972x1296, uint8) and
an npz with every cell centre (x, y), the uint16 instance map, the ROI box, the exact ROI labels, the target row and
the parameters. Forked pool sized clamp(16 - load1 - 1, 2, 12), one thread per worker, chunked imap.
usage: synth_dataset_20261103.py OUT N [SEED0]
"""
import json, os, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synth_endothelium_20261103 as S

CAL = json.load(open(os.path.join(HERE, '..', 'synth_calibration.json')))
G = [(tuple(float(x) for x in k.split('_')), v) for k, v in CAL['grid'].items()]
Y = None


def choose(rng, cv, hx):
    d = np.array([((v['cv'] - cv) / 0.04) ** 2 + ((v['hex'] - hx) / 0.03) ** 2 for _, v in G])
    k = rng.choice(np.argsort(d)[:3])
    (relax, size_sd, jitter), v = G[k]
    return int(relax), size_sd, jitter, float(np.sqrt(d[k]))


def make(task):
    out, i, seed = task
    if os.path.exists(f'{out}/meta/{i:05d}.npz'):
        return None
    rng = np.random.default_rng(seed)
    r = int(rng.integers(len(Y)))
    cd, cv, hx = Y[r]
    relax, size_sd, jitter, dist = choose(rng, cv, hx)
    img, lab, ctr, y, prm = S.sample(seed, cd=float(cd), jitter=jitter, size_sd=size_sd, relax=relax)
    from PIL import Image
    Image.fromarray(img).save(f'{out}/images/{i:05d}.png')
    np.savez_compressed(f'{out}/meta/{i:05d}.npz', centres=ctr.astype(np.float32), instance=lab.astype(np.uint16),
                        roi=np.array(y['roi']), labels=np.array([y['CD'], y['CV'], y['HEX']]), target=np.array([cd, cv, hx]),
                        target_row=r, params=json.dumps({**prm, 'grid_dist': dist}))
    return dict(i=i, target=[float(cd), float(cv), float(hx)], got=[y['CD'], y['CV'], y['HEX']], n_cells=int(len(ctr)), framed=prm['framed'])


def main():
    global Y
    out, n = sys.argv[1], int(sys.argv[2]); seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 20261103
    os.makedirs(f'{out}/images', exist_ok=True); os.makedirs(f'{out}/meta', exist_ok=True)
    sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
    import eval_oof_20261031 as E
    y = E.load_base()['y']; Y = y[(y > 0).all(1)]
    workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1)))
    tasks = [(out, i, seed0 + i) for i in range(n)]
    rows = []
    with get_context('fork').Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(make, tasks, chunksize=4)):
            if r: rows.append(r)
            if k % 200 == 0:
                print(json.dumps(dict(done=k, workers=workers)), flush=True)
    with open(f'{out}/index.jsonl', 'a') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    t = np.array([r['target'] for r in rows]); g = np.array([r['got'] for r in rows])
    if len(rows):
        print(json.dumps(dict(n=len(rows), workers=workers, target_median=np.median(t, 0).tolist(), got_median=np.median(g, 0).tolist(),
                              abs_rel_err_median=np.median(np.abs(g - t) / t, 0).tolist())), flush=True)


if __name__ == '__main__':
    main()
