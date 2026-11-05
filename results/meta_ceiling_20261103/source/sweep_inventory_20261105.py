"""Inventory of every stored prediction file for the model-combination sweep (2026-11-05). For each file: which rows, which prediction arrays, on what scale, score against hybrid B on the same rows, whether the rows match one validation fold.
Files group into families by path with the fold number masked. A family covering all five folds, each from its own fold's validation rows, can enter the sweep directly. Forked pool, clamp(16 - load1 - 1, 2, 12)."""
import json, os, re, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
ROOT = '/workspace/clear-ec-research/experiments'; OUT = f'{ROOT}/meta_ceiling_20261103/sweep'; os.makedirs(OUT, exist_ok=True)
b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']; pos = {int(i): n for n, i in enumerate(idx)}; K = list(E.K)
SKIP = re.compile(r'smoke|batchcheck|_discarded|/synth|keepalive|debug|/tmp')
PRED = ('pred_tta4', 'pred', 'pred_single', 'preds', 'prediction', 'predictions', 'oof', 'pred_scaled', 'y_pred', 'p')


def mask(path):
    p = re.sub(r'fold\d', 'fold{f}', path); p = re.sub(r'(?<=[/_])f\d(?=[_/.])', 'f{f}', p); p = re.sub(r'_fold_\d', '_fold_{f}', p); return p


def scan(path):
    try:
        with np.load(f'{ROOT}/{path}', allow_pickle=True) as z:
            keys = z.files; ik = next((k for k in ('indices', 'idx', 'rows', 'index') if k in keys), None)
            if ik is None or z[ik].ndim != 1:
                return dict(path=path, skip='no indices', keys=keys[:12])
            ind = z[ik].astype(int); rows = np.array([pos.get(int(i), -1) for i in ind]); out = []
            if (rows < 0).mean() > 0.2 or len(rows) < 100:
                return dict(path=path, skip='rows not in base', n=len(rows))
            for k in keys:
                a = z[k]
                if a.ndim == 2 and a.shape == (len(rows), 3) and np.issubdtype(a.dtype, np.floating) and k not in ('target', 'y', 'Y', 'targets', 'labels', 'gt', 'y_true'):
                    ok = rows >= 0; r, p = rows[ok], a[ok].astype(np.float64); s = (y[r] > 0) & (p > 0)
                    ratio = [float(np.median(p[s[:, j], j] / y[r][s[:, j], j])) for j in range(3)]
                    if not np.isfinite(ratio).all() or min(ratio) <= 0: continue
                    p2 = p / np.array(ratio) if max(abs(np.log(v)) for v in ratio) > 0.5 else p
                    sc = E.official(E.ape(p2, y[r])); s0 = E.official(E.ape(B[r], y[r]))
                    folds, counts = np.unique(fold[r], return_counts=True)
                    out.append(dict(key=k, n=int(ok.sum()), ratio=[round(v, 3) for v in ratio], score=round(sc['mean'], 3), B_same_rows=round(s0['mean'], 3), per={m: round(sc[m], 3) for m in K}, folds=folds.tolist(), fold_counts=counts.tolist(), rescaled=bool(max(abs(np.log(v)) for v in ratio) > 0.5)))
            return dict(path=path, family=mask(path), arrays=out, keys=keys[:14])
    except Exception as e:
        return dict(path=path, skip=f'{type(e).__name__}: {e}'[:120])


def main():
    paths = [l.strip()[2:] for l in open('/tmp/pred_files.txt') if not SKIP.search(l)]; workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1)))
    with get_context('fork').Pool(workers) as pool: res = pool.map(scan, paths, chunksize=4)
    json.dump(res, open(f'{OUT}/inventory_files.json', 'w'), indent=0)
    fam = {}
    for r in res:
        for a in r.get('arrays', []):
            fam.setdefault((r['family'], a['key']), []).append((r['path'], a))
    tab = []
    for (f, k), v in fam.items():
        cov = sorted({x for _, a in v for x in a['folds'] if a['fold_counts'][a['folds'].index(x)] > 100}); n = sum(a['n'] for _, a in v); w = np.array([a['n'] for _, a in v], float)
        tab.append(dict(family=f, key=k, files=len(v), n=n, folds=cov, score=round(float(np.average([a['score'] for _, a in v], weights=w)), 3), B=round(float(np.average([a['B_same_rows'] for _, a in v], weights=w)), 3), ratio=v[0][1]['ratio'], rescaled=v[0][1]['rescaled']))
    tab.sort(key=lambda t: (-len(t['folds']), t['score'] - t['B']))
    json.dump(tab, open(f'{OUT}/inventory_families.json', 'w'), indent=1)
    print('files', len(res), 'usable', sum(1 for r in res if r.get('arrays')), 'skipped', {s: sum(1 for r in res if str(r.get('skip', '')).startswith(s)) for s in ('no indices', 'rows not in base', 'Bad', 'Value')})
    five = [t for t in tab if len(t['folds']) == 5]; print('families with all five folds', len(five), 'families total', len(tab))
    for t in five[:70]: print(round(t['score'] - t['B'], 3), t['score'], t['n'], t['key'], t['files'], t['ratio'], t['family'][:100])
    print('--- partial, best relative to B'); part = sorted([t for t in tab if len(t['folds']) < 5 and t['n'] > 500], key=lambda t: t['score'] - t['B'])[:45]
    for t in part: print(round(t['score'] - t['B'], 3), t['score'], t['n'], t['folds'], t['key'], t['ratio'], t['family'][:95])


if __name__ == '__main__':
    main()
