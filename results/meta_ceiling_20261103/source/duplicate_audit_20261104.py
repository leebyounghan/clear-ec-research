"""Exact duplicate frames in the training set (2026-11-04). Pixel hash of every raw frame. Groups: how many, of which ID relation (same case with different punctuation; same donor, other eye; other case), whether labels agree,
how far apart the IDs are, fold spread, whether hybrid B sides with one label. Also near duplicates: pixel correlation of the closest thumbnail pairs that are not exact. Writes groups to retrieval/duplicate_groups.json."""
import hashlib, itertools, json, os, re, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
OUT = '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/retrieval'; RAW = '/workspace/clear-ec-research/datasets/clear_ec_fullframe_20260904/img_full_raw.dat'
b = E.load_base(); y, fold, B, idx = b['y'], b['fold'], b['B'], b['idx']; ids = [re.sub(r'\s+', ' ', i.strip()) for i in b['ids']]; N = len(ids)
def h(i): return hashlib.md5(np.asarray(np.memmap(RAW, dtype=np.uint8, mode='r', shape=(8954, 972, 1296))[i]).tobytes()).hexdigest()
workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1)))
with get_context('fork').Pool(workers) as pool: H = list(pool.imap(h, idx.tolist(), chunksize=32))
gr = {}
for n, v in enumerate(H): gr.setdefault(v, []).append(n)
groups = [g for g in gr.values() if len(g) > 1]; print('frames', N, 'unique', len(gr), 'groups', len(groups), 'rows in groups', sum(map(len, groups)), 'sizes', {k: sum(1 for g in groups if len(g) == k) for k in sorted({len(g) for g in groups})}, flush=True)
def norm(i):
    m = re.match(r'(\d{4})-(\d{4})[ -]*([O0][DS])', i)
    if m: return ('new', m.group(1), m.group(2), m.group(3).replace('0', 'O'))
    m = re.fullmatch(r'(\d{4})-(\d{2})', i)
    if m: return ('old', m.group(2), m.group(1), '')
    return ('other', i, '', '')
def relation(a, c):
    A, C = norm(a), norm(c)
    if A == C: return 'same_case_other_spelling'
    if A[0] != C[0]: return 'cross_format'
    if A[:3] == C[:3]: return 'same_donor_other_eye'
    if A[1] != C[1]: return 'other_year'
    d = sum(x != z for x, z in zip(A[2], C[2])); same = sorted(A[2]) == sorted(C[2])
    return 'digit_transposed' if same else f'digits_differ_{min(d, 3)}' + ('' if A[3] == C[3] else '_other_eye')
rows, rel = [], {}
for g in groups:
    for a, c in itertools.combinations(g, 2):
        ag = bool(np.allclose(y[a], y[c], atol=1e-6)); r = relation(ids[a], ids[c]); rel.setdefault(r, [0, 0]); rel[r][0 if ag else 1] += 1
        ra, rc = np.log(np.maximum(y[a], 1e-6) / B[a]), np.log(np.maximum(y[c], 1e-6) / B[c])
        rows.append(dict(a=ids[a], c=ids[c], rel=r, same_label=ag, ya=y[a].round(3).tolist(), yc=y[c].round(3).tolist(), Ba=B[a].round(3).tolist(), Bc=B[c].round(3).tolist(), ra=ra.round(3).tolist(), rc=rc.round(3).tolist(), fa=int(fold[a]), fc=int(fold[c]), ia=int(a), ic=int(c)))
print('pairs by relation [labels equal, labels differ]', json.dumps(rel), flush=True)
dif = [r for r in rows if not r['same_label']]; print('pairs with differing labels', len(dif), 'in different folds', sum(r['fa'] != r['fc'] for r in dif), 'all pairs in different folds', sum(r['fa'] != r['fc'] for r in rows), flush=True)
if dif:
    d = np.array([np.abs(np.log(np.array(r['ya']) + 1e-6) - np.log(np.array(r['yc']) + 1e-6)) for r in dif]); print('median |log diff| CD CV HEX', np.round(np.median(d, 0), 3).tolist(), 'p90', np.round(np.quantile(d, .9, 0), 3).tolist(), flush=True)
    mn = np.array([np.minimum(np.abs(r['ra'][0]), np.abs(r['rc'][0])) for r in dif]); mx = np.array([np.maximum(np.abs(r['ra'][0]), np.abs(r['rc'][0])) for r in dif]); print('CD |residual| of closer and farther label: median', round(float(np.median(mn)), 3), round(float(np.median(mx)), 3), flush=True)
dup_rows = sorted({n for g in groups for n in g}); d_ = np.zeros(N, bool); d_[dup_rows] = True; e = E.ape(B, y); print('error of B: rows in groups', {k: round(v, 3) for k, v in E.official(e[d_]).items() if k != 'n'}, 'other rows', {k: round(v, 3) for k, v in E.official(e[~d_]).items() if k != 'n'}, 'share of score', round(float(np.nansum(e[d_]) / np.nansum(e)), 4), flush=True)
yr = {}
for n in dup_rows: yr.setdefault(norm(ids[n])[0] + norm(ids[n])[1], []).append(n)
print('rows in groups by source-year', {k: len(v) for k, v in sorted(yr.items())}, flush=True)
json.dump(dict(groups=[[ids[n] for n in g] for g in groups], pairs=rows), open(f'{OUT}/duplicate_groups.json', 'w'), indent=1)
print('examples differ', json.dumps([{k: r[k] for k in ('a', 'c', 'rel', 'ya', 'yc', 'Ba')} for r in dif[:14]]), flush=True); print('done', flush=True)
