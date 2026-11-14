"""Exact duplicate weight files in the deploy census (cell-route lane, 2026-11-11): same size -> partial hash (first and last 32 MB) -> full sha256
for partial-hash collisions. Pool of 8 readers. usage: dup_check.py -> prints duplicate groups, writes docker/dup_check.json"""
import os, json, hashlib, collections
from multiprocessing import Pool
ROOT = '/workspace/clear-ec-research/'; os.chdir(ROOT); H = 'experiments/lane_cellroute_20261109/inference/'
fs = [l.strip() for l in open(H + 'docker/files.txt') if l.strip() and os.path.isfile(l.strip()) and os.stat(l.strip()).st_size > 1 << 20]
def part(f):
    s = os.stat(f).st_size; h = hashlib.sha256()
    with open(f, 'rb') as fh:
        h.update(fh.read(32 << 20)); fh.seek(max(0, s - (32 << 20))); h.update(fh.read(32 << 20))
    return f, h.hexdigest()
def full(f):
    h = hashlib.sha256()
    with open(f, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 24), b''): h.update(b)
    return f, h.hexdigest()
bysize = collections.defaultdict(list)
for f in fs: bysize[os.stat(f).st_size].append(f)
cand = [f for v in bysize.values() if len(v) > 1 for f in v]
with Pool(8) as p: ph = dict(p.map(part, cand))
grp = collections.defaultdict(list)
for f in cand: grp[(os.stat(f).st_size, ph[f])].append(f)
cand2 = [f for v in grp.values() if len(v) > 1 for f in v]
with Pool(8) as p: fh = dict(p.map(full, cand2))
dup = collections.defaultdict(list)
for f in cand2: dup[fh[f]].append(f)
dups = [dict(sha=k, gb=round(os.stat(v[0]).st_size / 1e9, 3), files=v) for k, v in dup.items() if len(v) > 1]
json.dump(dict(n_files=len(fs), same_size_candidates=len(cand), partial_collisions=len(cand2), duplicates=dups), open(H + 'docker/dup_check.json', 'w'), indent=1)
print('files', len(fs), 'same-size', len(cand), 'partial collisions', len(cand2))
for d in dups: print('%.3f GB x%d' % (d['gb'], len(d['files'])), d['files'])
print('saving if deduplicated: %.2f GB' % sum(d['gb'] * (len(d['files']) - 1) for d in dups))
