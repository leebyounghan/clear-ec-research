"""Prospective fold4 outer / three inner connected-group splits. No training."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
R=Path('/workspace/clear-ec-research')
def sha(p):
 h=hashlib.sha256()
 with (R/p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
paths={'meta':'datasets/clear_ec_fullframe_20260904/fullframe_meta.npz','csv':'datasets/clear_ec_official/final_train_ids.csv','pixels':'experiments/ssl_screen_20260911/preflight/pixel_fingerprints.npz','outer':'experiments/target_fold4_20260912/preflight/fold4.npz','raw':'datasets/clear_ec_fullframe_20260904/img_full_raw.dat'}
m=np.load(R/paths['meta'],allow_pickle=True); ids=m['ids'].astype(str); y=m['Y']; n=len(ids)
rows={Path(r['ID']).stem:r for r in csv.DictReader((R/paths['csv']).open())}
slides=[rows[Path(i).stem]['slide_id'].strip() for i in ids]
assert all(slides) and len(set(ids))==n
assert np.array_equal(y,np.asarray([[float(rows[Path(i).stem][k]) for k in ['CD','CV','HEX']] for i in ids],dtype=y.dtype))
f=np.load(R/paths['pixels'],allow_pickle=True); hashes=f['sha256'].astype(str)
raw=np.memmap(R/paths['raw'],mode='r',dtype=np.uint8,shape=(n,972,1296))
assert all(hashlib.sha256(raw[i].tobytes()).hexdigest()==hashes[i] for i in range(n))
parent={s:s for s in slides}
def find(s):
 while parent[s]!=s:
  parent[s]=parent[parent[s]];s=parent[s]
 return s
seen={}
for s,h in zip(slides,hashes):
 if h in seen: parent[find(s)]=find(seen[h])
 else: seen[h]=s
g=np.array([find(s) for s in slides]); outer=np.load(R/paths['outer']); tr=outer['train']; va=outer['validation']
assert not set(g[tr])&set(g[va])
assert (y[np.r_[tr,va]]>0).all() and np.isfinite(y[np.r_[tr,va]]).all()
# Target-blind largest-first balancing; seeded hash tie-break is fixed prospectively.
groups=sorted(set(g[tr]),key=lambda s:(-int((g[tr]==s).sum()),hashlib.sha256(('46:'+s).encode()).hexdigest()))
bins=[[] for _ in range(3)]; sizes=[0]*3
for s in groups:
 j=min(range(3),key=lambda j:(sizes[j],j));bins[j].append(s);sizes[j]+=int((g[tr]==s).sum())
inner=[]
for j,b in enumerate(bins):
 iv=tr[np.isin(g[tr],b)];it=tr[~np.isin(g[tr],b)]
 assert not set(g[it])&set(g[iv]) and not set(g[np.r_[it,iv]])&set(g[va])
 assert np.array_equal(np.sort(np.r_[it,iv]),np.sort(tr))
 inner.append(dict(inner=j,train=it.tolist(),validation=iv.tolist(),train_n=len(it),validation_n=len(iv),connected_overlap=0,outer_overlap=0))
assert np.array_equal(np.sort(np.concatenate([np.array(x['validation']) for x in inner])),np.sort(tr))
print(json.dumps(dict(status='PASS',training_started=False,outer_fold=4,inner_folds=3,seed=46,inputs={k:dict(path=p,sha256=sha(p)) for k,p in paths.items()},outer_train=tr.tolist(),outer_validation=va.tolist(),inner=inner,checks=dict(all_current_pixel_hashes=True,positive_targets=True,source_targets_exact=True,inner_oof_coverage_once=True,outer_connected_group_exclusion=True),limitations=['Previously reused development outer fold; not untouched evaluation','Patient identity and near-duplicate independence unresolved']),allow_nan=False))
