"""Verify nested base provenance and build image-only features; never fit a calibrator."""
import csv,hashlib,json,sys,datetime
from pathlib import Path
import numpy as np
R=Path('/workspace/clear-ec-research'); O=Path(sys.argv[1]); sys.path.insert(0,str(O))
from nested_oof_calibration_20261023 import image_quality,tests
from nested_oof_contract_20261023 import validate
assert not (O/'inputs_audit.json').exists()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
e=read(O/'split_preflight.json'); assert sha(O/'split_preflight.json')=='f7cf68eca150581d14714836ac845f255e0ee75b4add16bf8fb311aa5ac60ba7'
assert sha(O/'nested_oof_calibration_20261023.py')=='333a80395fd98f2c8d57d6b4583903f29db7193edcd2b0c92866fe4c1af0a1bc'
for item in e['inputs'].values():assert sha(R/item['path'])==item['sha256']
m=np.load(R/e['inputs']['meta']['path'],allow_pickle=True);ids=m['ids'].astype(str);y=m['Y'];n=len(ids)
rows={Path(z['ID']).stem:z for z in csv.DictReader((R/e['inputs']['csv']['path']).open())}
np.testing.assert_array_equal(y,np.asarray([[float(rows[Path(i).stem][k]) for k in ['CD','CV','HEX']] for i in ids],dtype=y.dtype))
slides=[rows[Path(i).stem]['slide_id'].strip() for i in ids];parent={s:s for s in slides}
f=np.load(R/e['inputs']['pixels']['path'],allow_pickle=True);hashes=f['sha256'].astype(str)
def find(s):
 while parent[s]!=s:parent[s]=parent[parent[s]];s=parent[s]
 return s
seen={}
for s,h in zip(slides,hashes):
 if h in seen:parent[find(s)]=find(seen[h])
 else:seen[h]=s
g=[find(s) for s in slides];contract=validate(e,g,ids.tolist());tr=e['outer_train'];va=e['outer_validation']
proof=[];parts=[]
for inner in [0,1,2,None]:
 root=R/('experiments/phase2_pooling_20261020/arms/mean' if inner is None else f'experiments/nested_oof_20261023/inner{inner}_regression_v1')
 mm=read(root/'manifest.json'); run=root/'runs/fold4/cpdino_vitb'; a=read(O/('outer_audit.json' if inner is None else f'inner{inner}_audit.json'))
 assert sha(root/'manifest.json')==a['manifest_sha'];assert sha(run/'last.pt')==a['checkpoint_sha'];assert sha(run/'predictions.npz')==a['predictions_sha']
 tt=tr if inner is None else e['inner'][inner]['train'];vv=va if inner is None else e['inner'][inner]['validation']
 assert mm['train_indices']==tt and mm['validation_indices']==vv
 assert mm['recipe']['epochs']==20 and mm['recipe']['pooling']=='mean' and mm['seed']==46
 logs=np.log(y[tt].astype(np.float64));np.testing.assert_array_equal(logs.mean(0),mm['log_mean']);np.testing.assert_array_equal(logs.std(0),mm['log_std'])
 ss=R/mm['ssl_root'];sm=read(ss/'manifest.json');assert sha(ss/'manifest.json')==mm['ssl_manifest_sha'];assert sha(ss/'run/last.pt')==mm['ssl_checkpoint_sha']
 assert sm['split_sha256']==mm['split_sha256'] and sm['train_ids_sha']==mm['train_ids_sha'] and sm['validation_images_used']==0
 assert sm['recipe']['epochs']==5 and sm['seed']==1003
 sp=np.load(R/mm['split_path']);np.testing.assert_array_equal(sp['train'],tt);np.testing.assert_array_equal(sp['validation'],vv)
 assert sha(R/mm['split_path'])==mm['split_sha256']
 z=np.load(run/'predictions.npz');np.testing.assert_array_equal(z['indices'],vv);np.testing.assert_array_equal(z['ids'].astype(str),ids[vv]);np.testing.assert_array_equal(z['target'],y[vv]);assert np.isfinite(z['pred']).all() and (z['pred']>0).all()
 item=dict(inner=inner,manifest_sha=sha(root/'manifest.json'),prediction_sha=sha(run/'predictions.npz'),ssl_manifest_sha=sha(ss/'manifest.json'),ssl_checkpoint_sha=mm['ssl_checkpoint_sha'],train_n=len(tt),validation_n=len(vv));proof.append(item)
 if inner is None:outer=z['pred'].copy()
 else:parts.append((np.asarray(vv),z['pred'].copy()))
ix=np.concatenate([x[0] for x in parts]);p=np.concatenate([x[1] for x in parts]);order=np.argsort(ix);ix=ix[order];p=p[order];np.testing.assert_array_equal(ix,tr)
raw=np.memmap(R/e['inputs']['raw']['path'],mode='r',dtype=np.uint8,shape=(n,972,1296));q=np.zeros((n,3))
for i in range(n):
 assert hashlib.sha256(raw[i].tobytes()).hexdigest()==hashes[i]
 q[i]=image_quality(raw[i])
 if i%1000==0:print('quality',i,flush=True)
with (O/'inputs.npz').open('xb') as f:np.savez_compressed(f,fit_indices=ix,fit_ids=ids[ix],fit_pred=p,fit_target=y[ix],fit_quality=q[ix],outer_indices=va,outer_ids=ids[va],outer_pred=outer,outer_target=y[va],outer_quality=q[va])
out=dict(status='PASS',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),contract=contract,base_provenance=proof,inputs_sha256=sha(O/'inputs.npz'),source_sha256=sha(__file__),calibration_cpu_tests=tests(),all_current_pixel_hashes=True,train_only_normalization=True,outer_base_reused_terminal20=True,calibration_fit_started=False,limitations=e['limitations'])
with (O/'inputs_audit.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out),flush=True)
