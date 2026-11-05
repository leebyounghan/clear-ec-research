"""Read-only terminal point-probe reproduction; no training or threshold fitting."""
import pathlib,json,hashlib,sys,fcntl,subprocess,datetime
import numpy as np
import torch
root=pathlib.Path('experiments/green_point_probe_20261023')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
m=json.loads((root/'manifest.json').read_text())
assert sha(root/'manifest.json')=='f8933d54a41c0829dca8cfb214c226127eeaf8cd9ba6279c4c12c502de8ce5a1'
for p,h in m['hashes'].items():assert sha(p)==h,p
lock=pathlib.Path('experiments/phase2_backbone_ssl_20261020/gpu0.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
sys.path.insert(0,str(root/'source'))
import phase2_pooling_inference_20261020 as inf
import green_point_spatial_20261023 as spatial
import green_point_probe_head_20261023 as probe
torch.set_num_threads(4)
result=json.loads((root/'result.json').read_text())
assert sha(root/'head_terminal200.pt')==result['checkpoint_sha256']
assert sha(m['ssl_checkpoint'])==m['ssl_checkpoint_sha256']
encoder=inf.build_model('mean')[0].original
ssl=torch.load(m['ssl_checkpoint'],map_location='cpu',weights_only=True)
encoder.load_state_dict(ssl['encoder'],strict=True);del ssl
head=torch.nn.Linear(768,1)
ck=torch.load(root/'head_terminal200.pt',map_location='cpu',weights_only=True)
assert ck['epoch']==200 and ck['manifest_sha256']==sha(root/'manifest.json')
head.load_state_dict(ck['head'],strict=True)
encoder.eval().requires_grad_(False).cuda();head.eval().requires_grad_(False).cuda()
ann=json.loads(pathlib.Path('datasets/clear_ec_official/green_eval/annotations.json').read_text())
with np.load('datasets/clear_ec_fullframe_20260904/fullframe_meta.npz',allow_pickle=True) as z:ids=z['ids'].astype(str)
raw=np.memmap('datasets/clear_ec_fullframe_20260904/img_full_raw.dat',mode='r',dtype=np.uint8,shape=(len(ids),972,1296))
rows=[]
for saved in result['rows']:
 name=saved['id'];i=int(np.flatnonzero(ids==name)[0])
 batch=inf.prepare_batch(inf.preprocess_native(np.array(raw[i])).unsqueeze(0),'cuda')
 with torch.inference_mode():
  with torch.autocast('cuda',dtype=torch.bfloat16):
   tokens=encoder.forward_features(batch)['x_norm_patchtokens']
  logits=torch.nn.functional.linear(tokens.float(),head.weight,head.bias)
  prob=logits.sigmoid().reshape(124,164).cpu().numpy()
 _,mask=spatial.point_target(ann[name]['markers_mha'],ann[name]['roi_box_mha'])
 points=probe.peaks(prob,mask)
 metrics=spatial.match_points(points,ann[name]['markers_mha'])
 for k,v in metrics.items():
  ref=saved['probe'][k]
  if v is None:assert ref is None
  else:assert abs(v-ref)<1e-6,(name,k,v,ref)
 rows.append(dict(id=name,metrics=metrics,detected_points=len(points)))
out=dict(status='passed',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_sha256=sha(root/'manifest.json'),checkpoint_sha256=sha(root/'head_terminal200.pt'),rows=rows,scope='All five heldout point-localization cases reloaded; fixed thresholds and matching; no training',forward='independent linear functional head on strict-reloaded encoder tokens',tolerance=1e-6,limits=result['limits'])
print(json.dumps(out))
