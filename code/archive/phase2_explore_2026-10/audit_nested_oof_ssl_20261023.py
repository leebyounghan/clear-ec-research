"""Independent CPU-only completion audit for duration-controlled SSL runs."""
import csv
import nested_oof_contract_20261023 as contract
import argparse,hashlib,json,math,datetime
from pathlib import Path
import numpy as np
import torch

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def idsha(ids):return hashlib.sha256('\n'.join(ids).encode()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(1)
    root=a.root;run=root/'run';m=read(root/'manifest.json');r=read(run/'result.json');c=read(run/'complete.json');epochs=read(run/'epochs.json');ms=sha(root/'manifest.json')
    assert c['status']=='completed' and c['result_sha']==sha(run/'result.json')
    assert c['manifest_sha']==r['manifest_sha']==ms
    for path,h in m['hashes'].items():assert sha(path)==h,path
    assert sha(m['data_path'])==m['raw_sha256']
    for name in ['cpu_tests.json','smoke.json']:
        report=read(root/name);assert report['status']=='passed' and report['manifest_sha']==ms
    with np.load(m['metadata_path'],allow_pickle=True) as z:ids=z['ids'].astype(str)
    with np.load(m['split_path']) as z:tr=z['train'];va=z['validation']
    assert not np.intersect1d(tr,va).size
    assert m.get('schema')=='nested_outer4_inner_ssl_v1' and m['outer_fold']==4
    assert m['outer_validation_images_used']==0
    evidence=read(Path(m['data_preflight_path']))
    for name in ['meta','csv','pixels','outer']:
        entry=evidence['inputs'][name];assert sha(entry['path'])==entry['sha256']
    assert evidence['inputs']['raw']['sha256']==m['raw_sha256']
    with open(evidence['inputs']['csv']['path'],newline='') as f:
        rows={Path(x['ID']).stem:x for x in csv.DictReader(f)}
    slides=[rows[Path(x).stem]['slide_id'].strip() for x in ids]
    with np.load(evidence['inputs']['pixels']['path'],allow_pickle=True) as z:hashes=z['sha256'].astype(str)
    parent=list(range(len(ids)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for keys in [slides,hashes]:
        seen={}
        for i,key in enumerate(keys):
            if key in seen:parent[find(i)]=find(seen[key])
            else:seen[key]=i
    contract.validate(evidence,[find(i) for i in range(len(ids))],ids.tolist())
    inner=next(x for x in evidence['inner'] if x['inner']==m['inner_fold'])
    np.testing.assert_array_equal(tr,inner['train']);np.testing.assert_array_equal(va,inner['validation'])
    with np.load(evidence['inputs']['outer']['path']) as z:
        np.testing.assert_array_equal(z['train'],evidence['outer_train'])
        np.testing.assert_array_equal(z['validation'],evidence['outer_validation'])
    assert idsha(ids[tr])==m['train_ids_sha']==r['train_ids_sha']
    assert idsha(ids[va])==m['validation_ids_sha']==r['validation_ids_sha']
    n=m['recipe']['epochs'];steps=math.ceil(len(tr)/m['recipe']['images_per_batch'])
    assert n==r['epochs']==len(epochs)==len(r['epochs_detail'])
    assert epochs==r['epochs_detail'] and r['steps']==r['total_steps']==m['total_steps']==steps*n
    assert r['train_image_presentations']==n*len(tr) and r['validation_images_used']==0 and r['target_arrays_loaded'] is False
    assert m['validation_images_used']==0 and m['target_arrays_loaded'] is False
    artifacts=[]
    for ep,item in enumerate(epochs,1):
        path=run/f'epoch{ep:02d}.pt';h=sha(path);assert h==item['checkpoint_sha']
        assert item['epoch']==ep and item['train_images_seen']==item['unique_train_images']==len(tr) and item['validation_images_used']==0
        order=np.random.default_rng(m['seed']+ep-1).permutation(tr)
        assert item['order_sha']==hashlib.sha256(order.tobytes()).hexdigest() and item['train_ids_sha']==idsha(ids[order])
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['manifest_sha']==ms and state['epoch']==ep and state['step']==ep*steps and state['seen_this_epoch']==len(tr)
        np.testing.assert_array_equal(state['order'].numpy(),order)
        assert state['encoder'] and state['projector'] and state['optimizer']['state']
        for component in ['encoder','projector']:
            assert all(torch.isfinite(t).all().item() for t in state[component].values() if t.is_floating_point())
        del state
        artifacts.append({'epoch':ep,'checkpoint_sha256':h})
    assert sha(run/'last.pt')==sha(run/f'epoch{n:02d}.pt')==r['checkpoint_sha']==c['checkpoint_sha']
    report=dict(status='independent_nested_ssl_completion_verified',outer_fold=4,inner_fold=m['inner_fold'],outer_validation_images_used=0,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_sha256=ms,result_sha256=sha(run/'result.json'),complete_sha256=sha(run/'complete.json'),checkpoint_sha256=r['checkpoint_sha'],epochs=n,total_steps=steps*n,train_image_presentations=n*len(tr),validation_images_used=0,target_arrays_loaded=False,epoch_artifacts=artifacts,accuracy_claim=False)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report))
if __name__=='__main__':main()
