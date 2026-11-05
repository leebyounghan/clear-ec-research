"""Frozen SSL point probe; explicit smoke gate, fixed terminal200, ROI diagnostic only."""
import argparse, json, hashlib, pathlib, datetime, os
import numpy as np
import torch
import green_point_spatial_20261023 as spatial
import green_point_probe_head_20261023 as probe
import phase2_pooling_inference_20261020 as inf

def sha(p):return inf.sha256_file(p)
def read(p):return json.loads(pathlib.Path(p).read_text())
def save(p,d):
    with pathlib.Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['smoke','train'])
    ap.add_argument('--root',type=pathlib.Path,required=True);args=ap.parse_args();root=args.root
    m=read(root/'manifest.json');ms=sha(root/'manifest.json')
    for p,h in m['hashes'].items():assert sha(p)==h,p
    assert m['epochs']==200 and m['seed']==46 and m['deadline'] is None
    assert torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.manual_seed(46)
    ann=read('datasets/clear_ec_official/green_eval/annotations.json')
    meta=np.load('datasets/clear_ec_fullframe_20260904/fullframe_meta.npz',allow_pickle=True)
    ids=meta['ids'].astype(str)
    raw=np.memmap('datasets/clear_ec_fullframe_20260904/img_full_raw.dat',mode='r',dtype=np.uint8,shape=(len(ids),972,1296))
    train=m['train_ids'];val=m['validation_ids']
    assert len(train)==20 and len(val)==5 and not set(train)&set(val)
    enc=inf.build_model('mean')[0].original
    ck=torch.load(m['ssl_checkpoint'],map_location='cpu',weights_only=True)
    assert sha(m['ssl_checkpoint'])==m['ssl_checkpoint_sha256']
    enc.load_state_dict(ck['encoder'],strict=True);del ck
    enc.eval().requires_grad_(False).cuda()
    assert not any(p.requires_grad for p in enc.parameters())
    def features(name):
        index=int(np.flatnonzero(ids==name)[0])
        im=np.array(raw[index],copy=True)
        x=inf.preprocess_native(im).unsqueeze(0)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            f=enc.forward_features(inf.prepare_batch(x,'cuda'))['x_norm_patchtokens']
        assert f.shape==(1,124*164,768) and torch.isfinite(f).all()
        return f[0].reshape(124,164,768).float().cpu()
    if args.mode=='smoke':
        assert not (root/'smoke.json').exists()
        before={k:v.detach().cpu().clone() for k,v in enc.state_dict().items()}
        f=features(train[0]);t,s=spatial.point_target(ann[train[0]]['markers_mha'],ann[train[0]]['roi_box_mha'])
        h=probe.build_head().cuda();opt=torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.0001)
        initial=h.weight.detach().clone()
        loss=probe.loss(h,f.cuda(),torch.from_numpy(t).cuda(),torch.from_numpy(s).cuda())
        loss.backward();assert all(torch.isfinite(p.grad).all() for p in h.parameters());opt.step()
        assert not torch.equal(initial,h.weight)
        assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in enc.state_dict().items())
        assert all(p.grad is None for p in enc.parameters())
        save(root/'smoke.json',dict(status='passed',manifest_sha256=ms,checked_utc=now(),feature_shape=list(f.shape),loss=float(loss),encoder_unchanged=True,head_updated=True,validation_used=False))
        return
    assert read(root/'smoke.json')['manifest_sha256']==ms and read(root/'smoke.json')['status']=='passed'
    assert not (root/'started.json').exists()
    save(root/'started.json',dict(manifest_sha256=ms,started_utc=now()))
    # Extract training features only; validation features are computed after head freezing.
    cache={}
    for name in train:
        f=features(name);t,s=spatial.point_target(ann[name]['markers_mha'],ann[name]['roi_box_mha'])
        cache[name]=(f,torch.from_numpy(t),torch.from_numpy(s))
    h=probe.build_head().cuda();opt=torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.0001)
    prior=float(np.mean([float(t[s].mean()) for _,t,s in cache.values()]))
    for epoch in range(1,201):
        order=np.random.default_rng(46+epoch).permutation(len(train));losses=[]
        for i in order:
            f,t,s=cache[train[i]];opt.zero_grad()
            loss=probe.loss(h,f.cuda(),t.cuda(),s.cuda());loss.backward();opt.step();losses.append(float(loss))
        print(json.dumps(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_used=False)),flush=True)
    h.eval().requires_grad_(False)
    torch.save(dict(head={k:v.cpu() for k,v in h.state_dict().items()},epoch=200,manifest_sha256=ms),root/'head_terminal200.pt')
    rows=[]
    for name in val:
        f=features(name);_,s=spatial.point_target(ann[name]['markers_mha'],ann[name]['roi_box_mha'])
        with torch.inference_mode():p=h(f.cuda()).squeeze(-1).sigmoid().cpu().numpy()
        pred=probe.peaks(p,s);baseline=probe.peaks(np.full_like(p,prior),s)
        pts=ann[name]['markers_mha']
        rows.append(dict(id=name,probe=spatial.match_points(pred,pts),baseline=spatial.match_points(baseline,pts),count_absolute_error=abs(len(pred)-len(pts))))
    result=dict(status='completed',manifest_sha256=ms,finished_utc=now(),epoch=200,rows=rows,mean_f1=float(np.mean([x['probe']['f1'] for x in rows])),baseline_mean_f1=float(np.mean([x['baseline']['f1'] for x in rows])),training_images=20,validation_images=5,validation_training_presentations=0,checkpoint_sha256=sha(root/'head_terminal200.pt'),limits=m['limits'],promotion='requires separate evidence review; never automatic')
    save(root/'result.json',result);save(root/'complete.json',dict(status='completed',result_sha256=sha(root/'result.json')))
if __name__=='__main__':main()
