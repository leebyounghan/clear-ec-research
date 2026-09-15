"""Official DINO B/8 continuation, native-scale adaptation (not paper reproduction).

All 9000 training images are intentionally used without SSL labels. Any later
fold probe is transductive and must never be represented as clean held-out CV.
"""
import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint

ROOT=Path('experiments/dino_all9000_20260913')
sys.path.insert(0,str(ROOT/'upstream'))
import vision_transformer as vit
from main_dino import DINOLoss
sys.path.insert(0,'experiments/raw_ssl_next_20260913/source')
import next_candidate_20260913 as old
STEPS=6000
BATCH=16
RATES={'lr1e6':1e-6,'lr5e6':5e-6}


class SingleDeviceLoss(DINOLoss):
    # Official loss, with all-reduce replaced by its world-size-one equivalent.
    @torch.no_grad()
    def update_center(self, outputs):
        self.center.mul_(self.center_momentum).add_(outputs.mean(0,keepdim=True),alpha=1-self.center_momentum)


def fast_attention(self,x):
    b,n,c=x.shape
    q,k,v=self.qkv(x).reshape(b,n,3,self.num_heads,c//self.num_heads).permute(2,0,3,1,4).unbind(0)
    x=F.scaled_dot_product_attention(q,k,v,dropout_p=0.,scale=self.scale)
    return self.proj_drop(self.proj(x.transpose(1,2).reshape(b,n,c))),None


def build(grad=False):
    # Retain the complete published projection head, not a randomly reinitialized one.
    model=nn.ModuleDict({'backbone':vit.vit_base(patch_size=8),'head':vit.DINOHead(768,65536)})
    ck=torch.load(ROOT/'initial_full.pth',map_location='cpu',weights_only=True)
    # Published inference-ready head has weight norm folded into plain weight.
    # Restore the equivalent parametrization for continuation; preserve logits.
    state=ck['teacher']
    if 'head.last_layer.weight' in state:
        w=state.pop('head.last_layer.weight')
        state['head.last_layer.weight_v']=w
        state['head.last_layer.weight_g']=w.norm(dim=1,keepdim=True)
    model.load_state_dict(ck['teacher'],strict=True)
    del ck
    for block in model['backbone'].blocks:
        block.attn.forward=fast_attention.__get__(block.attn,type(block.attn))
        if grad:
            fn=block.forward
            block.forward=lambda x,fn=fn:checkpoint.checkpoint(fn,x,use_reentrant=False)
    return model


def image_norm(x):
    mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
    std=x.new_tensor([.229,.224,.225])[None,:,None,None]
    return (x-mean)/std


def forward(model,x):
    return model['head'](model['backbone'](image_norm(x)))


def all_views():
    sys.path.insert(0,'experiments/deploy_region_20260907/source')
    from train_deploy_region import build_regions
    return build_regions('full_quality')


def four_views(views,indices):
    # Both globals are the SAME native field with independent D8/photometric views.
    # The two 128 local crops are native subregions of that field, never resized.
    arrays=[[],[],[],[]]
    for i in indices:
        tile=views.tile(int(i),views.train_origin(int(i)),0)
        for j,size in enumerate((256,256,128,128)):
            r,c=(0,0) if size==256 else tuple(np.random.randint(0,129,size=2))
            a=tile[:,r:r+size,c:c+size]
            a=np.rot90(a,np.random.randint(4),axes=(-2,-1))
            if np.random.rand()<.5:a=a[:,:,::-1]
            a=np.clip(a*np.random.uniform(.95,1.05)+np.random.uniform(-.02,.02),0,1)
            arrays[j].append(np.ascontiguousarray(a))
    return [torch.from_numpy(np.stack(a)).cuda() for a in arrays]


@torch.no_grad()
def diagnostics(model,probe):
    values=[]
    for start in range(0,len(probe),8):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            f=model['backbone'](image_norm(probe[start:start+8]))
        values.append(f.float())
    x=torch.cat(values);x=x-x.mean(0)
    s=torch.linalg.svdvals(x);p=s.square()/s.square().sum().clamp_min(1e-12)
    return dict(std=float(x.std(0,unbiased=False).mean()),rank=float(torch.exp(-(p*p.clamp_min(1e-12).log()).sum())))


def healthy(stats,initial):
    assert all(math.isfinite(v) for v in stats.values())
    assert stats['std']>max(1e-4,.2*initial['std']),'DINO feature contraction'
    assert stats['rank']>.25*initial['rank'],'DINO rank contraction'


def verify():
    old.verify()
    m=old.load(ROOT/'manifest.json')
    for p,h in m['hashes'].items():assert old.sha(p)==h,p
    return m


def prepare():
    assert not (ROOT/'manifest.json').exists()
    old.verify()
    views=all_views();assert views.images.shape[0]==9000
    # No label-dependent filtering: include even the two unusable regression labels.
    assert np.isfinite(views.tile(8999,views.selected[8999,0],0)).all()
    model=build();assert model['backbone'].embed_dim==768
    torch.manual_seed(5)
    layer=vit.Attention(48,6,qkv_bias=True).eval();x=torch.randn(2,12,48)
    a=layer(x)[0];b=fast_attention(layer,x)[0]
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    lossfn=SingleDeviceLoss(11,4,.04,.04,0,1)
    student=torch.randn(8,11,requires_grad=True);teacher=torch.randn(4,11,requires_grad=True)
    loss=lossfn(student,teacher,0);loss.backward()
    assert torch.isfinite(student.grad).all() and teacher.grad is None
    torch.testing.assert_close(lossfn.center,teacher.detach().mean(0,keepdim=True)*.1)
    files=list((ROOT/'source').glob('*.py'))+[ROOT/'initial_full.pth']
    files+=list((ROOT/'upstream').glob('*.py'))+[ROOT/'upstream/LICENSE']
    for p in ('datasets/clear_ec_final9000_20260904/meta.npz','datasets/clear_ec_final9000_20260904/manifest.json','experiments/raw_ssl_next_20260913/manifest.json'):
        files.append(Path(p))
    m=dict(created_at=time.strftime('%FT%T%z'),hashes={str(p):old.sha(p) for p in files},
        upstream_commit=subprocess.check_output(['git','-C',str(ROOT/'upstream'),'rev-parse','HEAD'],text=True).strip(),
        checkpoint_url='https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_full_checkpoint.pth',
        ssl_images=9000,ssl_labels_used=False,includes_all_internal_validation_images=True,
        stages='6000 SSL updates; fixed ridge probes only; no full fine-tune or deployment',
        steps=STEPS,batch=BATCH,rates=RATES,global_views=2,local_views=2,global_size=256,local_size=128,
        resize=False,architecture='official DINO ViT-B/8 + pretrained official 65536 projection head',
        modifications=['native-scale same-field multiview','mild photometric only','SDPA equivalent attention','BF16','single-device center','small-batch EMA .9995 to 1'],
        probe='fixed ridge alpha10; train-standardized last-layer CLS+mean patch features; 5 native views; reused fold2/3, transductive',
        no_automatic_submission=True,max_seconds=8*3600,cpu_tests_passed=True)
    old.save(ROOT/'manifest.json',m);print(json.dumps(m),flush=True)


def worker(tag,smoke):
    verify();old.setup_torch();random.seed(913);np.random.seed(913);torch.manual_seed(913)
    out=ROOT/('smoke' if smoke else 'runs')/tag;out.mkdir(parents=True,exist_ok=False)
    views=all_views();student=build(True).cuda().train();teacher=build().cuda().eval()
    for p in teacher.parameters():p.requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=RATES[tag],weight_decay=.04)
    criterion=SingleDeviceLoss(65536,4,.04,.04,0,1).cuda()
    probe=torch.from_numpy(np.stack([views.tile(i,views.selected[i,0],0) for i in range(32)])).cuda()
    initial=diagnostics(teacher,probe);history=[dict(step=0,**initial)]
    # Center initialized with this domain, using unlabeled probe images, no labels.
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        criterion.center.copy_(torch.cat([forward(teacher,p).float() for p in probe.split(8)]).mean(0,keepdim=True))
    steps=4 if smoke else STEPS;started=time.monotonic();order=np.random.permutation(9000);cursor=0;seen=set()
    for step in range(1,steps+1):
        if cursor+BATCH>9000:
            remaining=order[cursor:];order=np.random.permutation(9000)
            indices=np.concatenate((remaining,order[:BATCH-len(remaining)]));cursor=BATCH-len(remaining)
        else:indices=order[cursor:cursor+BATCH];cursor+=BATCH
        seen.update(map(int,indices));xs=four_views(views,indices)
        progress=(step-1)/max(1,steps-1)
        lr=RATES[tag]*min(step/200,1)*(.1+.9*.5*(1+math.cos(math.pi*progress)))
        for g in optimizer.param_groups:g['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            target=torch.cat([forward(teacher,x) for x in xs[:2]]).float()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=torch.cat([forward(student,x) for x in xs]).float()
        loss=criterion(output,target,0);assert torch.isfinite(loss),'nonfinite DINO loss'
        loss.backward()
        # Protect the pretrained prototype directions for first 200 updates.
        if step<=200:
            for p in student['head'].last_layer.parameters():p.grad=None
        norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        momentum=1-(1-.9995)*(.5*(1+math.cos(math.pi*progress)))
        with torch.no_grad():
            for p,q in zip(student.parameters(),teacher.parameters()):q.mul_(momentum).add_(p,alpha=1-momentum)
        if step%25==0 or step==steps:
            torch.cuda.synchronize();elapsed=time.monotonic()-started
            prob=torch.softmax((target-criterion.center)/.04,dim=-1)
            row=dict(step=step,total_steps=steps,loss=float(loss.detach()),grad_norm=float(norm),
                elapsed_seconds=elapsed,seconds_per_step=elapsed/step,unique_images_seen=len(seen),
                teacher_entropy=float(-(prob*prob.clamp_min(1e-12).log()).sum(-1).mean()),
                at=time.strftime('%FT%T%z'))
            old.save(out/'progress.json',row);print(json.dumps(row),flush=True)
        if step%500==0 or step==steps:
            stats=diagnostics(teacher,probe);healthy(stats,initial);history.append(dict(step=step,**stats));old.save(out/'diagnostics.json',history)
        if step%2000==0 or step==steps:
            torch.save(dict(teacher=teacher.state_dict(),student=student.state_dict(),center=criterion.center,
                optimizer=optimizer.state_dict(),step=step,tag=tag,all9000_ssl=True),out/f'checkpoint_{step:04d}.pth')
    if not smoke:assert len(seen)==9000
    saved=out/f'checkpoint_{steps:04d}.pth';ck=torch.load(saved,map_location='cpu',weights_only=True)
    teacher.load_state_dict(ck['teacher'],strict=True);del ck
    old.save(out/'ssl_complete.json',dict(steps=steps,unique_images_seen=len(seen),strict_reload=True,
        elapsed_seconds=time.monotonic()-started,checkpoint_sha256=old.sha(saved)))
    del student,optimizer,criterion,probe;gc.collect();torch.cuda.empty_cache()
    if not smoke:linear_probe(teacher,out,views)
    old.save(out/'complete.json',dict(smoke=smoke,at=time.strftime('%FT%T%z')))


@torch.no_grad()
def features(model,views):
    model.eval();values=[]
    for i in range(9000):
        x=torch.from_numpy(views.evaluate(i,False)).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            tokens=model['backbone'].get_intermediate_layers(image_norm(x),1)[0]
            f=torch.cat((tokens[:,0],tokens[:,1:].mean(1)),dim=-1).float().mean(0)
        values.append(f.cpu().numpy())
        if (i+1)%1000==0:print(json.dumps(dict(stage='probe_features',images=i+1)),flush=True)
    return np.stack(values)


def ridge_report(x):
    with np.load(old.ALLDATA/'meta.npz') as d:ids,y=d['ids'].astype(str),d['Y'].astype(float)
    with np.load('datasets/clear_ec_official/cache_native_meta_512.npz',allow_pickle=True) as d:old_ids=d['ids'].astype(str)
    lookup={s:i for i,s in enumerate(ids)};mapping=np.array([lookup[s] for s in old_ids]);m=old.load(old.ROOT/'manifest.json')
    rows={};errors=[]
    for f in (2,3):
        r=m['folds'][str(f)]
        with np.load(r['split_path']) as d:tr,va=mapping[d['train']],mapping[d['validation']]
        with np.load(Path(r['reference_root'])/'preflight/reference.npz') as d:keep=d['primary'][d['folds']==f]
        a=x[tr].astype(float);b=x[va].astype(float);mean=a.mean(0);sd=a.std(0).clip(.01);a=(a-mean)/sd;b=(b-mean)/sd
        z=np.log(y[tr]);zm=z.mean(0);z=z-zm
        w=np.linalg.solve(a.T@a+10*np.eye(a.shape[1]),a.T@z)
        pred=np.exp(np.clip(b@w+zm,-20,20));e=(np.abs(pred-y[va])/y[va]*100)[keep]
        assert np.isfinite(e).all();errors.append(e);rows[str(f)]=e.mean(0).tolist()
    e=np.concatenate(errors)
    return dict(n=len(e),CD_CV_HEX=e.mean(0).tolist(),mean=float(e.mean()),per_fold=rows,
        transductive=True,independent_validation=False,alpha=10,full_finetune=False)


def linear_probe(teacher,out,views):
    start=time.monotonic()
    adapted=features(teacher,views);np.save(out/'features_adapted.npy',adapted)
    a=ridge_report(adapted);del adapted
    baseline=build().cuda().eval();base=features(baseline,views);np.save(out/'features_baseline.npy',base)
    b=ridge_report(base)
    old.save(out/'probe_report.json',dict(adapted=a,baseline=b,delta=a['mean']-b['mean'],
        elapsed_seconds=time.monotonic()-start,warning='SSL saw every validation image; not comparable to clean CPSAM CV'))


def run():
    m=verify();assert torch.cuda.device_count()==2
    assert not (ROOT/'status.json').exists(),'Do not restart a partial run'
    logs=ROOT/'logs';logs.mkdir();active={};done=[];failed={};start=time.monotonic()
    def stop(*args):
        for p,_,_ in active.values():
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
        if args:raise SystemExit(128+args[0])
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(gpu,tag,smoke):
        cmd=[sys.executable,str(ROOT/'source/dino_all9000.py'),'worker','--tag',tag]+(['--smoke'] if smoke else [])
        with (logs/f'{tag}_{"smoke" if smoke else "full"}.log').open('x') as log:
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        active[gpu]=(p,tag,smoke)
    try:
        for gpu,tag in enumerate(RATES):launch(gpu,tag,True)
        while active:
            assert time.monotonic()-start<m['max_seconds'],'8h limit'
            for gpu,(p,tag,smoke) in list(active.items()):
                rc=p.poll()
                if rc is None:continue
                del active[gpu]
                assert rc==0,f'{tag} exit {rc}'
                assert (ROOT/('smoke' if smoke else 'runs')/tag/'complete.json').exists()
                if smoke:launch(gpu,tag,False)
                else:done.append(tag)
            old.save(ROOT/'status.json',dict(done=done,active={str(g):dict(tag=t,smoke=s) for g,(_,t,s) in active.items()},elapsed_seconds=time.monotonic()-start,at=time.strftime('%FT%T%z')))
            if active:time.sleep(10)
    except Exception as e:failed['controller']=repr(e);stop()
    old.save(ROOT/'complete.json',dict(done=done,failed=failed,elapsed_seconds=time.monotonic()-start,at=time.strftime('%FT%T%z')))
    if failed:raise RuntimeError(failed)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','worker','run'));p.add_argument('--tag',choices=RATES);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(4)
    if a.command=='worker':worker(a.tag,a.smoke)
    else:globals()[a.command]()
