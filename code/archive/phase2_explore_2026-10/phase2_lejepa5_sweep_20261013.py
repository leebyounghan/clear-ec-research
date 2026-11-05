"""SSL5 full-tuning head-learning-rate x gradient-clipping sweep.
Epoch20 terminal is primary; all epoch weights and diagnostics are retained.
"""
import argparse
from contextlib import nullcontext
import copy
import datetime
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import phase2_lejepa5_regression_20261008 as control
import phase2_foundation_train_20261002 as b
import phase2_lejepa_regression_20261003 as prior
from phase2_foundation_preflight_20261002 import digest,save

EXPERIMENT=Path('experiments/phase2_lejepa5_sweep_20261013')
CONTROL=Path('experiments/phase2_lejepa5_regression_20261008')
REUSED={
    'head1e3_clip1':CONTROL,
    'head3e4_clip1':Path('experiments/phase2_lejepa5_tuning_20261012/arms/full_head_low'),
}
ARMS={
    'head1e4_clip1':dict(mode='full',encoder_lr=1e-5,head_lr=1e-4,gradient_clip=1.),
    'head1e4_clip3':dict(mode='full',encoder_lr=1e-5,head_lr=1e-4,gradient_clip=3.),
    'head3e4_clip3':dict(mode='full',encoder_lr=1e-5,head_lr=3e-4,gradient_clip=3.),
    'head1e3_clip3':dict(mode='full',encoder_lr=1e-5,head_lr=1e-3,gradient_clip=3.),
}
ROOT=EXPERIMENT/'arms/head1e4_clip1'
ARM='head1e4_clip1'

def select(arm):
    global ROOT,ARM
    ARM=arm;ROOT=EXPERIMENT/'arms'/arm

BASE=Path('experiments/phase2_foundation_train_20261002')
ONE=Path('experiments/phase2_lejepa_regression_20261003')
SSL=Path('experiments/phase2_lejepa_total5_20261003')
WEIGHT=SSL/'run/last.pt'
ARCH='cpdino_vitb'
EXPECTED='6f1beb2d7dd1a34c7132cf65ca3cc80426c1428f6345b6569af1c538a7f39042'


def read(p): return json.loads(Path(p).read_text())


def prepare():
    for arm in ARMS:
        assert not (EXPERIMENT/'arms'/arm/'manifest.json').exists(),arm
    assert shutil.disk_usage(EXPERIMENT).free>50*2**30, 'Need 50 GiB free for retained epoch weights'
    control.ROOT=CONTROL;old=control.verify()
    ids,y,tr,va=b.arrays()
    rr=CONTROL/'runs/fold3'/ARCH
    result=prior.validate_result(rr,digest(CONTROL/'manifest.json'))
    assert len(result['history'])==20 and digest(rr/'predictions.npz')==result['predictions_sha']
    with np.load(rr/'predictions.npz') as z:
        np.testing.assert_array_equal(z['ids'],ids[va]);np.testing.assert_array_equal(z['target'],y[va])
        score=b.metrics(z['pred'],z['target'])
    for k in ('CD','CV','HEX','mean'):assert abs(score[k]-result['score'][k])<1e-6
    names=['phase2_foundation_train_20261002.py','phase2_foundation_preflight_20261002.py',
           'phase2_lejepa_regression_20261003.py','phase2_lejepa5_regression_20261008.py']
    for n in names:
        assert digest(EXPERIMENT/'source'/n)==old['hashes'][str(CONTROL/'source'/n)]
    reused_files=[]
    for name,root in REUSED.items():
        rm=read(root/'manifest.json');rrr=root/'runs/fold3'/ARCH
        res=prior.validate_result(rrr,digest(root/'manifest.json'))
        assert rm['ssl_checkpoint_sha']==EXPECTED
        assert rm['recipe']['encoder_lr']==1e-5 and rm['recipe']['gradient_clip']==1.
        assert rm['recipe']['head_lr']==(1e-3 if name=='head1e3_clip1' else 3e-4)
        assert rm.get('mode','full')=='full'
        assert len(res['history'])==20 and digest(rrr/'predictions.npz')==res['predictions_sha']
        with np.load(rrr/'predictions.npz') as z:
            np.testing.assert_array_equal(z['indices'],va);np.testing.assert_array_equal(z['target'],y[va])
        reused_files += [root/'manifest.json',rrr/'result.json',rrr/'complete.json',rrr/'predictions.npz']
    for arm,config in ARMS.items():
        select(arm);m=copy.deepcopy(old)
        m['recipe'].update(encoder_lr=config['encoder_lr'],head_lr=config['head_lr'],gradient_clip=config['gradient_clip'])
        files=[EXPERIMENT/'source'/n for n in names]
        files += [EXPERIMENT/'source/phase2_lejepa5_sweep_20261013.py',CONTROL/'manifest.json',
                  rr/'result.json',rr/'complete.json',rr/'predictions.npz']
        m['hashes'].update({str(p):digest(p) for p in files+reused_files})
        m.update(created_utc=time.strftime('%FT%TZ',time.gmtime()),arm=arm,mode=config['mode'],
                 control_manifest_sha=digest(CONTROL/'manifest.json'),control_score=score,
                 diagnostics=dict(epoch_weights='every epoch; model only, no exact resume',
                     epoch_predictions=True,online_train_loss=True,online_train_mape=True,
                     train_probe_n=256,train_probe_seed=12012,train_probe_mode='eval, no augmentation',
                     gradient_norms=True,clipping_fraction=True,checkpoint_selection='terminal20 primary; best is retrospective diagnostic'),
                 limits=['Reused fold3, single seed; tuning screen, not independent generalization',
                         'Historical full-tuning control; two GPU matched batch16 and same seed/data order',
                         'All six comparisons are full tuning; only head LR and gradient clipping differ',
                         'Online train metrics use augmented samples and changing weights; fixed train probe separately recorded',
                         'Patient independence and near-duplicate leakage unverified; T4 inference unverified'])
        save(ROOT/'manifest.json',m)
        print(json.dumps(dict(status='prepared',arm=arm,config=config,control=score)),flush=True)


def verify():
    b.ROOT=ROOT;m=b.verify();old=read(CONTROL/'manifest.json')
    assert m['arm']==ARM and m['mode']==ARMS[ARM]['mode']
    for k,v in old['recipe'].items():
        expected=ARMS[ARM][k] if k in ('encoder_lr','head_lr','gradient_clip') else v
        assert m['recipe'][k]==expected,k
    for k in ('log_mean','log_std','train_order_base_sha','validation_ids_sha','raw_sha256','ssl_checkpoint_sha','expected_head_sha'):
        assert m[k]==old[k],k
    assert digest(CONTROL/'manifest.json')==m['control_manifest_sha']
    return m


def setup(m):
    net,opt,mu,sigma,details=b.setup(ARCH,m)
    head=prior.state_sha(net[1:].state_dict());assert head==m['expected_head_sha']
    c=torch.load(WEIGHT,map_location='cpu',weights_only=True)
    assert c['epoch']==5 and c['step']==1120 and c['manifest_sha']==m['ssl_manifest_sha']
    expected=prior.state_sha(c['encoder'])
    net[0].original.load_state_dict(c['encoder'],strict=True)
    assert prior.state_sha(net[0].original.state_dict())==expected
    assert prior.state_sha(net[1:].state_dict())==head and not opt.state
    assert not any(isinstance(v,torch.nn.modules.batchnorm._BatchNorm) for v in net.modules())
    details.update(ssl_checkpoint_sha=EXPECTED,ssl_epochs=5,encoder_tensor_sha=expected,
                   head_initial_tensor_sha=head,projection_or_ssl_optimizer_loaded=False,
                   distributed_execution=m['execution'])
    if m['mode']=='head_only':
        net[0].requires_grad_(False);net[0].eval()
        opt=torch.optim.AdamW([{'params':[],'lr':0.},
                               {'params':list(net[1:].parameters()),'lr':m['recipe']['head_lr']}],
                              weight_decay=m['recipe']['weight_decay'],fused=True)
    else:
        assert all(p.requires_grad for p in net[0].parameters())
    details.update(mode=m['mode'],encoder_trainable=sum(p.numel() for p in net[0].parameters() if p.requires_grad),
                   head_trainable=sum(p.numel() for p in net[1:].parameters() if p.requires_grad))
    details['trainable_parameter_count']=sum(p.numel() for p in net.parameters() if p.requires_grad)
    return net,opt,mu,sigma,details


def init_dist():
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=20))
    assert dist.get_world_size()==2 and rank==dist.get_rank()
    return rank


def global_loader(order,y,n,epoch):
    return DataLoader(b.Frames(order,y,n,epoch,True),batch_size=16,shuffle=False,
                      num_workers=2,pin_memory=True,prefetch_factor=1,drop_last=False)


def distributed_backward(ddp,x,t,mu,sigma,rank):
    n=len(t); stats=torch.zeros(7,device=mu.device,dtype=torch.float64)
    for j in range(2):
        start=(2*j+rank)*4;count=max(0,min(4,n-start))
        xx=x[start:start+count] if count else x[:1]
        yy=t[start:start+count] if count else t[:1]
        with ddp.no_sync() if j==0 else nullcontext():
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=ddp(b.inputs(xx,ARCH)); loss=b.loss_fn(z,yy.cuda(non_blocking=True),mu,sigma)
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            if count:
                with torch.no_grad():
                    zt=(yy.cuda(non_blocking=True).log()-mu)/sigma
                    rel=torch.expm1((z.float()-zt)*sigma).abs()
                    stats[0]+=loss.detach().double()*count
                    stats[1:4]+=rel.double().sum(0)*100
                    stats[4:7]+=(z.float()-zt).abs().double().sum(0)
            (loss*(2*count/n)).backward()
    return stats



def cpu_tests():
    verify(); b.tests()
    for n in (1,4,5,8,9,12,15,16):
        covered=[];weight=0.
        for rank in (0,1):
            for j in range(2):
                start=(2*j+rank)*4;count=max(0,min(4,n-start))
                covered.extend(range(start,start+count));weight+=2*count/n/2
        assert sorted(covered)==list(range(n)) and abs(weight-1)<1e-12
    # Independent sample-weighted telemetry test includes partial and empty microbatches.
    mu=torch.tensor([1.,2.,3.]);sigma=torch.tensor([.18,.19,.16])
    for n in (1,9,16):
        z=torch.arange(n*3,dtype=torch.float32).reshape(n,3)/100
        targets=torch.exp(mu+sigma*(z+.13))
        total=torch.zeros(7,dtype=torch.float64)
        for rank in (0,1):
            for j in range(2):
                start=(2*j+rank)*4;count=max(0,min(4,n-start))
                if not count:continue
                zz=z[start:start+count];tt=targets[start:start+count]
                zt=(tt.log()-mu)/sigma
                total[0]+=b.loss_fn(zz,tt,mu,sigma).double()*count
                total[1:4]+=torch.expm1((zz-zt)*sigma).abs().double().sum(0)*100
                total[4:7]+=(zz-zt).abs().double().sum(0)
        torch.testing.assert_close((total/n)[0],b.loss_fn(z,targets,mu,sigma).double(),rtol=1e-5,atol=1e-7)
        expected=torch.expm1((z-(targets.log()-mu)/sigma)*sigma).abs().double().mean(0)*100
        torch.testing.assert_close((total/n)[1:4],expected)
    # Clipping threshold must change the actual gradient scale, not only metadata.
    for threshold in (1.,3.):
        p=torch.nn.Parameter(torch.ones(2));p.grad=torch.tensor([3.,4.])
        original=torch.nn.utils.clip_grad_norm_([p],threshold,error_if_nonfinite=True)
        assert abs(original.item()-5.)<1e-6
        assert abs(p.grad.norm().item()-threshold)<1e-5
    save(ROOT/'cpu_tests.json',dict(status='passed',gradient_clip_threshold_verified=True,telemetry_weighting_verified=True,ddp_sample_weights_and_partial_groups=True,
                                  manifest_sha=digest(ROOT/'manifest.json')))
    print('PASS: DDP sample weights, global groups, partial batches, original loss and metrics',flush=True)


def smoke():
    m=verify();rank=init_dist();net,opt,mu,sigma,details=setup(m)
    ddp=DDP(net,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=True)
    ids,y,tr,va=b.arrays()
    # Deterministic eval-mode gradient check includes9 samples and an empty rank1 microbatch.
    ds=b.Frames(tr[:9],y,len(ids));data=[ds[i] for i in range(9)]
    x=torch.stack([v[0] for v in data]);t=torch.stack([v[1] for v in data])
    ddp.eval();opt.zero_grad(set_to_none=True)
    distributed_backward(ddp,x,t,mu,sigma,rank)
    evidence=[None]
    if rank==0:
        ref,_,rmu,rsigma,_=setup(m);ref.eval();ref.zero_grad(set_to_none=True)
        for start in range(0,9,4):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=ref(b.inputs(x[start:start+4],ARCH))
                loss=b.loss_fn(z,t[start:start+4].cuda(),rmu,rsigma)
            (loss*len(t[start:start+4])/9).backward()
        error=0.;norm=0.;maximum=0.
        for (name,a),(other,c) in zip(net.named_parameters(),ref.named_parameters()):
            assert name==other and (a.grad is None)==(c.grad is None)
            if a.grad is not None:
                d=(a.grad-c.grad).float();error+=d.square().sum().item()
                norm+=c.grad.float().square().sum().item();maximum=max(maximum,d.abs().max().item())
        relative=math.sqrt(error/max(norm,1e-30))
        evidence[0]=dict(relative_gradient_l2=relative,max_abs_gradient_error=maximum,
                         passed=relative<.002,includes_final9_samples_and_zero_weight_dummy=True)
        del ref
    dist.broadcast_object_list(evidence,src=0)
    assert evidence[0]['passed'],evidence[0]
    before=prior.state_sha(net[0].original.state_dict())
    head_before=prior.state_sha(net[1:].state_dict())
    if m['mode']=='head_only':
        assert all(p.grad is None for p in net[0].parameters())
    else:
        blocks={re.search(r'blocks\.\d+',n).group() for n,p in net[0].named_parameters() if re.search(r'blocks\.\d+',n)}
        reached={re.search(r'blocks\.\d+',n).group() for n,p in net[0].named_parameters()
                 if re.search(r'blocks\.\d+',n) and p.grad is not None and p.grad.abs().max().item()>0}
        assert blocks==reached and len(blocks)==12

    torch.nn.utils.clip_grad_norm_(net.parameters(),m['recipe']['gradient_clip'],error_if_nonfinite=True);opt.step()
    after=prior.state_sha(net[0].original.state_dict())
    assert (after==before)==(m['mode']=='head_only')
    assert prior.state_sha(net[1:].state_dict())!=head_before
    hashes=[None,None];dist.all_gather_object(hashes,after);assert hashes[0]==hashes[1]
    # Train-mode smoke confirms synchronization with the real training path.
    ddp.train()
    if m['mode']=='head_only':net[0].eval()
    opt.zero_grad(set_to_none=True)
    distributed_backward(ddp,x,t,mu,sigma,rank)
    torch.nn.utils.clip_grad_norm_(net.parameters(),m['recipe']['gradient_clip'],error_if_nonfinite=True);opt.step()
    assert (prior.state_sha(net[0].original.state_dict())==before)==(m['mode']=='head_only')
    hashes=[None,None];dist.all_gather_object(hashes,prior.state_sha(net.state_dict()))
    assert hashes[0]==hashes[1]
    if rank==0:
        report=dict(status='passed',manifest_sha=digest(ROOT/'manifest.json'),details=details,
                    single_gpu_reference_gradient_check=evidence[0],ranks_synchronized=True,
                    encoder_freeze_or_update_verified=True,head_updated=True)
        if (ROOT/'smoke.json').exists(): raise FileExistsError('Smoke already exists')
        save(ROOT/'smoke.json',report);print(json.dumps(report),flush=True)
    dist.barrier();dist.destroy_process_group()


def train():
    m=verify();rank=init_dist();run=ROOT/'runs/fold3'/ARCH
    for name in ('cpu_tests.json','smoke.json'):
        chk=read(ROOT/name);assert chk['status']=='passed' and chk['manifest_sha']==digest(ROOT/'manifest.json')
    if rank==0:
        if run.exists(): raise FileExistsError('No overwrite or implicit resume')
        run.mkdir(parents=True)
    dist.barrier();net,opt,mu,sigma,details=setup(m)
    ddp=DDP(net,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=True)
    ids,y,tr,va=b.arrays();updates=math.ceil(len(tr)/16);step=0;history=[];wall=time.monotonic()
    val=b.loader(va,y,len(ids)) if rank==0 else None
    probe=np.sort(np.random.default_rng(12012).choice(tr,256,replace=False))
    probe_loader=b.loader(probe,y,len(ids)) if rank==0 else None
    initial_encoder=prior.state_sha(net[0].original.state_dict())

    if rank==0: save(run/'progress.json',dict(status='starting',epochs=20,world_size=2,details=details))
    for epoch in range(20):
        order=np.random.default_rng(42+epoch).permutation(tr)
        dl=global_loader(order,y,len(ids),epoch);seen=[];tick=time.monotonic()
        ddp.train()
        if m['mode']=='head_only':net[0].eval()
        torch.cuda.reset_peak_memory_stats()
        totals=torch.zeros(7,device=mu.device,dtype=torch.float64)
        norm_sum=torch.zeros((),device=mu.device);norm_max=torch.zeros_like(norm_sum);clipped=torch.zeros_like(norm_sum)

        for x,t,ix in dl:
            if step<updates:ratio=(step+1)/updates
            else:ratio=.1+.9*.5*(1+math.cos(math.pi*(step-updates)/max(1,updates*19-1)))
            opt.param_groups[0]['lr']=m['recipe']['encoder_lr']*ratio
            opt.param_groups[1]['lr']=m['recipe']['head_lr']*ratio
            opt.zero_grad(set_to_none=True);totals+=distributed_backward(ddp,x,t,mu,sigma,rank)
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),m['recipe']['gradient_clip'],error_if_nonfinite=True)
            norm_sum+=norm.detach();norm_max=torch.maximum(norm_max,norm.detach());clipped+=(norm.detach()>m['recipe']['gradient_clip']).float()
            opt.step();step+=1;seen.extend(ix.tolist())
            if rank==0 and (step%25==0 or step==1):
                item=dict(status='training',epoch=epoch+1,epochs=20,seen=len(seen),train_n=len(tr),
                          updates=step,world_size=2,arm=ARM,gradient_norm=norm.item(),
                          encoder_lr=opt.param_groups[0]['lr'],head_lr=opt.param_groups[1]['lr'],
                          seconds_this_epoch=time.monotonic()-tick,
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                save(run/'progress.json',item);print(json.dumps(item),flush=True)
        np.testing.assert_array_equal(np.array(seen),order);assert step==(epoch+1)*updates
        dist.all_reduce(totals)
        current_encoder=prior.state_sha(net[0].original.state_dict())
        assert (current_encoder==initial_encoder)==(m['mode']=='head_only')
        dist.barrier();train_seconds=time.monotonic()-tick

        if rank==0:
            pred=b.evaluate(net,ARCH,val,mu,sigma,va);score=b.metrics(pred,y[va])
            # Additional diagnostics must not alter the next training epoch RNG.
            with torch.random.fork_rng(devices=[rank]):
                probe_pred=b.evaluate(net,ARCH,probe_loader,mu,sigma,probe)
            probe_score=b.metrics(probe_pred,y[probe])
            vals=(totals/len(tr)).cpu().tolist()
            telemetry=dict(online_train_loss=vals[0],online_train_mape=dict(zip(('CD','CV','HEX'),vals[1:4])),
                online_standardized_log_l1=dict(zip(('CD','CV','HEX'),vals[4:7])),
                train_probe_score=probe_score,gradient_norm_mean=(norm_sum/updates).item(),
                gradient_norm_max=norm_max.item(),gradient_clip_fraction=(clipped/updates).item(),gradient_clip_threshold=m['recipe']['gradient_clip'],
                encoder_lr=opt.param_groups[0]['lr'],head_lr=opt.param_groups[1]['lr'],
                encoder_frozen=m['mode']=='head_only')
            item=dict(epoch=epoch+1,score=score,train_seconds=train_seconds,**telemetry,

                      epoch_seconds=time.monotonic()-tick,updates=step,
                      order_sha=hashlib.sha256(order.tobytes()).hexdigest(),
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            history.append(item);save(run/'history.json',history)
            torch.save(dict(model=net.state_dict(),epoch=epoch+1,architecture=ARCH,
                            manifest_sha=digest(ROOT/'manifest.json'),log_mean=m['log_mean'],log_std=m['log_std'],
                            note='Model-only epoch snapshot; terminal20 remains primary'),run/f'epoch_{epoch+1:02d}.pt')
            np.savez_compressed(run/f'predictions_epoch_{epoch+1:02d}.npz',indices=va,ids=ids[va],target=y[va],pred=pred)
            tmp=run/'last.pt.tmp'

            torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,
                            architecture=ARCH,manifest_sha=digest(ROOT/'manifest.json'),
                            log_mean=m['log_mean'],log_std=m['log_std'],history=history,
                            note='No automatic resume; rank1 RNG not checkpointed'),tmp)
            tmp.replace(run/'last.pt')
            save(run/'progress.json',dict(status='epoch_complete',**item))
            print(json.dumps(dict(epoch_complete=epoch+1,**item)),flush=True)
        dist.barrier()
    hashes=[None,None];dist.all_gather_object(hashes,prior.state_sha(net.state_dict()));assert hashes[0]==hashes[1]
    if rank==0:
        np.savez_compressed(run/'predictions.npz',indices=va,ids=ids[va],target=y[va],pred=pred)
        result=dict(architecture=ARCH,fold=3,score=score,history=history,details=details,
                    seconds=time.monotonic()-wall,manifest_sha=digest(ROOT/'manifest.json'),
                    checkpoint_sha=digest(run/'last.pt'),predictions_sha=digest(run/'predictions.npz'),
                    ranks_synchronized=True,limits=m['limits'],official_submission_changed=False)
        save(run/'result.json',result);save(run/'complete.json',dict(status='completed',result_sha=digest(run/'result.json')))
    dist.barrier();dist.destroy_process_group()


def audit():
    if (EXPERIMENT/'audit.json').exists():raise FileExistsError('No overwrite audit')
    scores={};histories=[];reference=None;diagnostics={};configs={}
    for name,root in list(REUSED.items())+[(a,EXPERIMENT/'arms'/a) for a in ARMS]:
        if name in ARMS:select(name);verify()
        manifest=read(root/'manifest.json');run=root/'runs/fold3'/ARCH
        assert manifest['ssl_checkpoint_sha']==EXPECTED
        configs[name]={k:manifest['recipe'][k] for k in ('encoder_lr','head_lr','gradient_clip','epochs','seed','effective_batch')}
        r=prior.validate_result(run,digest(root/'manifest.json'));assert len(r['history'])==20
        assert digest(run/'predictions.npz')==r['predictions_sha']
        with np.load(run/'predictions.npz') as z:
            fields={k:z[k].copy() for k in ('indices','ids','target')}
            if reference is None:reference=fields
            for k in fields:np.testing.assert_array_equal(fields[k],reference[k])
            scores[name]=b.metrics(z['pred'],z['target'])
        for k in ('CD','CV','HEX','mean'):assert abs(scores[name][k]-r['score'][k])<1e-6
        histories.append(r['history'])
        best=min(r['history'],key=lambda x:x['score']['mean'])
        diagnostics[name]=dict(best_epoch=best['epoch'],best_score=best['score'],retrospective_not_primary=True,
                              last5_mean=float(np.mean([x['score']['mean'] for x in r['history'][-5:]])))
        if name in ARMS:
            assert len(list(run.glob('epoch_*.pt')))==20
            assert len(list(run.glob('predictions_epoch_*.npz')))==20
    for rows in zip(*histories):
        assert len({(r['epoch'],r['updates'],r['order_sha']) for r in rows})==1
    ranking=sorted(scores,key=lambda a:(scores[a]['mean'],a))
    report=dict(status='ssl5_lr_clip_sweep_verified',scores=scores,configs=configs,lower_is_better=True,
        delta_vs_baseline={a:{k:scores[a][k]-scores['head1e3_clip1'][k] for k in ('CD','CV','HEX','mean')} for a in scores},
        retrospective_best=diagnostics,primary='fixed terminal20',fold=3,identical_orders_and_validation=True,
        ranking=ranking,shortlist=ranking[:2],selection_rule='Lowest terminal20 mean MAPE; alphabetical tie-break; best epoch diagnostic only',
        confirmation_status='not_started; separate seed and fold-specific SSL/data preflight required',
        limitations=['Reused development fold and single seed; shortlist is not independently validated'])
    save(EXPERIMENT/'audit.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','cpu_tests','smoke','train','audit'])
    p.add_argument('--arm',choices=list(ARMS),default='head1e4_clip1')
    args=p.parse_args();select(args.arm);globals()[args.command]()
