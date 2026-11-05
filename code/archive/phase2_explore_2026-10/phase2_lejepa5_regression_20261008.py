"""LeJEPA epoch5 -> 20-epoch full-native regression; 2 GPU matched-global-batch.

Reuses the frozen baseline data, model, head, loss, evaluation and LR recipe.
DDP halves four baseline microbatches between two ranks. Actual sample weights
compensate DDP averaging, including the final9-image group. A zero-weight dummy
forward keeps both ranks synchronized when a microbatch is empty. There is no
BatchNorm in the regression model. GPU smoke compares distributed gradients to
a one-GPU reference before any real training. Bitwise train parity is not claimed.
"""
import argparse
from contextlib import nullcontext
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import phase2_foundation_train_20261002 as b
import phase2_lejepa_regression_20261003 as prior
from phase2_foundation_preflight_20261002 import digest,save

ROOT=Path('experiments/phase2_lejepa5_regression_20261008')
BASE=Path('experiments/phase2_foundation_train_20261002')
ONE=Path('experiments/phase2_lejepa_regression_20261003')
SSL=Path('experiments/phase2_lejepa_total5_20261003')
WEIGHT=SSL/'run/last.pt'
ARCH='cpdino_vitb'
EXPECTED='6f1beb2d7dd1a34c7132cf65ca3cc80426c1428f6345b6569af1c538a7f39042'


def read(p): return json.loads(Path(p).read_text())


def prepare():
    if (ROOT/'manifest.json').exists(): raise FileExistsError('Frozen manifest exists')
    b.ROOT=BASE; m=copy.deepcopy(b.verify()); ids,y,tr,va=b.arrays()
    assert digest(b.RAW)==m['raw_sha256']
    sm=read(SSL/'manifest.json')
    for p,h in sm['hashes'].items(): assert digest(p)==h,p
    sr=prior.validate_result(SSL/'run',digest(SSL/'manifest.json'))
    assert digest(WEIGHT)==EXPECTED
    assert sr['epochs']==5 and sr['total_steps']==1120 and sr['additional_steps']==896
    assert sr['validation_images_used']==0 and sr['unique_train_images']==7161
    assert sm['validation_ids_sha']==m['validation_ids_sha']
    assert sm['train_ids_sha']==hashlib.sha256('\n'.join(ids[tr]).encode()).hexdigest()
    assert sm['raw_sha256']==m['raw_sha256']
    for ep in sr['epochs_detail']:
        order=np.random.default_rng(1003+ep['epoch']-1).permutation(tr).astype(np.int64)
        assert ep['order_sha']==hashlib.sha256(order.tobytes()).hexdigest()
        assert ep['unique_train_images']==7161 and ep['validation_images_used']==0
    refs={}; files=[]
    for name,root in [('ssl0',BASE),('ssl1',ONE)]:
        rr=root/'runs/fold3'/ARCH
        result=prior.validate_result(rr,digest(root/'manifest.json'))
        assert digest(rr/'predictions.npz')==result['predictions_sha']
        assert read(root/'manifest.json')['recipe']==m['recipe']
        with np.load(rr/'predictions.npz') as z:
            np.testing.assert_array_equal(z['indices'],va);np.testing.assert_array_equal(z['ids'],ids[va])
            np.testing.assert_array_equal(z['target'],y[va]); score=b.metrics(z['pred'],z['target'])
        for k in ('CD','CV','HEX','mean'): assert abs(score[k]-result['score'][k])<1e-6
        refs[name]=score
        files.extend([root/'manifest.json',rr/'result.json',rr/'complete.json',rr/'predictions.npz'])
    names=['phase2_foundation_train_20261002.py','phase2_foundation_preflight_20261002.py',
           'phase2_lejepa_regression_20261003.py']
    om=read(ONE/'manifest.json')
    for name in names:
        original=ONE/'source'/name
        assert digest(ROOT/'source'/name)==om['hashes'][str(original)]
    files.extend([ROOT/'source'/n for n in names])
    files.extend([ROOT/'source/phase2_lejepa5_regression_20261008.py',SSL/'manifest.json',
                  SSL/'run/result.json',SSL/'run/complete.json',WEIGHT,ONE/'cpu_tests.json'])
    m.update(created_utc=time.strftime('%FT%TZ',time.gmtime()),architectures=[ARCH],gpu_parallelism=2,
             references=refs,ssl_epochs=5,ssl_steps=1120,ssl_checkpoint_sha=EXPECTED,
             ssl_manifest_sha=digest(SSL/'manifest.json'),
             expected_head_sha=read(ONE/'cpu_tests.json')['initialization']['head_initial_tensor_sha'],
             execution=dict(world_size=2,global_batch=16,rank_micro_batch=4,rank_accumulations=2,
                            global_order='same seed42+epoch ordering as baseline',
                            empty_microbatch='zero weighted dummy; no BatchNorm',
                            validation='rank0 alone, baseline single-view batch4'),
             limits=['Same reused fold3 and single seed; not an independent Phase II score',
                     '2GPU DDP versus prior1GPU; same global batch/updates, but floating-point and stochastic execution can differ',
                     'Patient independence and near-duplicate leakage unverified',
                     'Epoch20 terminal primary; intermediate best diagnostic only',
                     'T4 inference remains unverified'])
    m['hashes'].update({str(p):digest(p) for p in files})
    save(ROOT/'manifest.json',m)
    print(json.dumps(dict(status='prepared',references=refs,execution=m['execution'])),flush=True)


def verify():
    b.ROOT=ROOT;m=b.verify();old=read(BASE/'manifest.json')
    assert m['recipe']==old['recipe']
    for k in ('log_mean','log_std','train_order_base_sha','validation_ids_sha','raw_sha256'):
        assert m[k]==old[k]
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
    n=len(t)
    for j in range(2):
        start=(2*j+rank)*4;count=max(0,min(4,n-start))
        xx=x[start:start+count] if count else x[:1]
        yy=t[start:start+count] if count else t[:1]
        with ddp.no_sync() if j==0 else nullcontext():
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=ddp(b.inputs(xx,ARCH)); loss=b.loss_fn(z,yy.cuda(non_blocking=True),mu,sigma)
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            (loss*(2*count/n)).backward()


def cpu_tests():
    verify(); b.tests()
    for n in (1,4,5,8,9,12,15,16):
        covered=[];weight=0.
        for rank in (0,1):
            for j in range(2):
                start=(2*j+rank)*4;count=max(0,min(4,n-start))
                covered.extend(range(start,start+count));weight+=2*count/n/2
        assert sorted(covered)==list(range(n)) and abs(weight-1)<1e-12
    save(ROOT/'cpu_tests.json',dict(status='passed',ddp_sample_weights_and_partial_groups=True,
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
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
    after=prior.state_sha(net[0].original.state_dict());assert after!=before
    hashes=[None,None];dist.all_gather_object(hashes,after);assert hashes[0]==hashes[1]
    # Train-mode smoke confirms synchronization with the real training path.
    ddp.train();opt.zero_grad(set_to_none=True)
    distributed_backward(ddp,x,t,mu,sigma,rank)
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
    hashes=[None,None];dist.all_gather_object(hashes,prior.state_sha(net[0].original.state_dict()))
    assert hashes[0]==hashes[1]
    if rank==0:
        report=dict(status='passed',manifest_sha=digest(ROOT/'manifest.json'),details=details,
                    single_gpu_reference_gradient_check=evidence[0],ranks_synchronized=True)
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
    if rank==0: save(run/'progress.json',dict(status='starting',epochs=20,world_size=2,details=details))
    for epoch in range(20):
        order=np.random.default_rng(42+epoch).permutation(tr)
        dl=global_loader(order,y,len(ids),epoch);seen=[];tick=time.monotonic()
        ddp.train();torch.cuda.reset_peak_memory_stats()
        for x,t,ix in dl:
            if step<updates:ratio=(step+1)/updates
            else:ratio=.1+.9*.5*(1+math.cos(math.pi*(step-updates)/max(1,updates*19-1)))
            opt.param_groups[0]['lr']=m['recipe']['encoder_lr']*ratio
            opt.param_groups[1]['lr']=m['recipe']['head_lr']*ratio
            opt.zero_grad(set_to_none=True);distributed_backward(ddp,x,t,mu,sigma,rank)
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            opt.step();step+=1;seen.extend(ix.tolist())
            if rank==0 and (step%25==0 or step==1):
                item=dict(status='training',epoch=epoch+1,epochs=20,seen=len(seen),train_n=len(tr),
                          updates=step,world_size=2,gradient_norm=norm.item(),
                          seconds_this_epoch=time.monotonic()-tick,
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                save(run/'progress.json',item);print(json.dumps(item),flush=True)
        np.testing.assert_array_equal(np.array(seen),order);assert step==(epoch+1)*updates
        dist.barrier();train_seconds=time.monotonic()-tick
        if rank==0:
            pred=b.evaluate(net,ARCH,val,mu,sigma,va);score=b.metrics(pred,y[va])
            item=dict(epoch=epoch+1,score=score,train_seconds=train_seconds,
                      epoch_seconds=time.monotonic()-tick,updates=step,
                      order_sha=hashlib.sha256(order.tobytes()).hexdigest(),
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            history.append(item);save(run/'history.json',history)
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
    m=verify()
    if (ROOT/'audit.json').exists():raise FileExistsError('Audit already exists')
    scores={};histories=[];reference=None
    for name,root in [('ssl0',BASE),('ssl1',ONE),('ssl5',ROOT)]:
        run=root/'runs/fold3'/ARCH
        r=prior.validate_result(run,digest(root/'manifest.json'));assert len(r['history'])==20
        assert digest(run/'predictions.npz')==r['predictions_sha']
        with np.load(run/'predictions.npz') as z:
            fields={k:z[k].copy() for k in ('indices','ids','target')}
            if reference is None: reference=fields
            for k in fields:np.testing.assert_array_equal(fields[k],reference[k])
            scores[name]=b.metrics(z['pred'],z['target'])
        for k in ('CD','CV','HEX','mean'):assert abs(scores[name][k]-r['score'][k])<1e-6
        histories.append(r['history'])
    for rows in zip(*histories):
        assert len({(r['epoch'],r['updates'],r['order_sha']) for r in rows})==1
    report=dict(status='ssl0_1_5_comparison_verified',scores=scores,lower_is_better=True,
                delta_ssl5_minus={name:{k:scores['ssl5'][k]-scores[name][k] for k in ('CD','CV','HEX','mean')}
                                 for name in ('ssl0','ssl1')},
                identical_training_orders=True,identical_validation_ids_and_targets=True,
                epoch=20,fold=3,manifest_sha=digest(ROOT/'manifest.json'),limits=m['limits'])
    save(ROOT/'audit.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','cpu_tests','smoke','train','audit'])
    args=p.parse_args();globals()[args.command]()
