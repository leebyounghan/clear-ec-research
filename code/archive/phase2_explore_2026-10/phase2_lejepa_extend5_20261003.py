"""Explicit continuation of LeJEPA epoch1 through epoch5, in a new directory.

The frozen pilot module owns the model, views, objective, gradient caching and
optimizer implementation. Only the training duration/epoch index changes.
The epoch1 checkpoint, running regression and official release stay untouched.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
from torch import nn

import phase2_lejepa_20261003 as pilot
from phase2_foundation_preflight_20261002 import digest, save

ROOT=Path('experiments/phase2_lejepa_total5_20261003')
PARENT=Path('experiments/phase2_lejepa_20261003')
WEIGHT=PARENT/'pilot/last.pt'
EXPECTED='f49ed126e05f5ea819d4cb235018dbbeac9adb38a9504fd77519696f3899db65'


def read(path):
    return json.loads(Path(path).read_text())


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape
        assert torch.equal(a.detach().cpu(),b.detach().cpu())
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal_tree(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal_tree(x,y)
    else:
        assert a==b


def prepare():
    if (ROOT/'manifest.json').exists(): raise FileExistsError('Frozen manifest already exists')
    parent=pilot.verify(); ids,tr,va=pilot.split()
    assert digest(pilot.RAW)==parent['raw_sha256']
    result=read(PARENT/'pilot/result.json'); complete=read(PARENT/'pilot/complete.json')
    assert complete['status']=='completed'
    assert complete['result_sha']==digest(PARENT/'pilot/result.json')
    assert result['manifest_sha']==digest(PARENT/'manifest.json')
    assert digest(WEIGHT)==result['checkpoint_sha']==EXPECTED
    assert result['steps']==224 and result['validation_images_used']==0
    assert result['unique_train_images']==result['train_images_seen']==len(tr)
    order=np.random.default_rng(1003).permutation(tr).astype(np.int64)
    assert result['seen_order_sha']==hashlib.sha256(order.tobytes()).hexdigest()
    for name in ('phase2_lejepa_20261003.py','phase2_foundation_preflight_20261002.py'):
        assert digest(ROOT/'source'/name)==parent['hashes'][str(PARENT/'source'/name)]
    files=[ROOT/'source/phase2_lejepa_extend5_20261003.py',ROOT/'source/phase2_lejepa_20261003.py',
           ROOT/'source/phase2_foundation_preflight_20261002.py',PARENT/'manifest.json',
           PARENT/'pilot/result.json',PARENT/'pilot/complete.json',WEIGHT]
    recipe=copy.deepcopy(parent['recipe'])
    recipe.update(epochs=5,status='EXPLICIT_CONTINUATION_EPOCH2_TO5',
                  resume_epoch=1,resume_step=224,additional_epochs=4,
                  order_seed_rule='1003 + zero_based_epoch',
                  lr_policy='retain pilot LR after its completed20step warmup; no warmup restart')
    m=dict(created_utc=time.strftime('%FT%TZ',time.gmtime()),recipe=recipe,
           architecture=parent['architecture'],train_n=len(tr),val_n=len(va),
           train_ids_sha=pilot.id_sha(ids,tr),validation_ids_sha=pilot.id_sha(ids,va),
           parent_manifest_sha=digest(PARENT/'manifest.json'),parent_checkpoint_sha=EXPECTED,
           raw_sha256=parent['raw_sha256'],raw_bytes=parent['raw_bytes'],
           hashes={str(p):digest(p) for p in files},additional_steps=896,total_steps=1120,
           limitations=['SSL only; regression accuracy remains unmeasured',
                        'Same development fold3; patient independence unverified',
                        'Full-native two-view adaptation, not original multi-crop LeJEPA',
                        'No automatic extension past epoch5 or automatic downstream regression'],
           official_submission_changed=False)
    save(ROOT/'manifest.json',m)
    print(json.dumps(dict(status='prepared',recipe=recipe,total_steps=1120,additional_steps=896)),flush=True)


def verify():
    m=read(ROOT/'manifest.json'); parent=pilot.verify()
    for p,h in m['hashes'].items(): assert digest(p)==h,p
    assert digest(PARENT/'manifest.json')==m['parent_manifest_sha']
    for k,v in parent['recipe'].items():
        if k not in ('epochs','status'): assert m['recipe'][k]==v,k
    ids,tr,va=pilot.split()
    assert pilot.id_sha(ids,tr)==m['train_ids_sha']
    assert pilot.id_sha(ids,va)==m['validation_ids_sha']
    return m


def load_parent():
    c=torch.load(WEIGHT,map_location='cpu',weights_only=True)
    assert c['step']==224 and c['manifest_sha']==digest(PARENT/'manifest.json')
    assert len(c['rng_cuda'])==1
    assert c['optimizer']['state']
    assert all(int(s['step'].item())==224 for s in c['optimizer']['state'].values())
    assert c['optimizer']['param_groups'][0]['lr']==pilot.RECIPE['encoder_lr']
    assert c['optimizer']['param_groups'][1]['lr']==pilot.RECIPE['projector_lr']
    return c


def setup():
    enc,proj,reg,opt,details=pilot.setup()
    c=load_parent()
    enc.original.load_state_dict(c['encoder'],strict=True)
    proj.load_state_dict(c['projector'],strict=True)
    opt.load_state_dict(c['optimizer'])
    equal_tree(enc.original.state_dict(),c['encoder'])
    equal_tree(proj.state_dict(),c['projector'])
    equal_tree(opt.state_dict(),c['optimizer'])
    torch.set_rng_state(c['rng_cpu']); torch.cuda.set_rng_state_all(c['rng_cuda'])
    assert torch.equal(torch.get_rng_state(),c['rng_cpu'])
    assert torch.equal(torch.cuda.get_rng_state(),c['rng_cuda'][0])
    details.update(resumed_from=str(WEIGHT),resume_step=224,optimizer_and_rng_restored=True)
    print(json.dumps(dict(resume_verified=True,step=224,encoder_projector_optimizer_exact=True,
                         cpu_cuda_rng_restored=True,warmup_restarted=False)),flush=True)
    return enc,proj,reg,opt,details


def tests():
    verify();pilot.tests();c=load_parent()
    # Serialization/resume must produce the same next update, including
    # optimizer momentum, dropout RNG and BatchNorm buffers.
    torch.manual_seed(99)
    net=nn.Sequential(nn.Linear(7,9),nn.BatchNorm1d(9),nn.Dropout(.2),nn.Linear(9,3))
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    x=torch.randn(8,7)
    def update(model,optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss=model(x).square().mean();loss.backward();optimizer.step()
    update(net,opt)
    f=io.BytesIO();torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()),f)
    f.seek(0);saved=torch.load(f,weights_only=True)
    update(net,opt)
    expected=copy.deepcopy(net.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
    net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
    update(net,opt);equal_tree(net.state_dict(),expected);equal_tree(opt.state_dict(),expected_opt)
    result=dict(status='passed',manifest_sha=digest(ROOT/'manifest.json'),
                parent_step=c['step'],parent_optimizer_steps_all224=True,
                next_update_serialization_parity=True,gradient_cache_tests_passed=True)
    save(ROOT/'cpu_tests.json',result);print(json.dumps(result),flush=True)


def smoke():
    verify()
    if (ROOT/'smoke.json').exists(): raise FileExistsError('No repeat smoke')
    enc,proj,reg,opt,details=setup();ids,tr,va=pilot.split()
    order=np.random.default_rng(1004).permutation(tr)
    x,ix=next(iter(pilot.loader(order[:8],len(ids),1,batch=8)))
    assert not np.intersect1d(ix.numpy(),va).size
    tracked={n:p for n,p in enc.named_parameters() if p.ndim>=2}
    before={n:p.detach().clone() for n,p in tracked.items()}
    value=pilot.update(enc,proj,reg,opt,x,True)
    blocks=lambda names:{re.search(r'blocks\.\d+',n).group() for n in names if re.search(r'blocks\.\d+',n)}
    changed=[n for n,p in tracked.items() if not torch.equal(before[n],p)]
    assert blocks(changed)==blocks(tracked) and any('patch_embed' in n for n in changed)
    assert all(int(s['step'].item())==225 for s in opt.state.values())
    result=dict(status='passed',manifest_sha=digest(ROOT/'manifest.json'),values=value,
                encoder_blocks_updated=sorted(blocks(changed)),optimizer_step_after_update=225,
                checkpoint_overwritten=False,details=details)
    save(ROOT/'smoke.json',result);print(json.dumps(result),flush=True)


def train():
    m=verify();run=ROOT/'run'
    if run.exists(): raise FileExistsError('Never auto-resume or overwrite continuation')
    for name in ('cpu_tests.json','smoke.json'):
        check=read(ROOT/name)
        assert check['status']=='passed' and check['manifest_sha']==digest(ROOT/'manifest.json')
    run.mkdir();enc,proj,reg,opt,details=setup();ids,tr,va=pilot.split()
    reference=read(PARENT/'pilot/result.json')['last']['feature_std']
    history=[];epoch_results=[];started=time.monotonic();global_step=224
    save(run/'progress.json',dict(status='starting',resume_step=224,epochs=5,additional_epochs=4,details=details))
    for epoch in range(2,6):
        order=np.random.default_rng(1003+epoch-1).permutation(tr)
        dl=pilot.loader(order,len(ids),epoch-1)
        seen=[];epoch_start=time.monotonic()
        for step,(x,ix) in enumerate(dl,1):
            assert not np.intersect1d(ix.numpy(),va).size
            tick=time.monotonic();torch.cuda.reset_peak_memory_stats()
            values=pilot.update(enc,proj,reg,opt,x,step==1)
            torch.cuda.synchronize();global_step+=1;seen.extend(ix.tolist())
            if values['feature_std']<max(1e-6,reference*1e-3) or values['projector_std']<1e-6:
                raise ValueError('Representation collapse tripwire: stop without retry')
            item=dict(status='ssl_training',epoch=epoch,epochs=5,step_this_epoch=step,
                      steps_this_epoch=len(dl),global_step=global_step,total_steps=1120,
                      seen_this_epoch=len(seen),train_n=len(tr),
                      seconds_step=time.monotonic()-tick,elapsed_seconds=time.monotonic()-started,
                      peak_gib=torch.cuda.max_memory_allocated()/2**30,**values)
            history.append(item);save(run/'progress.json',item)
            if step==1 or step%5==0: print(json.dumps(item),flush=True)
            if step%50==0 or step==len(dl):
                tmp=run/'last.pt.tmp'
                torch.save(dict(encoder=enc.original.state_dict(),projector=proj.state_dict(),
                                optimizer=opt.state_dict(),epoch=epoch,step=global_step,
                                manifest_sha=digest(ROOT/'manifest.json'),
                                parent_checkpoint_sha=m['parent_checkpoint_sha'],
                                rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()),tmp)
                tmp.replace(run/'last.pt');save(run/'history.json',history)
        np.testing.assert_array_equal(np.array(seen),order)
        assert set(seen)==set(tr.tolist()) and global_step==epoch*224
        epoch_result=dict(epoch=epoch,unique_train_images=len(set(seen)),validation_images_used=0,
                          seconds=time.monotonic()-epoch_start,last=history[-1],
                          order_sha=hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest())
        epoch_results.append(epoch_result);save(run/'epochs.json',epoch_results)
        print(json.dumps(dict(epoch_complete=epoch,**epoch_result)),flush=True)
    assert global_step==1120 and len(history)==896 and digest(WEIGHT)==m['parent_checkpoint_sha']
    result=dict(status='SSL_TOTAL5_COMPLETE_NOT_ACCURACY_VALIDATED',epochs=5,additional_epochs=4,
                total_steps=1120,additional_steps=896,train_image_presentations=4*len(tr),
                unique_train_images=len(tr),validation_images_used=0,
                seconds=time.monotonic()-started,epochs_detail=epoch_results,details=details,
                checkpoint_sha=digest(run/'last.pt'),parent_checkpoint_preserved=True,
                manifest_sha=digest(ROOT/'manifest.json'),limitations=m['limitations'])
    save(run/'result.json',result);save(run/'complete.json',dict(status='completed',result_sha=digest(run/'result.json')))
    save(run/'progress.json',dict(status='completed',epochs=5,global_step=1120))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','tests','smoke','train'])
    args=p.parse_args()
    if args.command in ('smoke','train'): assert torch.cuda.device_count()==1
    globals()[args.command]()
