"""Native-frame single-backbone screen. Research only, never an official score.

Fold3 is deliberately used first: all three raw targets are positive in both
partitions. Zero-target inputs fail closed, never silently dropped/clamped.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from phase2_foundation_preflight_20261002 import build, digest, save

ROOT = Path('experiments/phase2_foundation_train_20261002')
ARCHS = ('cpdino_vitb', 'dinov3_vitb', 'convnext_base')
META = Path('datasets/clear_ec_official/cache_native_meta_512.npz')
RAW = Path('datasets/clear_ec_fullframe_20260904/img_full_raw.dat')
SPLIT = Path('experiments/overnight_target_20260912/preflight/fold3.npz')
AUDIT = Path('experiments/phase2_foundation_optimization_20261002/data_audit.json')
WEIGHTS = {
    'cpdino_vitb': Path('experiments/phase2_foundation_20261002/weights/cellpose/cpdino-vitb'),
    'dinov3_vitb': Path('experiments/phase2_foundation_20261002/weights/dinov3-vitb16/model.safetensors'),
    'convnext_base': Path('experiments/phase2_backbone_screen_20261001/hf_cache/hub/models--timm--convnext_base.fb_in22k_ft_in1k/snapshots/67758fcbcf9f007ec7630bf8636b74a8f24e52a0/model.safetensors'),
}


def read(path):
    return json.loads(Path(path).read_text())


def arrays():
    with np.load(META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    with np.load(SPLIT) as z:
        train, val = z['train'].copy(), z['validation'].copy()
        np.testing.assert_array_equal(val, z['primary_validation'])
    if not np.isfinite(y[np.r_[train, val]]).all() or not (y[np.r_[train, val]] > 0).all():
        raise ValueError('This positive-target-only protocol cannot score zeros; do not filter them')
    assert len(train) == 7161 and len(val) == 1791 and not np.intersect1d(train, val).size
    return ids, y, train, val


def prepare():
    target = ROOT/'manifest.json'
    if target.exists():
        raise FileExistsError(target)
    audit = read(AUDIT)
    if audit['errors'] or audit['failed_checks'] or not audit['pixels']['verified']:
        raise ValueError('Data integrity audit failed')
    ids, y, train, val = arrays()
    raw_sha = digest(RAW)
    if raw_sha != audit['inputs'][str(RAW.resolve())]['sha256']:
        raise ValueError('Full-frame cache changed since audit')
    files = [META, SPLIT, AUDIT, ROOT/'source/phase2_foundation_train_20261002.py',
             ROOT/'source/phase2_foundation_preflight_20261002.py', *WEIGHTS.values()]
    log_y = np.log(y[train].astype(np.float64))
    if not np.isfinite(log_y.std(0)).all() or not (log_y.std(0)>0).all():
        raise ValueError('Invalid training log target standard deviation')
    recipe = dict(epochs=20, seed=42, micro_batch=4, effective_batch=16,
                  encoder_lr=1e-5, head_lr=1e-3, weight_decay=1e-4,
                  warmup_epochs=1, minimum_lr_ratio=0.1,
                  objective='0.5 standardized log L1 + 0.5 physical MAPE / 0.2',
                  gradient_clip=1.0, precision='bfloat16', checkpointing=False,
                  head='mean pool -> LayerNorm -> Linear128 -> GELU -> Linear3',
                  augmentation='independent deterministic H/V flips only',
                  view='whole native972x1296; replicate-pad bottom20/right16 to992x1312 for ALL architectures',
                  normalization='common CLAHE+percentile01; fixed ImageNet normalization for DINOv3/ConvNeXt, Cellpose01 for CPDINO',
                  checkpoint_policy='fixed terminal epoch20 primary; epoch curves diagnostic only',
                  validation='every epoch, one whole-frame view, no TTA or calibration',
                  labels='original unsmoothed labels; train-only log mean/std',
                  metric='mean of three per-target MAPE percentages on ALL 1791 fold3 images; no exclusions')
    manifest = dict(created_utc=time.strftime('%FT%TZ', time.gmtime()), architectures=list(ARCHS),
                    fold=3, train_n=len(train), validation_n=len(val), images=len(ids),
                    recipe=recipe, hashes={str(p): digest(p) for p in files},
                    raw_sha256=raw_sha, raw_bytes=RAW.stat().st_size,
                    train_order_base_sha=hashlib.sha256(train.tobytes()).hexdigest(),
                    validation_ids_sha=hashlib.sha256('\n'.join(ids[val]).encode()).hexdigest(),
                    log_mean=log_y.mean(0).tolist(), log_std=log_y.std(0).tolist(),
                    torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    official_release_untouched=True, gpu_parallelism=2,
                    limits=['reused development fold; not independent Phase II score',
                            'architecture, external pretraining, token stride and required normalization differ',
                            'patient independence and near-duplicate leakage unverified',
                            '20 epochs is screening, not proof of asymptotic backbone ranking',
                            'T4 deployment not yet verified; H200 BF16 is not T4 certification',
                            'fold2 confirmation requires explicit official zero-target policy'])
    save(target, manifest)
    print(json.dumps(manifest), flush=True)


def verify():
    m = read(ROOT/'manifest.json')
    for p, expected in m['hashes'].items():
        if digest(p) != expected:
            raise ValueError(f'Frozen input/source checksum mismatch: {p}')
    if RAW.stat().st_size != m['raw_bytes']:
        raise ValueError('Raw cache size changed')
    arrays()
    return m


class Frames(Dataset):
    def __init__(self, indices, targets, total, epoch=0, augment=False):
        self.indices, self.targets, self.total = indices, targets, total
        self.epoch, self.augment, self.raw = epoch, augment, None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, row):
        import cv2
        cv2.setNumThreads(1)
        if self.raw is None:
            self.raw = np.memmap(RAW, mode='r', dtype=np.uint8, shape=(self.total,972,1296))
        idx = int(self.indices[row])
        a = cv2.createCLAHE(2.0,(1296//32,972//32)).apply(np.asarray(self.raw[idx]))
        lo, hi = np.percentile(a,(1,99))
        a = np.clip((a.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)
        if self.augment:
            rng = np.random.default_rng(np.random.SeedSequence([42,self.epoch,idx]))
            if rng.random() < .5:
                a = a[:,::-1]
            if rng.random() < .5:
                a = a[::-1,:]
        return torch.from_numpy(np.ascontiguousarray(a[None])), torch.from_numpy(self.targets[idx]), idx


def loader(indices, y, n, epoch=0, augment=False):
    return DataLoader(Frames(indices,y,n,epoch,augment), batch_size=4, shuffle=False,
                      num_workers=4, pin_memory=True, prefetch_factor=2, drop_last=False)


def inputs(x, architecture):
    x = x.cuda(non_blocking=True).expand(-1,3,-1,-1)
    x = F.pad(x,(0,16,0,20),mode='replicate')
    if architecture in ('dinov3_vitb','convnext_base'):
        x = (x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
    return x.contiguous(memory_format=torch.channels_last if architecture=='convnext_base' else torch.contiguous_format)


def metrics(pred, y):
    if pred.shape != y.shape or not np.isfinite(pred).all() or not np.isfinite(y).all() or not (y>0).all():
        raise ValueError('Invalid predictions/targets: no zero exclusions allowed')
    v = (np.abs(pred-y)/y).mean(0)*100
    return dict(CD=float(v[0]),CV=float(v[1]),HEX=float(v[2]),mean=float(v.mean()),n=len(y))


def loss_fn(z, targets, mu, sigma):
    zt = (targets.log()-mu)/sigma
    return 0.5*(z.float()-zt).abs().mean()+2.5*torch.expm1((z.float()-zt)*sigma).abs().mean()


def seed():
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def setup(arch,m):
    seed()
    net, multiple, details = build(arch,False,native_convnext_norm=True)
    assert 992%multiple==0 and 1312%multiple==0
    torch.manual_seed(42)
    nn.init.normal_(net[-1].weight,std=1e-3); nn.init.zeros_(net[-1].bias)
    net = net.cuda()
    if arch=='convnext_base':
        net = net.to(memory_format=torch.channels_last)
    details['common_padded_input']=[992,1312]
    details['patch_or_feature_grid']=([124,164] if arch=='cpdino_vitb' else
                                      [62,82] if arch=='dinov3_vitb' else [31,41])
    r=m['recipe']
    opt = torch.optim.AdamW([{'params':net[0].parameters(),'lr':r['encoder_lr']},
                            {'params':list(net[1:].parameters()),'lr':r['head_lr']}],
                           weight_decay=r['weight_decay'],fused=True)
    mu=torch.tensor(m['log_mean'],device='cuda',dtype=torch.float32)
    sigma=torch.tensor(m['log_std'],device='cuda',dtype=torch.float32)
    if not torch.isfinite(sigma).all() or not (sigma>0).all():
        raise ValueError('Invalid target standard deviation')
    details.update(gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
                   cuda_version=torch.version.cuda,bitwise_determinism_guaranteed=False)
    return net,opt,mu,sigma,details


def finite_grad(net):
    values=[torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None]
    if not values or not torch.stack(values).all().item():
        raise ValueError('Missing/nonfinite gradients')


def smoke(arch):
    m=verify(); ids,y,tr,va=arrays(); net,opt,mu,sigma,details=setup(arch,m)
    net.train(); checks=[]
    for x,t,ix in loader(tr[:8],y,len(ids)):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=net(inputs(x,arch))
            loss=loss_fn(z,t.cuda(),mu,sigma)
        loss.backward(); finite_grad(net)
        tracked=[]
        for name,p in net[0].named_parameters():
            if p.grad is not None and p.ndim>=2 and p.grad.abs().max().item()>0:
                tracked.append((name,p,p.detach().flatten()[:256].clone()))
        if not tracked:
            raise ValueError('No encoder gradient: frozen-feature training would invalidate comparison')
        def block_key(name):
            match=re.search(r'stages\.\d+\.blocks\.\d+|blocks\.\d+|layer\.\d+',name)
            return match.group(0) if match else None
        expected_blocks={block_key(n) for n,p in net[0].named_parameters() if p.requires_grad and block_key(n)}
        grad_blocks={block_key(n) for n,p,b in tracked if block_key(n)}
        if not expected_blocks or expected_blocks!=grad_blocks:
            raise ValueError(f'Disconnected encoder blocks: {expected_blocks-grad_blocks}')
        if not any(('patch' in n or 'stem' in n) for n,p,b in tracked):
            raise ValueError('No gradient reaches patch/stem projection')
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0,error_if_nonfinite=True)
        opt.step()
        changed=[name for name,p,before in tracked if not torch.equal(p.detach().flatten()[:256],before)]
        if not changed:
            raise ValueError('Encoder weights did not update')
        if {block_key(n) for n in changed if block_key(n)}!=expected_blocks:
            raise ValueError('An encoder block has gradients but no measured weight update')
        checks.append(dict(loss=loss.item(),encoder_grad_tensors=len(tracked),changed_tensors=changed,
                           connected_blocks=sorted(expected_blocks)))
    net.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        out=net(inputs(x,arch)).float()
    if not torch.isfinite(out).all():
        raise ValueError('Nonfinite smoke eval')
    save(ROOT/'smoke'/f'{arch}.json',dict(status='passed',checks=checks,details=details,
         inputs=[4,3,992,1312],training_only=True,accuracy_claim=False,
         manifest_sha=digest(ROOT/'manifest.json')))
    print(json.dumps({'smoke':arch,'status':'passed','encoder_updated_tensor_counts':[len(c['changed_tensors']) for c in checks]}),flush=True)


def evaluate(net,arch,dl,mu,sigma,expected):
    net.eval(); predictions=[]; indices=[]
    with torch.inference_mode():
        for x,t,ix in dl:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=net(inputs(x,arch)).float()
            predictions.append(torch.exp(z*sigma+mu).cpu().numpy())
            indices.append(ix.numpy())
    np.testing.assert_array_equal(np.concatenate(indices),expected)
    return np.concatenate(predictions)


def train(arch):
    m=verify(); r=m['recipe']; path=ROOT/'runs'/f'fold3/{arch}'
    if path.exists():
        raise FileExistsError(f'Never implicitly resume/overwrite: {path}')
    smoke_result=read(ROOT/'smoke'/f'{arch}.json')
    assert smoke_result['status']=='passed' and smoke_result['manifest_sha']==digest(ROOT/'manifest.json')
    path.mkdir(parents=True)
    ids,y,tr,va=arrays(); net,opt,mu,sigma,details=setup(arch,m)
    validation=loader(va,y,len(ids)); updates=math.ceil(len(tr)/16)
    history=[]; global_step=0; wall0=time.monotonic()
    save(path/'progress.json',dict(status='starting',architecture=arch,details=details))
    for epoch in range(r['epochs']):
        order=np.random.default_rng(42+epoch).permutation(tr)
        dl=loader(order,y,len(ids),epoch,True)
        order_sha=hashlib.sha256(order.tobytes()).hexdigest()
        net.train(); opt.zero_grad(set_to_none=True); total_loss=0.; seen=0
        torch.cuda.reset_peak_memory_stats(); started=time.monotonic()
        for micro,(x,t,ix) in enumerate(dl):
            group_start=(micro//4)*16; group_n=min(16,len(tr)-group_start)
            if micro%4==0:
                if global_step<updates:
                    ratio=(global_step+1)/updates
                else:
                    phase=(global_step-updates)/max(1,updates*(r['epochs']-1)-1)
                    ratio=.1+.9*.5*(1+math.cos(math.pi*phase))
                opt.param_groups[0]['lr']=r['encoder_lr']*ratio
                opt.param_groups[1]['lr']=r['head_lr']*ratio
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=net(inputs(x,arch))
                loss=loss_fn(z,t.cuda(non_blocking=True),mu,sigma)
            if not torch.isfinite(loss).item():
                raise ValueError('Nonfinite objective')
            (loss*(len(t)/group_n)).backward()
            total_loss+=loss.item()*len(t); seen+=len(t)
            if (micro+1)%4==0 or micro+1==len(dl):
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.0,error_if_nonfinite=True)
                opt.step(); opt.zero_grad(set_to_none=True); global_step+=1
                if global_step%25==0:
                    progress=dict(status='training',epoch=epoch+1,epochs=r['epochs'],seen=seen,train_n=len(tr),
                                  updates=global_step,loss=total_loss/seen,seconds_this_epoch=time.monotonic()-started,
                                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                                  architecture=arch,order_sha=order_sha)
                    save(path/'progress.json',progress); print(json.dumps(progress),flush=True)
        assert seen==len(tr) and global_step==(epoch+1)*updates
        torch.cuda.synchronize(); train_seconds=time.monotonic()-started
        pred=evaluate(net,arch,validation,mu,sigma,va)
        score=metrics(pred,y[va]); elapsed=time.monotonic()-started
        item=dict(epoch=epoch+1,score=score,train_loss=total_loss/seen,train_seconds=train_seconds,
                  epoch_seconds=elapsed,train_images_per_second=seen/train_seconds,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  order_sha=order_sha,updates=global_step)
        history.append(item); save(path/'history.json',history)
        # Keep one recoverable epoch checkpoint, never silently auto-resume it.
        tmp=path/'last.pt.tmp'
        torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,
                        architecture=arch,manifest_sha=digest(ROOT/'manifest.json'),
                        log_mean=m['log_mean'],log_std=m['log_std'],history=history,
                        rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()),tmp)
        tmp.replace(path/'last.pt')
        save(path/'progress.json',dict(status='epoch_complete',**item,
                                      remaining_hours_estimate=elapsed*(r['epochs']-epoch-1)/3600))
        print(json.dumps({'epoch_complete':arch,**item}),flush=True)
    np.savez_compressed(path/'predictions.npz',indices=va,ids=ids[va],target=y[va],pred=pred)
    result=dict(architecture=arch,fold=3,score=score,history=history,details=details,
                seconds=time.monotonic()-wall0,manifest_sha=digest(ROOT/'manifest.json'),
                checkpoint_sha=digest(path/'last.pt'),predictions_sha=digest(path/'predictions.npz'),
                limits=m['limits'],official_submission_changed=False)
    save(path/'result.json',result)
    save(path/'complete.json',dict(status='completed',result_sha=digest(path/'result.json')))


def tests():
    yy=np.array([[100.,.4,.6],[200.,.5,.7]])
    assert abs(metrics(yy*1.1,yy)['mean']-10)<1e-10
    try:
        metrics(yy,np.zeros_like(yy))
    except ValueError:
        pass
    else:
        raise AssertionError('Zero labels must not be silently removed')
    y=torch.tensor(yy); mu=y.log().mean(0); sd=y.log().std(0)
    z=(y.log()-mu)/sd
    assert loss_fn(z,y,mu,sd).item()<1e-6
    for n in (1,4,15,16,17,7161):
        weights=[]
        for start in range(0,n,16):
            group=min(16,n-start)
            weights.append(sum(min(4,group-j)/group for j in range(0,group,4)))
        assert all(abs(v-1)<1e-12 for v in weights)
    print('PASS: metrics, zero rejection, loss identity, partial accumulation',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['prepare','smoke','train','tests'])
    p.add_argument('--architecture',choices=ARCHS); a=p.parse_args()
    if a.command=='prepare': prepare()
    elif a.command=='tests': tests()
    else:
        if not a.architecture: p.error('architecture required')
        if torch.cuda.device_count()!=1: raise RuntimeError('Exactly one visible GPU required')
        smoke(a.architecture) if a.command=='smoke' else train(a.architecture)
