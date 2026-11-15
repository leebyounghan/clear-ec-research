"""Bounded full-native CellposeDINO LeJEPA pilot; research, not a submission.

Objective follows Balestriero & LeCun's official LeJEPA minimal example:
https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md
Our adaptations: existing CellposeDINO, full-native scale-preserving views,
pooled patch features, and exact feature-gradient caching across real images.
No teacher, stop-gradient objective, masks, regression targets, or validation
images are used during SSL. Detached cached features receive an exact chain-rule
backward replay; this is memory management, not a stop-gradient SSL objective.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from phase2_foundation_preflight_20261002 import build, digest, save

ROOT = Path('experiments/phase2_lejepa_20261003')
BASE = Path('experiments/phase2_foundation_train_20261002')
META = Path('datasets/clear_ec_official/cache_native_meta_512.npz')
RAW = Path('datasets/clear_ec_fullframe_20260904/img_full_raw.dat')
SPLIT = Path('experiments/overnight_target_20260912/preflight/fold3.npz')
WEIGHT = Path('experiments/phase2_foundation_20261002/weights/cellpose/cpdino-vitb')
EXPECTED_WEIGHT = '3ed4c06a3963ab13ff377d4e2957174aaaf637434eb031ae9931fcbfaf9a217f'
RECIPE = dict(seed=1003, epochs=1, images_per_batch=32, views=2, micro_views=4,
              encoder_lr=1e-5, projector_lr=1e-3, weight_decay=1e-4,
              lamb=0.02, projector_dim=128, sigreg_directions=256, knots=17,
              precision='encoder BF16; projector/loss FP32', gradient_clip=1.,
              warmup_steps=20, checkpoint_every=50,
              augmentation='independent H/V flips, gain[0.9,1.1], offset[-0.03,0.03]; no resize/crop/mask',
              pooling='mean of patch tokens, matching baseline',
              view='972x1296 whole native, bottom20/right16 replicate pad to992x1312',
              status='ONE_EPOCH_SSL_PILOT; no automatic extension or finetuning')


def read(path):
    return json.loads(Path(path).read_text())


def split():
    # Deliberately never load target Y. Train/validation membership is frozen.
    with np.load(META, allow_pickle=True) as z:
        ids = z['ids'].astype(str)
    with np.load(SPLIT) as z:
        tr, va = z['train'].copy(), z['validation'].copy()
        np.testing.assert_array_equal(va, z['primary_validation'])
    assert len(tr) == 7161 and len(va) == 1791
    assert not np.intersect1d(tr, va).size
    assert len(np.unique(tr)) == len(tr) and len(np.unique(ids)) == len(ids)
    return ids, tr, va


def id_sha(ids, indices):
    return hashlib.sha256('\n'.join(ids[indices]).encode()).hexdigest()


def prepare():
    if (ROOT/'manifest.json').exists():
        raise FileExistsError('Never replace a frozen experiment manifest')
    base = read(BASE/'manifest.json')
    for p in (META, SPLIT, WEIGHT):
        assert digest(p) == base['hashes'][str(p)], str(p)
    assert digest(WEIGHT) == EXPECTED_WEIGHT
    assert digest(RAW) == base['raw_sha256']
    ids, tr, va = split()
    assert id_sha(ids, va) == base['validation_ids_sha']
    inputs = [META, SPLIT, WEIGHT, BASE/'manifest.json',
              ROOT/'source/phase2_lejepa_20261003.py',
              ROOT/'source/phase2_foundation_preflight_20261002.py']
    result = dict(created_utc=time.strftime('%FT%TZ', time.gmtime()),
                  architecture='CellposeDINO-B (DINOv3 ViT-B, kernel16/stride8)',
                  initialization='external CellposeDINO checkpoint; not CLEAR-EC supervised last.pt',
                  recipe=RECIPE, train_n=len(tr), val_n=len(va), total_images=len(ids),
                  train_ids_sha=id_sha(ids,tr), validation_ids_sha=id_sha(ids,va),
                  hashes={str(p):digest(p) for p in inputs}, raw_sha256=base['raw_sha256'],
                  raw_bytes=RAW.stat().st_size, source_reference_commit='c293d291ca87cd4fddee9d3fffe4e914c7272052',
                  limitations=['one-epoch pilot is not an accuracy result',
                              'reused development fold3; patient independence unverified',
                              'full-frame two-view adaptation, not the original multi-crop recipe',
                              'gradient caching computes SIGReg across32 actual images, not32 accumulation steps',
                              'T4 feasibility is still unverified'],
                  official_submission_changed=False)
    save(ROOT/'manifest.json',result)
    print(json.dumps(result),flush=True)


def verify():
    m = read(ROOT/'manifest.json')
    assert m['recipe'] == RECIPE
    for p, h in m['hashes'].items():
        assert digest(p) == h, f'Frozen file changed: {p}'
    assert RAW.stat().st_size == m['raw_bytes']
    ids,tr,va = split()
    assert id_sha(ids,tr) == m['train_ids_sha']
    assert id_sha(ids,va) == m['validation_ids_sha']
    return m


class Views(Dataset):
    def __init__(self, indices, n, epoch):
        self.indices, self.n, self.epoch, self.raw = indices,n,epoch,None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, row):
        import cv2
        cv2.setNumThreads(1)
        if self.raw is None:
            self.raw = np.memmap(RAW,mode='r',dtype=np.uint8,shape=(self.n,972,1296))
        ix = int(self.indices[row])
        a = cv2.createCLAHE(2.,(1296//32,972//32)).apply(np.asarray(self.raw[ix]))
        lo,hi = np.percentile(a,(1,99))
        a = np.clip((a.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)
        views = []
        for view in range(2):
            rng = np.random.default_rng(np.random.SeedSequence([1003,self.epoch,ix,view]))
            b = a
            if rng.random()<.5: b = b[:,::-1]
            if rng.random()<.5: b = b[::-1,:]
            b = np.clip(b*rng.uniform(.9,1.1)+rng.uniform(-.03,.03),0,1)
            views.append(torch.from_numpy(np.ascontiguousarray(b[None])))
        return torch.stack(views),ix


def loader(indices,n,epoch,batch=32):
    return DataLoader(Views(indices,n,epoch),batch_size=batch,shuffle=False,
                      num_workers=2,pin_memory=True,prefetch_factor=1,drop_last=False)


def to_gpu(x):
    assert x.shape[1:] == (1,972,1296)
    return F.pad(x.cuda(non_blocking=True).expand(-1,3,-1,-1),(0,16,0,20),mode='replicate').contiguous()


class SIGReg(nn.Module):
    """Characteristic-function quadrature from the official LeJEPA example.

    Input [views, actual_images, projection_dim]. Statistics are over images,
    separately per view. No normalization or clamping of embeddings is applied.
    """
    def __init__(self):
        super().__init__()
        t = torch.linspace(0,3,17)
        w = torch.full((17,),2*3/16)
        w[0] = w[-1] = 3/16
        self.register_buffer('t',t)
        self.register_buffer('target',torch.exp(-t.square()/2))
        self.register_buffer('weights',w*torch.exp(-t.square()/2))

    def forward(self,z,directions=None):
        assert z.ndim == 3 and z.shape[1] >= 2
        if directions is None:
            directions = F.normalize(torch.randn(z.shape[-1],256,device=z.device),dim=0)
        phase = (z@directions).unsqueeze(-1)*self.t
        err = (phase.cos().mean(1)-self.target).square()+phase.sin().mean(1).square()
        return ((err@self.weights)*z.shape[1]).mean()


def objective(flat,projector,sigreg):
    z = projector(flat.float()).reshape(-1,2,128).transpose(0,1)
    inv = (z-z.mean(0,keepdim=True)).square().mean()
    reg = sigreg(z)
    loss = (1-RECIPE['lamb'])*inv+RECIPE['lamb']*reg
    return loss,z,inv,reg


def rng_state():
    return torch.get_rng_state(),torch.cuda.get_rng_state() if torch.cuda.is_available() else None


def restore_rng(state):
    torch.set_rng_state(state[0])
    if state[1] is not None: torch.cuda.set_rng_state(state[1])


def cache_backward(encoder,projector,sigreg,x,micro,forward,check_replay=False):
    """Compute the global-batch objective, then replay encoder vector-Jacobians.

    Encoder must have no mutable batch statistics. RNG is replayed exactly for
    dropout/RoPE jitter, and restored afterwards. Projector BN sees ALL images.
    No optimizer update occurs between cache and replay.
    """
    if any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in encoder.modules()):
        raise ValueError('Gradient-cache encoder must not mutate BN statistics')
    features,states = [],[]
    with torch.no_grad():
        for part in x.split(micro):
            states.append(rng_state())
            features.append(forward(encoder,part).float())
    leaf = torch.cat(features).detach().requires_grad_(True)
    loss,z,inv,reg = objective(leaf,projector,sigreg)
    if not torch.isfinite(loss): raise ValueError('Nonfinite SSL objective')
    loss.backward()
    gradients = leaf.grad.detach()
    after = rng_state()
    max_error = 0.
    for i,part in enumerate(x.split(micro)):
        restore_rng(states[i])
        emb = forward(encoder,part).float()
        if check_replay:
            error = (emb.detach()-features[i]).abs().max().item()
            max_error = max(error,max_error)
            if not torch.allclose(emb.detach(),features[i],atol=2e-3,rtol=2e-3):
                raise ValueError(f'Gradient cache replay drift {error}; stop, never approximate silently')
        torch.autograd.backward(emb,gradients[i*micro:i*micro+len(part)])
    restore_rng(after)
    with torch.no_grad():
        image_features = leaf.reshape(-1,2,leaf.shape[-1]).mean(1)
        sd = image_features.std(0).mean().item()
        proj_sd = z.std(dim=1).mean().item()
        singular = torch.linalg.svdvals(image_features-image_features.mean(0))
        q = singular.square()/singular.square().sum().clamp_min(1e-12)
        rank = (-torch.where(q>0,q*q.clamp_min(1e-12).log(),0).sum()).exp().item()
    return dict(loss=loss.item(),invariance=inv.item(),sigreg=reg.item(),
                feature_std=sd,projector_std=proj_sd,effective_rank=rank,
                replay_max_abs_error=max_error,actual_images=x.shape[0]//2)


def setup():
    torch.set_num_threads(2)
    torch.manual_seed(1003); np.random.seed(1003); torch.cuda.manual_seed_all(1003)
    torch.backends.cuda.matmul.allow_tf32=True
    model,_,details=build('cpdino_vitb',False)
    encoder=model[0].cuda().train()
    del model
    assert details['patch_stride']==[8,8] and details['patch_kernel']==[16,16]
    projector=nn.Sequential(nn.Linear(768,2048),nn.BatchNorm1d(2048),nn.ReLU(),
                            nn.Linear(2048,2048),nn.BatchNorm1d(2048),nn.ReLU(),
                            nn.Linear(2048,128)).cuda().train()
    sigreg=SIGReg().cuda()
    opt=torch.optim.AdamW([{'params':encoder.parameters(),'lr':RECIPE['encoder_lr']},
                          {'params':projector.parameters(),'lr':RECIPE['projector_lr']}],
                         weight_decay=RECIPE['weight_decay'],fused=True)
    details.update(gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
                   encoder_parameters=sum(p.numel() for p in encoder.parameters()),
                   train_only_projector_parameters=sum(p.numel() for p in projector.parameters()))
    return encoder,projector,sigreg,opt,details


def gpu_forward(encoder,x):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        return encoder(to_gpu(x))


def update(encoder,projector,sigreg,opt,x,check=False):
    opt.zero_grad(set_to_none=True)
    values=cache_backward(encoder,projector,sigreg,x.flatten(0,1),4,gpu_forward,check)
    allp=list(encoder.parameters())+list(projector.parameters())
    norm=nn.utils.clip_grad_norm_(allp,1.,error_if_nonfinite=True)
    if norm.item() <= 0: raise ValueError('Zero gradient')
    values['gradient_norm']=norm.item()
    opt.step()
    return values


def smoke():
    verify(); ids,tr,va=split()
    if (ROOT/'smoke.json').exists(): raise FileExistsError('smoke already exists')
    enc,proj,reg,opt,details=setup()
    x,ix=next(iter(loader(tr[:8],len(ids),0,batch=8)))
    assert not np.intersect1d(ix.numpy(),va).size
    tracked={n:p for n,p in enc.named_parameters() if p.ndim>=2}
    before={n:p.detach().clone() for n,p in tracked.items()}
    started=time.monotonic(); torch.cuda.reset_peak_memory_stats()
    v=update(enc,proj,reg,opt,x,True)
    changed=[n for n,p in tracked.items() if not torch.equal(before[n],p.detach())]
    blocks={re.search(r'blocks\.\d+',n).group() for n in tracked if re.search(r'blocks\.\d+',n)}
    changed_blocks={re.search(r'blocks\.\d+',n).group() for n in changed if re.search(r'blocks\.\d+',n)}
    if blocks != changed_blocks or not any('patch_embed' in n for n in changed):
        raise ValueError('Encoder blocks or patch projection did not update')
    torch.cuda.synchronize()
    result=dict(status='passed',manifest_sha=digest(ROOT/'manifest.json'),
                details=details,values=v,changed_blocks=sorted(changed_blocks),
                seconds=time.monotonic()-started,peak_gib=torch.cuda.max_memory_allocated()/2**30,
                input_shape=[972,1296],cropped_or_resized=False,validation_images_used=0)
    save(ROOT/'smoke.json',result); print(json.dumps(result),flush=True)


def train():
    m=verify(); ids,tr,va=split(); output=ROOT/'pilot'
    if output.exists(): raise FileExistsError('No implicit resume or overwrite')
    sm=read(ROOT/'smoke.json')
    assert sm['status']=='passed' and sm['manifest_sha']==digest(ROOT/'manifest.json')
    output.mkdir()
    enc,proj,reg,opt,details=setup()
    order=np.random.default_rng(1003).permutation(tr)
    dl=loader(order,len(ids),0)
    history=[]; seen=[]; started=time.monotonic(); reference_std=None
    save(output/'progress.json',dict(status='starting',steps=len(dl),details=details))
    for step,(x,ix) in enumerate(dl,1):
        assert not np.intersect1d(ix.numpy(),va).size
        ratio=min(1.,step/RECIPE['warmup_steps'])
        opt.param_groups[0]['lr']=RECIPE['encoder_lr']*ratio
        opt.param_groups[1]['lr']=RECIPE['projector_lr']*ratio
        tick=time.monotonic(); torch.cuda.reset_peak_memory_stats()
        v=update(enc,proj,reg,opt,x,step==1)
        torch.cuda.synchronize()
        if reference_std is None: reference_std=v['feature_std']
        if v['feature_std']<max(1e-6,reference_std*1e-3) or v['projector_std']<1e-6:
            raise ValueError('Representation collapse tripwire; no automatic continuation')
        seen.extend(ix.tolist())
        item=dict(status='ssl_pilot_training',step=step,steps=len(dl),seen=len(seen),train_n=len(tr),
                  seconds_step=time.monotonic()-tick,elapsed_seconds=time.monotonic()-started,
                  peak_gib=torch.cuda.max_memory_allocated()/2**30,**v)
        history.append(item)
        save(output/'progress.json',item)
        if step==1 or step%5==0:
            print(json.dumps(item),flush=True)
        if step%50==0 or step==len(dl):
            # Canonical encoder keys can be loaded directly into the original
            # CPDINO encoder. Projector and optimizer are research-only extras.
            tmp=output/'last.pt.tmp'
            torch.save(dict(encoder=enc.original.state_dict(),projector=proj.state_dict(),
                            optimizer=opt.state_dict(),step=step,manifest_sha=digest(ROOT/'manifest.json'),
                            rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()),tmp)
            tmp.replace(output/'last.pt')
            save(output/'history.json',history)
    np.testing.assert_array_equal(np.array(seen),order)
    assert set(seen)==set(tr.tolist())
    result=dict(status='SSL_PILOT_COMPLETE_NOT_ACCURACY_VALIDATED',steps=len(history),
                train_images_seen=len(seen),unique_train_images=len(set(seen)),validation_images_used=0,
                seen_order_sha=hashlib.sha256(np.array(seen,dtype=np.int64).tobytes()).hexdigest(),
                seconds=time.monotonic()-started,details=details,last=history[-1],
                checkpoint_sha=digest(output/'last.pt'),manifest_sha=digest(ROOT/'manifest.json'),
                limitations=m['limitations'],next='inspect pilot, then choose bounded SSL extension and matched regression evaluation')
    save(output/'result.json',result)
    save(output/'complete.json',dict(status='completed',result_sha=digest(output/'result.json')))
    save(output/'progress.json',dict(status='completed',step=len(history),train_n=len(tr)))
    print(json.dumps(result),flush=True)


def tests():
    # Compare exact chain-rule caching against retaining ALL encoder graphs.
    # Dropout verifies RNG replay; BN is in the full-batch projector only.
    torch.set_num_threads(2); torch.manual_seed(9)
    enc=nn.Sequential(nn.Linear(7,13),nn.Dropout(.2),nn.GELU(),nn.Linear(13,768))
    proj=nn.Sequential(nn.Linear(768,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Linear(256,128))
    enc2,proj2=copy.deepcopy(enc),copy.deepcopy(proj)
    x=torch.randn(18,7); reg=SIGReg(); initial=rng_state()
    emb=torch.cat([enc(p) for p in x.split(4)])
    loss,_,_,_=objective(emb,proj,reg); loss.backward()
    restore_rng(initial)
    values=cache_backward(enc2,proj2,reg,x,4,lambda e,a:e(a),True)
    assert abs(loss.item()-values['loss'])<1e-6
    max_error=0.
    for a,b in zip(list(enc.parameters())+list(proj.parameters()),list(enc2.parameters())+list(proj2.parameters())):
        torch.testing.assert_close(a.grad,b.grad,rtol=2e-5,atol=2e-6)
        max_error=max(max_error,(a.grad-b.grad).abs().max().item())
    z=torch.randn(2,9,128,requires_grad=True)
    directions=F.normalize(torch.randn(128,256),dim=0)
    manual=[]
    for view in z:
        phase=(view@directions).unsqueeze(-1)*reg.t
        error=(phase.cos().mean(0)-reg.target).square()+phase.sin().mean(0).square()
        manual.append((error@reg.weights*len(view)).mean())
    torch.testing.assert_close(reg(z,directions),torch.stack(manual).mean())
    print(json.dumps(dict(status='PASS',gradient_cache_max_abs_error=max_error,
                         tests=['exact encoder/projector gradients','RNG replay with dropout',
                                'full-batch projector BN','SIGReg image/view axes','partial microbatch'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['prepare','tests','smoke','train'])
    a=p.parse_args()
    if a.command in ('smoke','train') and torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one H200 visible GPU required')
    globals()[a.command]()
