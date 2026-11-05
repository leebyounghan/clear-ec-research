"""Fold-clean LeJEPA5 regression, one independent single-model arm per root.

Full-frame CPDINO, positive targets, 2 GPUs/effective batch16. Each terminal
checkpoint is scored separately; no ensembles, TTA, automatic resume or implicit
overwrite. The terminal epoch is primary; intermediate minima are diagnostics.
"""
import phase2_pooling_20261019 as pooling
import argparse
from contextlib import contextmanager, nullcontext
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import phase2_foundation_train_20261002 as b
import phase2_lejepa_regression_20261003 as prior
import phase2_foundation_preflight_20261002 as preflight
from phase2_foundation_preflight_20261002 import digest, save

ARCH = 'cpdino_vitb'
ROOT = SPLIT = SSL = None
ARGS = None
UTC = datetime.timezone.utc
EXPECTED_REUSED_SOURCES = {
    'phase2_foundation_train_20261002.py':'20a849cb607c3c5777102e3266c6c1b5d9ba4533f9bfac9f1f6e9d18232dac2f',
    'phase2_foundation_preflight_20261002.py':'196500b67d01ad9423ea4d7d8c1a1e4b2d886df2328d36bfb1e1c94058ca4b1e',
    'phase2_lejepa_regression_20261003.py':'f66fe903c2360c6e2392b407c55392dc1d7a45a03e624bed0acee8cbc32b33b8',
}


def read(path):
    return json.loads(Path(path).read_text())


def ids_sha(ids):
    return hashlib.sha256('\n'.join(ids.astype(str)).encode()).hexdigest()


def array_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def utc_now():
    return datetime.datetime.now(UTC)


def parse_deadline(value):
    if value == 'none':
        return datetime.datetime.max.replace(tzinfo=UTC)
    parsed = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('deadline-utc must have an explicit timezone')
    return parsed.astimezone(UTC)


def ensure_before_deadline(value):
    if value == 'none':
        return
    if utc_now() >= parse_deadline(value):
        raise RuntimeError('Training deadline has already passed')


def arrays():
    with np.load(b.META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    with np.load(SPLIT) as z:
        tr, va = z['train'].copy(), z['validation'].copy()
        if 'primary_validation' in z:
            historical = z['primary_validation']
            assert len(np.unique(historical)) == len(historical)
            assert np.isin(historical,va).all(), 'Historical subset must be in full validation'
    assert ids.ndim == 1 and len(np.unique(ids)) == len(ids)
    assert y.shape == (len(ids), 3)
    for name, ix in (('train', tr), ('validation', va)):
        assert ix.ndim == 1 and np.issubdtype(ix.dtype, np.integer), name
        assert len(ix) > 0 and len(np.unique(ix)) == len(ix), name
        assert ix.min() >= 0 and ix.max() < len(ids), name
    assert not np.intersect1d(tr, va).size, 'Train/validation overlap'
    selected = np.r_[tr, va]
    if not np.isfinite(y[selected]).all() or not (y[selected] > 0).all():
        raise ValueError('Positive-target protocol rejects zeros; no exclusions or clamping')
    return ids, y, tr, va


def historical_validation():
    with np.load(SPLIT) as z:
        return z['primary_validation'].copy() if 'primary_validation' in z else z['validation'].copy()


def secondary_score(pred,y,va):
    historical=historical_validation()
    positions={int(ix):i for i,ix in enumerate(va)}
    take=np.array([positions[int(ix)] for ix in historical],dtype=np.int64)
    return b.metrics(pred[take],y[historical])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def build_cpu(seed):
    seed_everything(seed)
    net, multiple, details = b.build(ARCH, False, native_convnext_norm=True)
    assert 992 % multiple == 0 and 1312 % multiple == 0
    net[0] = pooling.SpatialPoolEncoder(net[0].original, ARGS.pooling)
    dim = 768 * len(pooling.MODES[ARGS.pooling])
    if dim != 768:
        torch.manual_seed(seed)
        net[1] = torch.nn.LayerNorm(dim)
        net[2] = torch.nn.Linear(dim, 128)
    details['pooling'] = ARGS.pooling
    details['pooled_dimension'] = dim
    torch.manual_seed(seed)
    torch.nn.init.normal_(net[-1].weight, std=1e-3)
    torch.nn.init.zeros_(net[-1].bias)
    return net, details


def source_files():
    return [Path(__file__).resolve(),Path(pooling.__file__).resolve(),Path(b.__file__).resolve(),
            Path(prior.__file__).resolve(),Path(preflight.__file__).resolve()]


def freeze_sources():
    directory=ROOT/'source'
    directory.mkdir(parents=True,exist_ok=True)
    frozen=[]
    for source in source_files():
        if source.name in EXPECTED_REUSED_SOURCES:
            assert digest(source)==EXPECTED_REUSED_SOURCES[source.name],source
        destination=directory/source.name
        if destination.exists():
            assert digest(destination)==digest(source), 'Frozen source differs'
        else:
            with source.open('rb') as src,destination.open('xb') as dst:
                shutil.copyfileobj(src,dst)
        frozen.append(destination)
    return frozen


def ssl_evidence(ids, tr, va):
    sm = read(SSL/'manifest.json')
    for path, expected in sm['hashes'].items():
        assert digest(path) == expected, path
    assert sm['fold'] == ARGS.fold
    assert sm['split_sha256'] == digest(SPLIT)
    assert sm['train_ids_sha'] == ids_sha(ids[tr])
    assert sm['validation_ids_sha'] == ids_sha(ids[va])
    assert sm['train_n'] == len(tr) and sm['val_n'] == len(va)
    assert sm['total_images'] == len(ids)
    sr = prior.validate_result(SSL/'run', digest(SSL/'manifest.json'))
    assert sr['validation_images_used'] == 0, 'SSL must use train-only images'
    assert sr['epochs'] == 5
    assert len(sr['epochs_detail']) == 5
    for epoch,item in enumerate(sr['epochs_detail'],1):
        assert item['epoch']==epoch and item['validation_images_used']==0
        assert item['unique_train_images'] == len(tr)
        assert item['train_images_seen']==len(tr)
    checkpoint = torch.load(SSL/'run/last.pt', map_location='cpu', weights_only=True)
    assert checkpoint['epoch'] == 5
    assert checkpoint['manifest_sha'] == digest(SSL/'manifest.json')
    assert checkpoint['step'] == sr['steps']
    encoder_sha = prior.state_sha(checkpoint['encoder'])
    del checkpoint
    return sm, sr, encoder_sha


def prepare():
    ensure_before_deadline(ARGS.deadline_utc)
    if (ROOT/'manifest.json').exists():
        raise FileExistsError('Frozen manifest already exists')
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT/'runs').exists(), 'Existing output cannot be reused'
    required_gib = max(12, 0.65*ARGS.epochs + 4)
    assert shutil.disk_usage(ROOT).free > required_gib*2**30, 'Insufficient disk for epoch snapshots'
    ids, y, tr, va = arrays()
    audit = read(b.AUDIT)
    assert not audit['errors'] and not audit['failed_checks'] and audit['pixels']['verified']
    raw_sha = digest(b.RAW)
    assert raw_sha == audit['inputs'][str(b.RAW.resolve())]['sha256']
    sm, sr, encoder_sha = ssl_evidence(ids, tr, va)
    assert sm['raw_sha256'] == raw_sha and sm['raw_bytes'] == b.RAW.stat().st_size
    log_y = np.log(y[tr].astype(np.float64))
    assert np.isfinite(log_y.std(0)).all() and (log_y.std(0) > 0).all()
    net, _ = build_cpu(ARGS.seed)
    head_sha = prior.state_sha(net[1:].state_dict())
    del net
    files = [b.META, SPLIT, b.AUDIT, b.WEIGHTS[ARCH], *source_files(), *freeze_sources(),
             SSL/'manifest.json', SSL/'run/result.json', SSL/'run/complete.json', SSL/'run/last.pt']
    recipe = dict(epochs=ARGS.epochs, schedule_epochs=ARGS.schedule_epochs, seed=ARGS.seed,
                  micro_batch=4, effective_batch=16, encoder_lr=1e-5,
                  head_lr=ARGS.head_lr, gradient_clip=ARGS.clip, weight_decay=1e-4,
                  warmup_epochs=1, minimum_lr_ratio=0.1,
                  objective='0.5 standardized log L1 + 0.5 physical MAPE / 0.2',
                  precision='bfloat16', checkpointing=False,
                  head=f'{ARGS.pooling} pool -> LayerNorm -> Linear128 -> GELU -> Linear3',
                  pooling=ARGS.pooling, pooling_statistics=list(pooling.MODES[ARGS.pooling]),
                  pooled_dimension=768*len(pooling.MODES[ARGS.pooling]),
                  std_population=True, std_epsilon=pooling.STD_EPS,
                  augmentation='independent deterministic H/V flips; seed, epoch, image index',
                  loader_rng='historical global generator; train2 workers, eval4 workers; probe RNG restored',
                  view='whole native972x1296; replicate-pad bottom20/right16 to992x1312',
                  normalization='CLAHE+percentile01; Cellpose01',
                  checkpoint_policy=f'fixed terminal epoch{ARGS.epochs} primary; intermediate minima diagnostic',
                  validation=f'all {len(va)} validation images, one whole-frame view, no TTA or calibration',
                  labels='original positive labels; train-only log mean/std; no zero exclusions',
                  metric='mean of three per-target MAPE percentages')
    m = dict(created_utc=utc_now().isoformat(), architecture=ARCH, fold=ARGS.fold,
             seed=ARGS.seed, split_path=str(SPLIT), split_sha256=digest(SPLIT),
             train_n=len(tr), validation_n=len(va), images=len(ids),
             train_ids_sha=ids_sha(ids[tr]), validation_ids_sha=ids_sha(ids[va]),
             train_order_base_sha=array_sha(tr), train_indices=tr.tolist(), validation_indices=va.tolist(),
             primary_validation='all full validation images; no target or group subset filtering',
             historical_validation_indices=historical_validation().tolist(),
             historical_validation_n=len(historical_validation()),
             excluded_from_split_indices=np.setdiff1d(np.arange(len(ids)),np.r_[tr,va]).tolist(),
             historical_subset_role='secondary diagnostic only; old historical-training overlap exclusions',
             log_mean=log_y.mean(0).tolist(), log_std=log_y.std(0).tolist(), recipe=recipe,
             raw_sha256=raw_sha, raw_bytes=b.RAW.stat().st_size,
             hashes={str(p): digest(p) for p in files},
             ssl_root=str(SSL), ssl_manifest_sha=digest(SSL/'manifest.json'),
             ssl_checkpoint_sha=digest(SSL/'run/last.pt'), ssl_encoder_tensor_sha=encoder_sha,
             ssl_epochs=5, ssl_seed=sm['seed'], ssl_steps=sr['steps'],
             expected_head_sha=head_sha, mode='full', deadline_utc=ARGS.deadline_utc,
             execution=dict(world_size=2, micro_batch_per_gpu=4, global_batch=16,
                            accumulation_rounds=2, partial_groups='sample weighted; zero-weight dummy as needed'),
             diagnostics=dict(train_probe_n=min(256,len(tr)), train_probe_seed=12012,
                              train_probe_mode='eval/no augmentation; all RNG states restored',
                              epoch_weights='every epoch model only; no exact resume',
                              epoch_predictions=True, online_train_metrics=True,
                              gradient_norms=True, clipping_fraction=True),
             limits=['Single model; no ensemble, TTA or calibration',
                     'Fold-specific SSL uses only this fold training partition',
                     'Patient independence and near-duplicate leakage not independently verified',
                     'Terminal epoch primary; intermediate minima retrospective diagnostics',
                     'No exact resume is supported; deadline interruption produces an incomplete arm',
                     'H200 training is not T4 deployment certification'],
             torch_version=torch.__version__, cuda_version=torch.version.cuda,
             official_release_untouched=True)
    save(ROOT/'manifest.json', m)
    print(json.dumps(dict(status='prepared',root=str(ROOT),fold=ARGS.fold,
                          recipe=recipe,ssl_checkpoint_sha=m['ssl_checkpoint_sha'])),flush=True)


def verify():
    m = read(ROOT/'manifest.json')
    assert m['fold'] == ARGS.fold and m['seed'] == ARGS.seed
    assert m['split_sha256'] == digest(SPLIT) and m['ssl_root'] == str(SSL)
    assert m['deadline_utc'] == ARGS.deadline_utc
    for key, value in (('pooling', ARGS.pooling), ('head_lr', ARGS.head_lr), ('gradient_clip', ARGS.clip),
                       ('epochs', ARGS.epochs), ('schedule_epochs', ARGS.schedule_epochs)):
        assert m['recipe'][key] == value, key
    for path, expected in m['hashes'].items():
        assert digest(path) == expected, f'Frozen input/source changed: {path}'
    for source in source_files():
        assert digest(source)==m['hashes'][str(ROOT/'source'/source.name)]
    assert b.RAW.stat().st_size == m['raw_bytes']
    ids,y,tr,va = arrays()
    assert m['train_ids_sha'] == ids_sha(ids[tr]) and m['validation_ids_sha'] == ids_sha(ids[va])
    np.testing.assert_array_equal(tr, m['train_indices'])
    np.testing.assert_array_equal(va, m['validation_indices'])
    np.testing.assert_array_equal(historical_validation(),m['historical_validation_indices'])
    logs = np.log(y[tr].astype(np.float64))
    np.testing.assert_array_equal(logs.mean(0), m['log_mean'])
    np.testing.assert_array_equal(logs.std(0), m['log_std'])
    return m


def flip_decisions(seed, epoch, index):
    return np.random.default_rng(np.random.SeedSequence([seed, epoch, index])).random(2) < .5


class Frames(b.Frames):
    """Original full-frame transform with the requested experiment seed."""
    def __init__(self, indices, targets, total, epoch=0, augment=False, seed=42):
        super().__init__(indices, targets, total, epoch, False)
        self.use_augmentation, self.seed = augment, seed

    def __getitem__(self, row):
        x, y, index = super().__getitem__(row)
        if self.use_augmentation:
            horizontal, vertical = flip_decisions(self.seed, self.epoch, index)
            if horizontal:
                x = torch.flip(x, (-1,))
            if vertical:
                x = torch.flip(x, (-2,))
        return x, y, index


def loader(indices, targets, total, epoch=0, augment=False, batch_size=4):
    return DataLoader(Frames(indices,targets,total,epoch,augment,ARGS.seed),
                      batch_size=batch_size,shuffle=False,num_workers=2 if augment else 4,pin_memory=True,
                      prefetch_factor=1 if augment else 2,drop_last=False)


@contextmanager
def preserved_rng(rank):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[rank]):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def setup(m):
    net, details = build_cpu(m['seed'])
    head = prior.state_sha(net[1:].state_dict())
    assert head == m['expected_head_sha']
    checkpoint = torch.load(SSL/'run/last.pt', map_location='cpu', weights_only=True)
    assert checkpoint['epoch'] == 5 and checkpoint['step'] == m['ssl_steps']
    assert checkpoint['manifest_sha'] == m['ssl_manifest_sha']
    net[0].original.load_state_dict(checkpoint['encoder'], strict=True)
    assert prior.state_sha(net[0].original.state_dict()) == m['ssl_encoder_tensor_sha']
    assert prior.state_sha(net[1:].state_dict()) == head
    del checkpoint
    assert all(p.requires_grad for p in net.parameters())
    assert not any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in net.modules())
    net = net.cuda()
    recipe = m['recipe']
    opt = torch.optim.AdamW([{'params':net[0].parameters(),'lr':recipe['encoder_lr']},
                            {'params':list(net[1:].parameters()),'lr':recipe['head_lr']}],
                           weight_decay=recipe['weight_decay'],fused=True)
    mu = torch.tensor(m['log_mean'],device='cuda',dtype=torch.float32)
    sigma = torch.tensor(m['log_std'],device='cuda',dtype=torch.float32)
    details.update(common_padded_input=[992,1312],patch_or_feature_grid=[124,164],
                   ssl_checkpoint_sha=m['ssl_checkpoint_sha'],ssl_epochs=5,
                   encoder_tensor_sha=m['ssl_encoder_tensor_sha'],head_initial_tensor_sha=head,
                   projection_or_ssl_optimizer_loaded=False,distributed_execution=m['execution'],
                   mode='full',seed=m['seed'],gpu=torch.cuda.get_device_name(),
                   torch_version=torch.__version__,cuda_version=torch.version.cuda,
                   bitwise_determinism_guaranteed=False,
                   encoder_trainable=sum(p.numel() for p in net[0].parameters()),
                   head_trainable=sum(p.numel() for p in net[1:].parameters()))
    return net,opt,mu,sigma,details


def init_dist():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=20))
    assert dist.get_world_size() == 2 and rank == dist.get_rank()
    return rank


def distributed_backward(ddp,x,t,mu,sigma,rank):
    n = len(t)
    stats = torch.zeros(7,device=mu.device,dtype=torch.float64)
    for j in range(2):
        start = (2*j+rank)*4
        count = max(0,min(4,n-start))
        xx, yy = (x[start:start+count],t[start:start+count]) if count else (x[:1],t[:1])
        with ddp.no_sync() if j == 0 else nullcontext():
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z = ddp(b.inputs(xx,ARCH))
                loss = b.loss_fn(z,yy.cuda(non_blocking=True),mu,sigma)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            if count:
                with torch.no_grad():
                    zt = (yy.cuda(non_blocking=True).log()-mu)/sigma
                    rel = torch.expm1((z.float()-zt)*sigma).abs()
                    stats[0] += loss.detach().double()*count
                    stats[1:4] += rel.double().sum(0)*100
                    stats[4:7] += (z.float()-zt).abs().double().sum(0)
            (loss*(2*count/n)).backward()
    return stats


def lr_ratio(step,updates,schedule_epochs):
    if step < updates:
        return (step+1)/updates
    phase = min(1., (step-updates)/max(1,updates*(schedule_epochs-1)-1))
    return .1+.9*.5*(1+math.cos(math.pi*phase))


def cpu_tests():
    m = verify()
    if (ROOT/'cpu_tests.json').exists():
        raise FileExistsError('CPU test evidence already exists')
    pooling.cpu_contract_tests()
    b.tests()
    for seed in (ARGS.seed,ARGS.seed+1):
        seed_everything(seed)
        a = (random.random(),np.random.random(),torch.rand(4))
        seed_everything(seed)
        c = (random.random(),np.random.random(),torch.rand(4))
        assert a[:2] == c[:2] and torch.equal(a[2],c[2])
    # Test actual head initialization, including first linear and LayerNorm.
    net,_ = build_cpu(ARGS.seed)
    assert prior.state_sha(net[1:].state_dict()) == m['expected_head_sha']
    same_head = [net[index].weight.detach().clone() for index in (2,4)]
    del net
    net,_ = build_cpu(ARGS.seed+1)
    assert prior.state_sha(net[1:].state_dict()) != m['expected_head_sha']
    for index,weight in zip((2,4),same_head):
        assert not torch.equal(net[index].weight,weight), index
    del net,same_head
    a = np.array([flip_decisions(ARGS.seed,3,i) for i in range(128)])
    assert np.array_equal(a,np.array([flip_decisions(ARGS.seed,3,i) for i in range(128)]))
    assert not np.array_equal(a,np.array([flip_decisions(ARGS.seed+1,3,i) for i in range(128)]))
    assert not np.array_equal(a,np.array([flip_decisions(ARGS.seed,4,i) for i in range(128)]))
    _,_,tr,_ = arrays()
    assert not np.array_equal(np.random.default_rng(ARGS.seed).permutation(tr),
                              np.random.default_rng(ARGS.seed+1).permutation(tr))
    for schedule in (2,8,20):
        ratios = [lr_ratio(step,7,schedule) for step in range(7*(schedule+2))]
        assert ratios[0] == 1/7 and ratios[6] == 1 and ratios[7] == 1
        assert abs(ratios[7*schedule-1]-.1)<1e-12
        assert all(.1-1e-12 <= v <= 1 for v in ratios)
        assert all(ratios[i+1] <= ratios[i]+1e-12 for i in range(7,len(ratios)-1))
    # Numerically verify DDP averaged gradients against full-batch CPU references.
    mu=torch.tensor([1.,2.,3.],dtype=torch.float64)
    sigma=torch.tensor([.18,.19,.16],dtype=torch.float64)
    for n in (1,4,5,8,9,12,14,15,16):
        inputs=torch.arange(n*5,dtype=torch.float64).reshape(n,5)/100
        target=torch.exp(mu+sigma*torch.arange(n*3,dtype=torch.float64).reshape(n,3)/100)
        ref=torch.nn.Linear(5,3,bias=True,dtype=torch.float64)
        ref.zero_grad();b.loss_fn(ref(inputs),target,mu,sigma).backward()
        gradients=[torch.zeros_like(p) for p in ref.parameters()]
        covered=[];weight=0.
        for rank in (0,1):
            local=torch.nn.Linear(5,3,bias=True,dtype=torch.float64)
            local.load_state_dict(ref.state_dict());local.zero_grad()
            for j in range(2):
                start=(2*j+rank)*4;count=max(0,min(4,n-start))
                covered.extend(range(start,start+count));weight+=2*count/n/2
                xx=inputs[start:start+count] if count else inputs[:1]
                yy=target[start:start+count] if count else target[:1]
                (b.loss_fn(local(xx),yy,mu,sigma)*(2*count/n)).backward()
            for total,p in zip(gradients,local.parameters()):
                total += p.grad/2
        assert sorted(covered)==list(range(n)) and abs(weight-1)<1e-12
        for expected,actual in zip(ref.parameters(),gradients):
            torch.testing.assert_close(expected.grad,actual,rtol=1e-6,atol=1e-7)
    for threshold in (1.,3.):
        p=torch.nn.Parameter(torch.ones(2));p.grad=torch.tensor([3.,4.])
        norm=torch.nn.utils.clip_grad_norm_([p],threshold,error_if_nonfinite=True)
        assert abs(norm.item()-5.)<1e-6 and abs(p.grad.norm().item()-threshold)<1e-5
    deadline=parse_deadline(ARGS.deadline_utc)
    assert deadline.utcoffset().total_seconds()==0
    save(ROOT/'cpu_tests.json',dict(status='passed',manifest_sha=digest(ROOT/'manifest.json'),
         actual_head_seed_verified=True,all_rng_seeded=True,augmentation_seed_verified=True,
         order_seed_verified=True,partial_ddp_gradient_equivalence=True,
         clipping_verified=True,schedule_terminal_and_clamping_verified=True,metric_identity=True))
    print('PASS: seeds/head/augmentation/order, partial DDP gradients, metrics, schedule, clipping',flush=True)


def block_key(name):
    match=re.search(r'blocks\.\d+',name)
    return match.group() if match else None


def block_hashes(net):
    blocks={block_key(n) for n,_ in net[0].original.named_parameters() if block_key(n)}
    state=net[0].original.state_dict()
    return {block:prior.state_sha({n:v for n,v in state.items() if block_key(n)==block}) for block in blocks}


def smoke():
    m=verify();ensure_before_deadline(m['deadline_utc']);rank=init_dist()
    if (ROOT/'smoke.json').exists():
        raise FileExistsError('Smoke already exists')
    net,opt,mu,sigma,details=setup(m)
    ddp=DDP(net,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=True)
    ids,y,tr,_=arrays()
    ds=Frames(tr[:9],y,len(ids));data=[ds[i] for i in range(9)]
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
                diff=(a.grad-c.grad).float();error+=diff.square().sum().item()
                norm+=c.grad.float().square().sum().item();maximum=max(maximum,diff.abs().max().item())
        relative=math.sqrt(error/max(norm,1e-30))
        evidence[0]=dict(relative_gradient_l2=relative,max_abs_gradient_error=maximum,
                         passed=relative<.002,includes_final9_samples_and_zero_weight_dummy=True)
        del ref
    dist.broadcast_object_list(evidence,src=0);assert evidence[0]['passed'],evidence[0]
    before=block_hashes(net);assert len(before)==12
    head_before=prior.state_sha(net[1:].state_dict())
    reached={block_key(n) for n,p in net[0].original.named_parameters()
             if block_key(n) and p.grad is not None and p.grad.abs().max().item()>0}
    assert set(before)==reached
    assert any('patch_embed' in n and p.grad is not None and p.grad.abs().max().item()>0
               for n,p in net[0].original.named_parameters())
    b.finite_grad(net)
    torch.nn.utils.clip_grad_norm_(net.parameters(),m['recipe']['gradient_clip'],error_if_nonfinite=True)
    opt.step();after=block_hashes(net)
    assert all(before[k]!=after[k] for k in before), 'All encoder blocks must actually update'
    assert prior.state_sha(net[1:].state_dict())!=head_before
    hashes=[None,None];dist.all_gather_object(hashes,prior.state_sha(net.state_dict()))
    assert hashes[0]==hashes[1]
    ddp.train();opt.zero_grad(set_to_none=True)
    distributed_backward(ddp,x,t,mu,sigma,rank)
    b.finite_grad(net)
    torch.nn.utils.clip_grad_norm_(net.parameters(),m['recipe']['gradient_clip'],error_if_nonfinite=True)
    opt.step()
    hashes=[None,None];dist.all_gather_object(hashes,prior.state_sha(net.state_dict()))
    assert hashes[0]==hashes[1]
    if rank==0:
        report=dict(status='passed',manifest_sha=digest(ROOT/'manifest.json'),details=details,
                    single_gpu_reference_gradient_check=evidence[0],ranks_synchronized=True,
                    all_12_encoder_blocks_updated=True,patch_gradient_verified=True,head_updated=True)
        save(ROOT/'smoke.json',report);print(json.dumps(report),flush=True)
    dist.barrier();dist.destroy_process_group()


def deadline_hit(m,rank):
    # Rank0's live clock is authoritative, synchronized before every optimizer update.
    stop=torch.tensor([int(rank==0 and utc_now()>=parse_deadline(m['deadline_utc']))],device='cuda')
    dist.broadcast(stop,src=0)
    return bool(stop.item())


def train():
    m=verify();ensure_before_deadline(m['deadline_utc']);rank=init_dist()
    run=ROOT/'runs'/f'fold{m["fold"]}'/ARCH
    for name in ('cpu_tests.json','smoke.json'):
        check=read(ROOT/name)
        assert check['status']=='passed' and check['manifest_sha']==digest(ROOT/'manifest.json')
    if rank==0:
        if run.exists():
            raise FileExistsError('No overwrite or implicit resume')
        run.mkdir(parents=True)
    dist.barrier()
    net,opt,mu,sigma,details=setup(m)
    ddp=DDP(net,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=True)
    ids,y,tr,va=arrays();updates=math.ceil(len(tr)/16);step=0;history=[];wall=time.monotonic()
    val=loader(va,y,len(ids)) if rank==0 else None
    probe=np.sort(np.random.default_rng(12012).choice(tr,min(256,len(tr)),replace=False))
    probe_loader=loader(probe,y,len(ids)) if rank==0 else None
    initial_encoder=prior.state_sha(net[0].original.state_dict())
    if rank==0:
        save(run/'progress.json',dict(status='starting',epochs=m['recipe']['epochs'],world_size=2,
                                     details=details,deadline_utc=m['deadline_utc']))
    for epoch in range(m['recipe']['epochs']):
        order=np.random.default_rng(m['seed']+epoch).permutation(tr)
        dl=loader(order,y,len(ids),epoch,True,batch_size=16);seen=[];tick=time.monotonic()
        ddp.train();torch.cuda.reset_peak_memory_stats()
        totals=torch.zeros(7,device=mu.device,dtype=torch.float64)
        norm_sum=torch.zeros((),device=mu.device);norm_max=torch.zeros_like(norm_sum);clipped=torch.zeros_like(norm_sum)
        stopped=False
        for x,t,ix in dl:
            if deadline_hit(m,rank):
                stopped=True
                break
            ratio=lr_ratio(step,updates,m['recipe']['schedule_epochs'])
            opt.param_groups[0]['lr']=m['recipe']['encoder_lr']*ratio
            opt.param_groups[1]['lr']=m['recipe']['head_lr']*ratio
            opt.zero_grad(set_to_none=True);totals+=distributed_backward(ddp,x,t,mu,sigma,rank)
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),m['recipe']['gradient_clip'],error_if_nonfinite=True)
            norm_sum+=norm.detach();norm_max=torch.maximum(norm_max,norm.detach())
            clipped+=(norm.detach()>m['recipe']['gradient_clip']).float()
            if deadline_hit(m,rank):
                stopped=True
                break
            opt.step();step+=1;seen.extend(ix.tolist())
            if rank==0 and (step%25==0 or step==1):
                item=dict(status='training',epoch=epoch+1,epochs=m['recipe']['epochs'],seen=len(seen),
                          train_n=len(tr),updates=step,world_size=2,gradient_norm=norm.item(),
                          encoder_lr=opt.param_groups[0]['lr'],head_lr=opt.param_groups[1]['lr'],
                          seconds_this_epoch=time.monotonic()-tick,
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                save(run/'progress.json',item);print(json.dumps(item),flush=True)
        if stopped:
            if rank==0:
                torch.save(dict(model=net.state_dict(),epoch_complete=epoch,partial_epoch_seen=len(seen),
                                updates=step,manifest_sha=digest(ROOT/'manifest.json'),
                                log_mean=m['log_mean'],log_std=m['log_std'],
                                note='Deadline model snapshot; incomplete, no resume or primary score'),run/'deadline.pt')
                save(run/'stopped.json',dict(status='deadline_stopped',completed_epochs=epoch,
                     partial_epoch_seen=len(seen),updates=step,deadline_utc=m['deadline_utc'],
                     stopped_utc=utc_now().isoformat(),checkpoint_sha=digest(run/'deadline.pt')))
                save(run/'progress.json',read(run/'stopped.json'))
            dist.barrier();dist.destroy_process_group()
            return
        np.testing.assert_array_equal(np.array(seen),order)
        assert step==(epoch+1)*updates
        dist.all_reduce(totals)
        assert prior.state_sha(net[0].original.state_dict())!=initial_encoder
        dist.barrier();train_seconds=time.monotonic()-tick
        if rank==0:
            # Retain the historical validation RNG path; only added probe is protected.
            pred=b.evaluate(net,ARCH,val,mu,sigma,va)
            score=b.metrics(pred,y[va])
            historical_score=secondary_score(pred,y,va)
            with preserved_rng(rank):
                probe_pred=b.evaluate(net,ARCH,probe_loader,mu,sigma,probe)
                probe_score=b.metrics(probe_pred,y[probe])
            values=(totals/len(tr)).cpu().tolist()
            item=dict(epoch=epoch+1,score=score,train_seconds=train_seconds,
                      historical_subset_score=historical_score,
                      online_train_loss=values[0],online_train_mape=dict(zip(('CD','CV','HEX'),values[1:4])),
                      online_standardized_log_l1=dict(zip(('CD','CV','HEX'),values[4:7])),
                      train_probe_score=probe_score,gradient_norm_mean=(norm_sum/updates).item(),
                      gradient_norm_max=norm_max.item(),gradient_clip_fraction=(clipped/updates).item(),
                      gradient_clip_threshold=m['recipe']['gradient_clip'],
                      encoder_lr=opt.param_groups[0]['lr'],head_lr=opt.param_groups[1]['lr'],
                      epoch_seconds=time.monotonic()-tick,updates=step,order_sha=array_sha(order),
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            history.append(item);save(run/'history.json',history)
            torch.save(dict(model=net.state_dict(),epoch=epoch+1,architecture=ARCH,
                            manifest_sha=digest(ROOT/'manifest.json'),log_mean=m['log_mean'],log_std=m['log_std'],
                            note=f'Model only; terminal epoch{m["recipe"]["epochs"]} primary; no resume'),
                       run/f'epoch_{epoch+1:02d}.pt')
            np.savez_compressed(run/f'predictions_epoch_{epoch+1:02d}.npz',indices=va,ids=ids[va],target=y[va],pred=pred)
            tmp=run/'last.pt.tmp'
            torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,
                            architecture=ARCH,manifest_sha=digest(ROOT/'manifest.json'),
                            log_mean=m['log_mean'],log_std=m['log_std'],history=history,
                            note='No resume supported; not both-rank RNG checkpoint'),tmp)
            tmp.replace(run/'last.pt')
            save(run/'progress.json',dict(status='epoch_complete',**item))
            print(json.dumps(dict(epoch_complete=epoch+1,**item)),flush=True)
        dist.barrier()
    hashes=[None,None];dist.all_gather_object(hashes,prior.state_sha(net.state_dict()))
    assert hashes[0]==hashes[1]
    if rank==0:
        np.savez_compressed(run/'predictions.npz',indices=va,ids=ids[va],target=y[va],pred=pred)
        result=dict(architecture=ARCH,fold=m['fold'],seed=m['seed'],score=score,history=history,details=details,
                    historical_subset_score=historical_score,
                    seconds=time.monotonic()-wall,manifest_sha=digest(ROOT/'manifest.json'),
                    checkpoint_sha=digest(run/'last.pt'),predictions_sha=digest(run/'predictions.npz'),
                    ranks_synchronized=True,limits=m['limits'],official_submission_changed=False,
                    primary=f'fixed terminal epoch{m["recipe"]["epochs"]}',single_model=True)
        save(run/'result.json',result)
        save(run/'complete.json',dict(status='completed',result_sha=digest(run/'result.json')))
        save(run/'progress.json',dict(status='completed',epoch=m['recipe']['epochs'],score=score,updates=step))
    dist.barrier();dist.destroy_process_group()


def audit():
    m=verify()
    if (ROOT/'audit.json').exists():
        raise FileExistsError('Audit already exists')
    run=ROOT/'runs'/f'fold{m["fold"]}'/ARCH
    result=prior.validate_result(run,digest(ROOT/'manifest.json'))
    assert digest(run/'predictions.npz')==result['predictions_sha']
    assert len(result['history'])==m['recipe']['epochs']
    assert result['ranks_synchronized'] and result['single_model']
    ids,y,tr,va=arrays();updates=math.ceil(len(tr)/16)
    epochs=[]
    for epoch,item in enumerate(result['history'],1):
        assert item['epoch']==epoch and item['updates']==epoch*updates
        assert item['order_sha']==array_sha(np.random.default_rng(m['seed']+epoch-1).permutation(tr))
        predicted=run/f'predictions_epoch_{epoch:02d}.npz'
        weights=run/f'epoch_{epoch:02d}.pt'
        with np.load(predicted) as z:
            np.testing.assert_array_equal(z['indices'],va)
            np.testing.assert_array_equal(z['ids'],ids[va])
            np.testing.assert_array_equal(z['target'],y[va])
            score=b.metrics(z['pred'],z['target'])
            historical_score=secondary_score(z['pred'],y,va)
        for key in ('CD','CV','HEX','mean','n'):
            assert abs(score[key]-item['score'][key])<1e-6
            assert abs(historical_score[key]-item['historical_subset_score'][key])<1e-6
        checkpoint=torch.load(weights,map_location='cpu',weights_only=True)
        assert checkpoint['epoch']==epoch and checkpoint['manifest_sha']==digest(ROOT/'manifest.json')
        del checkpoint
        epochs.append(dict(epoch=epoch,weights_sha=digest(weights),predictions_sha=digest(predicted),score=score))
    with np.load(run/'predictions.npz') as z, np.load(run/f'predictions_epoch_{m["recipe"]["epochs"]:02d}.npz') as final:
        for key in ('indices','ids','target','pred'):
            np.testing.assert_array_equal(z[key],final[key])
        score=b.metrics(z['pred'],z['target'])
    for key in ('CD','CV','HEX','mean','n'):
        assert abs(score[key]-result['score'][key])<1e-6
    checkpoint=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
    final=torch.load(run/f'epoch_{m["recipe"]["epochs"]:02d}.pt',map_location='cpu',weights_only=True)
    assert prior.state_sha(checkpoint['model'])==prior.state_sha(final['model'])
    assert checkpoint['epoch']==m['recipe']['epochs']
    assert checkpoint['manifest_sha']==digest(ROOT/'manifest.json')
    best=min(result['history'],key=lambda item:item['score']['mean'])
    report=dict(status='single_model_crossfold_regression_verified',fold=m['fold'],seed=m['seed'],
                manifest_sha=digest(ROOT/'manifest.json'),score=score,recipe=m['recipe'],
                primary=f'fixed terminal epoch{m["recipe"]["epochs"]}',lower_is_better=True,
                retrospective_best=dict(epoch=best['epoch'],score=best['score'],not_primary=True),
                historical_subset_score=result['historical_subset_score'],
                ssl_split_exact_match=True,ssl_train_only=True,all_validation_images_scored=True,
                epoch_artifacts=epochs,single_model=True,limits=m['limits'])
    save(ROOT/'audit.json',report);print(json.dumps(report),flush=True)


def main():
    global ARGS,ROOT,SPLIT,SSL
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','cpu_tests','cpu-tests','smoke','train','audit'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--split',type=Path,required=True)
    parser.add_argument('--fold',type=int,required=True)
    parser.add_argument('--pooling', choices=list(pooling.MODES), required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--ssl-root',type=Path,required=True)
    parser.add_argument('--head-lr',type=float,required=True)
    parser.add_argument('--clip',type=float,required=True)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--schedule-epochs',type=int,default=20)
    parser.add_argument('--deadline-utc',required=True)
    ARGS=parser.parse_args()
    if ARGS.seed<0 or ARGS.seed>=2**32-1 or ARGS.fold<0:
        parser.error('seed must be in [0,2**32-2], fold nonnegative')
    if ARGS.epochs<1 or ARGS.schedule_epochs<2 or ARGS.head_lr<=0 or ARGS.clip<=0:
        parser.error('positive epochs, head-lr, clip and schedule-epochs >= 2 required')
    parse_deadline(ARGS.deadline_utc)
    ROOT,SPLIT,SSL=ARGS.root,ARGS.split,ARGS.ssl_root
    globals()[ARGS.command.replace('-','_')]()


if __name__=='__main__':
    main()
