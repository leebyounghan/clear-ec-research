"""Matched 20-epoch regression after the one-epoch LeJEPA pilot.

Import an immutable copy of the original trainer. The only model change is a
strict encoder weight replacement AFTER original head/optimizer initialization.
The original trainer, official release, and SSL pilot are never modified.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import phase2_foundation_train_20261002 as baseline
from phase2_foundation_preflight_20261002 import build, digest, save

ROOT = Path('experiments/phase2_lejepa_regression_20261003')
BASE = Path('experiments/phase2_foundation_train_20261002')
SSL = Path('experiments/phase2_lejepa_20261003')
SSL_WEIGHT = SSL/'pilot/last.pt'
ARCH = 'cpdino_vitb'
BASE_RUN = BASE/'runs/fold3'/ARCH
ORIGINAL_SETUP = baseline.setup


def read(path):
    return json.loads(Path(path).read_text())


def state_sha(state):
    h = hashlib.sha256()
    for name,value in sorted(state.items()):
        v = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(v.dtype).encode()); h.update(str(v.shape).encode())
        h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def validate_result(folder,expected_manifest):
    complete=read(folder/'complete.json'); result=read(folder/'result.json')
    assert complete['status']=='completed'
    assert digest(folder/'result.json')==complete['result_sha']
    assert result['manifest_sha']==expected_manifest
    assert digest(folder/'last.pt')==result['checkpoint_sha']
    return result


def prepare():
    if (ROOT/'manifest.json').exists(): raise FileExistsError('Frozen manifest already exists')
    baseline.ROOT=BASE
    base=baseline.verify()
    assert digest(baseline.RAW)==base['raw_sha256']
    ids,y,tr,va=baseline.arrays()
    sm=read(SSL/'manifest.json')
    for p,expected in sm['hashes'].items(): assert digest(p)==expected,p
    sr=validate_result(SSL/'pilot',digest(SSL/'manifest.json'))
    assert sr['status']=='SSL_PILOT_COMPLETE_NOT_ACCURACY_VALIDATED'
    assert sr['train_images_seen']==sr['unique_train_images']==len(tr)
    assert sr['validation_images_used']==0 and sr['steps']==224
    expected_order=np.random.default_rng(1003).permutation(tr).astype(np.int64)
    assert sr['seen_order_sha']==hashlib.sha256(expected_order.tobytes()).hexdigest()
    assert sm['raw_sha256']==base['raw_sha256']
    assert sm['validation_ids_sha']==base['validation_ids_sha']
    assert sm['train_ids_sha']==hashlib.sha256('\n'.join(ids[tr]).encode()).hexdigest()
    for path in (baseline.META,baseline.SPLIT,baseline.WEIGHTS[ARCH]):
        assert sm['hashes'][str(path)]==base['hashes'][str(path)]
    br=validate_result(BASE_RUN,digest(BASE/'manifest.json'))
    assert digest(BASE_RUN/'predictions.npz')==br['predictions_sha']
    with np.load(BASE_RUN/'predictions.npz') as z:
        np.testing.assert_array_equal(z['indices'],va)
        np.testing.assert_array_equal(z['ids'],ids[va])
        np.testing.assert_array_equal(z['target'],y[va])
        score=baseline.metrics(z['pred'],z['target'])
    for k in ('CD','CV','HEX','mean'): assert abs(score[k]-br['score'][k])<1e-6
    for name in ('phase2_foundation_train_20261002.py','phase2_foundation_preflight_20261002.py'):
        assert digest(ROOT/'source'/name)==base['hashes'][str(BASE/'source'/name)]
    files=[ROOT/'source/phase2_lejepa_regression_20261003.py',
           ROOT/'source/phase2_foundation_train_20261002.py',
           ROOT/'source/phase2_foundation_preflight_20261002.py',
           BASE/'manifest.json',BASE_RUN/'complete.json',BASE_RUN/'result.json',BASE_RUN/'predictions.npz',
           SSL/'manifest.json',SSL/'pilot/complete.json',SSL/'pilot/result.json',SSL_WEIGHT]
    m=copy.deepcopy(base)
    m.update(created_utc=time.strftime('%FT%TZ',time.gmtime()),architectures=[ARCH],gpu_parallelism=1,
             intervention='Only encoder initialization: external CellposeDINO -> train-fold-only LeJEPA epoch1',
             baseline_score=score,ssl_steps=224,ssl_epochs=1,
             baseline_manifest_sha=digest(BASE/'manifest.json'),
             ssl_manifest_sha=digest(SSL/'manifest.json'),ssl_checkpoint_sha=digest(SSL_WEIGHT),
             limits=['Same reused development fold3, not an independent Phase II score',
                     'Single seed, one-epoch SSL pilot only; not an optimum SSL duration claim',
                     'Patient independence and near-duplicate leakage remain unverified',
                     'Fixed epoch20 primary; intermediate epoch scores diagnostic only',
                     'T4 inference remains unverified'])
    m['hashes'].update({str(p):digest(p) for p in files})
    save(ROOT/'manifest.json',m)
    print(json.dumps(dict(status='prepared',baseline_score=score,recipe=m['recipe'],
                          ssl_checkpoint_sha=m['ssl_checkpoint_sha'],changes=['encoder_initialization_only'])),flush=True)


def verify():
    baseline.ROOT=ROOT
    m=baseline.verify()
    base=read(BASE/'manifest.json')
    assert m['recipe']==base['recipe']
    for key in ('log_mean','log_std','train_order_base_sha','validation_ids_sha','raw_sha256'):
        assert m[key]==base[key],key
    return m


def replace_encoder(net,m):
    head_before=state_sha(net[1:].state_dict())
    old_encoder_sha=state_sha(net[0].original.state_dict())
    rng=torch.get_rng_state().clone()
    checkpoint=torch.load(SSL_WEIGHT,map_location='cpu',weights_only=True)
    assert checkpoint['step']==224 and checkpoint['manifest_sha']==m['ssl_manifest_sha']
    new_state=checkpoint['encoder']
    expected_sha=state_sha(new_state)
    net[0].original.load_state_dict(new_state,strict=True)
    actual_sha=state_sha(net[0].original.state_dict())
    assert actual_sha==expected_sha and actual_sha!=old_encoder_sha
    assert head_before==state_sha(net[1:].state_dict())
    assert torch.equal(rng,torch.get_rng_state())
    return dict(encoder_tensor_sha=actual_sha,head_initial_tensor_sha=head_before,
                external_encoder_tensor_sha=old_encoder_sha,strict_load=True,
                projection_head_loaded=False,ssl_optimizer_loaded=False)


def matched_setup(arch,m):
    assert arch==ARCH
    net,opt,mu,sigma,details=ORIGINAL_SETUP(arch,m)
    assert not opt.state, 'Optimizer must start fresh'
    init=replace_encoder(net,m)
    details.update(ssl_initialization=init,ssl_checkpoint=str(SSL_WEIGHT),
                   ssl_checkpoint_sha=m['ssl_checkpoint_sha'],
                   experiment='LeJEPA epoch1 -> matched20epoch regression')
    print(json.dumps(dict(initialization_verified=init)),flush=True)
    return net,opt,mu,sigma,details


def tests():
    m=verify(); baseline.tests()
    baseline.seed()
    net,_,_=build(ARCH,False,native_convnext_norm=True)
    torch.manual_seed(42)
    torch.nn.init.normal_(net[-1].weight,std=1e-3); torch.nn.init.zeros_(net[-1].bias)
    init=replace_encoder(net,m)
    save(ROOT/'cpu_tests.json',dict(status='passed',initialization=init,
                                  manifest_sha=digest(ROOT/'manifest.json')))
    print(json.dumps(dict(status='PASS',strict_encoder_load=init,recipe_identical=True)),flush=True)


def audit():
    m=verify(); run=ROOT/'runs/fold3'/ARCH
    if (ROOT/'audit.json').exists(): raise FileExistsError('Do not repeat completed audit')
    new=validate_result(run,digest(ROOT/'manifest.json'))
    old=validate_result(BASE_RUN,digest(BASE/'manifest.json'))
    assert len(new['history'])==len(old['history'])==20
    for n,o in zip(new['history'],old['history']):
        assert n['epoch']==o['epoch'] and n['order_sha']==o['order_sha'] and n['updates']==o['updates']
    assert digest(run/'predictions.npz')==new['predictions_sha']
    assert digest(BASE_RUN/'predictions.npz')==old['predictions_sha']
    with np.load(run/'predictions.npz') as n, np.load(BASE_RUN/'predictions.npz') as o:
        for k in ('indices','ids','target'): np.testing.assert_array_equal(n[k],o[k])
        ns=baseline.metrics(n['pred'],n['target']); os=baseline.metrics(o['pred'],o['target'])
    for score,result in ((ns,new),(os,old)):
        for k in ('CD','CV','HEX','mean'): assert abs(score[k]-result['score'][k])<1e-6
    report=dict(status='matched_comparison_verified',baseline=os,lejepa_epoch1=ns,
                delta_lejepa_minus_baseline={k:ns[k]-os[k] for k in ('CD','CV','HEX','mean')},
                lower_is_better=True,epoch=20,fold=3,validation_n=ns['n'],
                identical_training_orders=True,identical_validation_ids_and_targets=True,
                manifest_sha=digest(ROOT/'manifest.json'),limitations=m['limits'])
    save(ROOT/'audit.json',report); print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','tests','smoke','train','audit'])
    args=p.parse_args()
    if args.command in ('prepare','tests','audit'):
        globals()[args.command]()
    else:
        verify()
        assert torch.cuda.device_count()==1
        baseline.setup=matched_setup
        if args.command=='smoke':
            assert not (ROOT/'smoke'/f'{ARCH}.json').exists()
            baseline.smoke(ARCH)
        else:
            baseline.train(ARCH)
