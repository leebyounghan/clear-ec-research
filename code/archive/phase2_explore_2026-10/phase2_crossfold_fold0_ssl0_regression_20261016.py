"""Matched fold0 SSL0 control: external CPDINO, no additional CLEAR-EC SSL.

The pinned SSL5 runner supplies the unchanged loader, loss, schedule, DDP
gradient accumulation, smoke checks, training loop and epoch diagnostics. This
adapter replaces only initialization/provenance handling. All files in existing
experiments and all original Python sources remain untouched.

Local model-free contract checks: python this_file.py contract-tests
These synthetic checks do not load data, instantiate a model or verify GPU training.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys


# Fold0 is deliberately a separate, fixed control; this is not a generic tuner.
EXPECTED_SPLIT = 'd45a329b8f68103cf9d13b4b89ab88efd258466407f9241412dd891f2eb71ad8'
EXPECTED_DATA_PREFLIGHT = '98c04ceb0c01f852995be4089709ecf0e6c453e4e238830f0f31ec4dbc9ec07e'
DEFAULT_SPLIT = 'experiments/ssl_screen_20260911/preflight/fold0.npz'
DEFAULT_DATA_PREFLIGHT = 'experiments/phase2_crossfold_20261015/data_preflight_legacy.json'


def fixed_control_contract(fold, seed, head_lr, clip, epochs, schedule_epochs,
                           *, train_n=None, validation_n=None, split_sha=None):
    """Used by the real CLI/reference path and the model-free fixtures below."""
    integer_fields = (fold, seed, epochs, schedule_epochs)
    if any(type(value) is not int for value in integer_fields):
        raise ValueError('Control fold/seed/epoch fields must be integers')
    if (fold, seed, head_lr, clip, epochs, schedule_epochs) != (0, 42, 1e-3, 1., 20, 20):
        raise ValueError('SSL0 control is fixed to fold0/seed42/head1e-3/clip1/epochs20/schedule20')
    if train_n is not None and (type(train_n) is not int or train_n != 7151):
        raise ValueError('Fold0 requires exactly 7151 clean training images')
    if validation_n is not None and (type(validation_n) is not int or validation_n != 1791):
        raise ValueError('Fold0 primary requires all 1791 validation images')
    if split_sha is not None and split_sha != EXPECTED_SPLIT:
        raise ValueError('Unexpected fold0 clean split SHA256')


def legacy_preflight_contract(report_sha, evidence, entry):
    """Extra fold0 provenance gate; full pixel/group/array checks still follow."""
    if report_sha != EXPECTED_DATA_PREFLIGHT:
        raise ValueError('Fold0 requires the pinned verified legacy preflight report')
    if evidence.get('status') != 'shared_checks_passed' or evidence.get('errors') or evidence.get('failed_shared_checks'):
        raise ValueError('Legacy shared data checks did not pass')
    if not evidence.get('pixels', {}).get('verified'):
        raise ValueError('Legacy report lacks verified pixels')
    if entry.get('eligible_positive_target_candidate') is not True or entry.get('positive_targets_train_and_validation') is not True:
        raise ValueError('Chosen fold0 split is ineligible')
    if entry.get('split_sha256') != EXPECTED_SPLIT:
        raise ValueError('Legacy report describes another split')
    provenance = entry.get('provenance', {})
    if '0' not in provenance.get('matched_manifest_folds', []):
        raise ValueError('Legacy report does not match fold0')
    chain = provenance.get('legacy_explicit_chain', {})
    if (chain.get('status') != 'verified' or chain.get('fold') != '0'
            or chain.get('split_sha256') != EXPECTED_SPLIT
            or chain.get('explicit_downstream_split_path') != DEFAULT_SPLIT
            or chain.get('current_pixel_and_group_checks_reused_unchanged') is not True):
        raise ValueError('Missing or inconsistent explicit legacy fold0 provenance chain')
    if entry.get('train', {}).get('n') != 7151 or entry.get('validation', {}).get('n') != 1791:
        raise ValueError('Legacy fold0 population counts differ')


def contract_tests():
    """Synthetic configuration checks only; no NumPy, torch, real data or model."""
    accepted, rejected = [], []
    control = dict(fold=0, seed=42, head_lr=1e-3, clip=1., epochs=20,
                   schedule_epochs=20, train_n=7151, validation_n=1791, split_sha=EXPECTED_SPLIT)
    fixed_control_contract(**control); accepted.append('exact_fold0_control')
    for key, value in [('fold',4),('seed',43),('head_lr',1e-4),('clip',3.),('epochs',19),
                       ('schedule_epochs',19),('train_n',7150),('validation_n',1790),
                       ('validation_n',1781),('split_sha','0'*64),('fold',False),('head_lr',float('nan'))]:
        changed={**control,key:value}
        try:
            fixed_control_contract(**changed)
        except ValueError:
            rejected.append('control_'+key+'_'+str(value))
        else:
            raise AssertionError('Invalid control accepted: '+key)
    evidence=dict(status='shared_checks_passed',errors=[],failed_shared_checks=[],pixels=dict(verified=True))
    entry=dict(eligible_positive_target_candidate=True,positive_targets_train_and_validation=True,
               split_sha256=EXPECTED_SPLIT,train=dict(n=7151),validation=dict(n=1791),
               provenance=dict(matched_manifest_folds=['0'],legacy_explicit_chain=dict(
                   status='verified',fold='0',split_sha256=EXPECTED_SPLIT,
                   explicit_downstream_split_path=DEFAULT_SPLIT,current_pixel_and_group_checks_reused_unchanged=True)))
    legacy_preflight_contract(EXPECTED_DATA_PREFLIGHT,evidence,entry); accepted.append('exact_legacy_header')
    mutations=[('wrong_report_sha', 'report', ('sha',), '0'*64),
               ('failed_shared_checks','evidence',('failed_shared_checks',),['failed']),
               ('pixels_unverified','evidence',('pixels','verified'),False),
               ('ineligible_split','entry',('eligible_positive_target_candidate',),False),
               ('wrong_split_sha','entry',('split_sha256',),'0'*64),
               ('missing_matched_fold','entry',('provenance','matched_manifest_folds'),[]),
               ('missing_legacy_chain','entry',('provenance','legacy_explicit_chain'),{}),
               ('wrong_legacy_fold','entry',('provenance','legacy_explicit_chain','fold'),'4'),
               ('wrong_legacy_path','entry',('provenance','legacy_explicit_chain','explicit_downstream_split_path'),'fold0.npz'),
               ('wrong_validation_count','entry',('validation','n'),1781)]
    for name, target, path, value in mutations:
        objects=dict(report=dict(sha=EXPECTED_DATA_PREFLIGHT),evidence=copy.deepcopy(evidence),entry=copy.deepcopy(entry))
        obj=objects[target]
        for key in path[:-1]: obj=obj[key]
        obj[path[-1]]=value
        try:
            legacy_preflight_contract(objects['report']['sha'],objects['evidence'],objects['entry'])
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('Invalid legacy evidence accepted: '+name)
    if any(name in sys.modules for name in ('numpy','torch','phase2_crossfold_regression_20261015')):
        raise AssertionError('Model-free fixtures imported a training dependency')
    report=dict(status='passed',scope='synthetic model-free contract fixtures only; not real-data, model or GPU validation',
                accepted=accepted,rejected=rejected,cases=len(accepted)+len(rejected),
                numpy_imported=False,torch_imported=False,real_data_loaded=False,training_performed=False,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print(json.dumps(report,allow_nan=False))


# This single-purpose entry point must run before optional training imports.
if __name__ == '__main__' and sys.argv[1:] == ['contract-tests']:
    contract_tests()
    raise SystemExit(0)


import numpy as np
import torch

import phase2_crossfold_regression_20261015 as base

b, prior, preflight = base.b, base.prior, base.preflight
digest, save, read = base.digest, base.save, base.read
ARCH = base.ARCH
EXPECTED_BASE = '7289b583643c9c74ad300f73e2a0d6debbe38be5d6393b03508437cdc8232ffe'
EXPECTED_EXTERNAL = '3ed4c06a3963ab13ff377d4e2957174aaaf637434eb031ae9931fcbfaf9a217f'
EXPECTED_RAW = '93e67ac6df5887a1a5396c85e236b8398297dd3e728cdb6513238c1628c1c88a'
EXPECTED_META = '956f448a2de4d65b4e091d78f56bdde09aab9554e3ab027fc7fe58f8b69625e6'
FULL_META = Path('datasets/clear_ec_fullframe_20260904/fullframe_meta.npz')
ARGS = None
COMMON_FIELDS = (
    'architecture','fold','seed','split_path','split_sha256','train_n','validation_n','images',
    'train_ids_sha','validation_ids_sha','train_order_base_sha','train_indices','validation_indices',
    'primary_validation','historical_validation_indices','historical_validation_n',
    'excluded_from_split_indices','historical_subset_role','log_mean','log_std','recipe',
    'raw_sha256','raw_bytes','expected_head_sha','mode','deadline_utc','execution','diagnostics',
)


def source_files():
    return [Path(__file__).resolve(),Path(base.__file__).resolve(),Path(b.__file__).resolve(),
            Path(prior.__file__).resolve(),Path(preflight.__file__).resolve()]


def assert_pinned_sources():
    expected={**base.EXPECTED_REUSED_SOURCES,Path(base.__file__).name:EXPECTED_BASE}
    for source in source_files():
        if source.name in expected:
            assert digest(source)==expected[source.name],f'Pinned implementation changed: {source}'


def evidence_array_sha(value):
    value=np.ascontiguousarray(value)
    h=hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def preflight_evidence(ids,y,tr,va):
    evidence=read(ARGS.data_preflight)
    assert evidence['status']=='shared_checks_passed'
    assert not evidence['errors'] and not evidence['failed_shared_checks']
    assert evidence['checks'] and all(c['passed'] for c in evidence['checks'])
    pixels=evidence['pixels']
    assert pixels['verified'] and pixels['whole_cache_sha256']==EXPECTED_RAW
    assert pixels['n']==len(ids) and pixels['mismatch_rows']==0
    entries=[entry for path,entry in evidence['splits'].items()
             if Path(path).resolve()==ARGS.split.resolve()]
    assert len(entries)==1, 'Chosen split missing or ambiguous in data preflight'
    entry=entries[0]
    legacy_preflight_contract(digest(ARGS.data_preflight),evidence,entry)
    assert entry['eligible_positive_target_candidate'] and entry['positive_targets_train_and_validation']
    assert entry['split_sha256']==digest(ARGS.split)
    assert str(ARGS.fold) in entry['provenance']['matched_manifest_folds']
    assert entry['checks'] and all(c['passed'] for c in entry['checks'])
    checks={c['name']:c['passed'] for c in entry['checks']}
    for name in ('slide_disjoint','exact_pixel_disjoint','slide_exact_connected_disjoint',
                 'clean_train_reconstructs','id_disjoint','index_disjoint'):
        assert checks.get(name) is True, name
    for key in ('slide','exact_pixel','slide_exact_connected'):
        assert all(value==0 for value in entry['overlap'][key].values()),key
    for key,ix in (('train',tr),('validation',va)):
        part=entry[key]
        assert part['n']==len(ix)
        assert part['indices_sha256']==evidence_array_sha(ix)
        assert part['ids_sha256']==evidence_array_sha(ids[ix])
        assert part['target_sha256']==evidence_array_sha(y[ix])
    # Bind every source of the preflight's slide/pixel/provenance evidence.
    for path,record in evidence['inputs'].items():
        assert digest(path)==record['sha256'],f'Preflight input changed: {path}'
    for path in (b.META,FULL_META):
        matches=[record for name,record in evidence['inputs'].items()
                 if Path(name).resolve()==path.resolve()]
        assert len(matches)==1 and matches[0]['sha256']==digest(path)
    return evidence,entry


def matched_reference(ids,y,tr,va):
    ref=read(ARGS.matched_ssl5_root/'manifest.json')
    assert ref['architecture']==ARCH and ref['mode']=='full'
    fixed_control_contract(ref['fold'],ref['seed'],ref['recipe']['head_lr'],
                           ref['recipe']['gradient_clip'],ref['recipe']['epochs'],
                           ref['recipe']['schedule_epochs'],train_n=len(tr),
                           validation_n=len(va),split_sha=digest(ARGS.split))
    assert ref['fold']==ARGS.fold==0 and ref['seed']==ARGS.seed==42
    assert ref['train_n']==len(tr)==7151 and ref['validation_n']==len(va)==1791
    assert ref['images']==len(ids) and ref['split_sha256']==digest(ARGS.split)
    assert ref['deadline_utc']==ARGS.deadline_utc
    for key,actual in (('train_indices',tr),('validation_indices',va),
                       ('historical_validation_indices',base.historical_validation())):
        np.testing.assert_array_equal(ref[key],actual)
    assert ref['train_ids_sha']==base.ids_sha(ids[tr])
    assert ref['validation_ids_sha']==base.ids_sha(ids[va])
    assert ref['train_order_base_sha']==base.array_sha(tr)
    logs=np.log(y[tr].astype(np.float64))
    np.testing.assert_array_equal(ref['log_mean'],logs.mean(0))
    np.testing.assert_array_equal(ref['log_std'],logs.std(0))
    assert ref['raw_sha256']==EXPECTED_RAW and ref['raw_bytes']==b.RAW.stat().st_size
    assert ref['ssl_epochs']==5
    expected_recipe=dict(seed=42,epochs=20,schedule_epochs=20,encoder_lr=1e-5,
                         head_lr=1e-3,gradient_clip=1.,weight_decay=1e-4,
                         micro_batch=4,effective_batch=16,warmup_epochs=1,minimum_lr_ratio=.1)
    for key,value in expected_recipe.items():
        assert ref['recipe'][key]==value,key
    assert ref['recipe']['head_lr']==ARGS.head_lr and ref['recipe']['gradient_clip']==ARGS.clip
    assert ref['recipe']['epochs']==ARGS.epochs and ref['recipe']['schedule_epochs']==ARGS.schedule_epochs
    assert ref['execution']==dict(world_size=2,micro_batch_per_gpu=4,global_batch=16,
                                 accumulation_rounds=2,partial_groups='sample weighted; zero-weight dummy as needed')
    for source in source_files()[1:]:
        hashes=[value for name,value in ref['hashes'].items() if Path(name).name==source.name]
        assert hashes and all(value==digest(source) for value in hashes),source
    return ref


def prepare():
    base.ensure_before_deadline(ARGS.deadline_utc)
    assert_pinned_sources()
    root=ARGS.root
    if (root/'manifest.json').exists() or (root/'runs').exists():
        raise FileExistsError('No overwrite or implicit resume')
    root.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(root).free>max(12,.65*ARGS.epochs+4)*2**30
    ids,y,tr,va=base.arrays()
    evidence,entry=preflight_evidence(ids,y,tr,va)
    ref=matched_reference(ids,y,tr,va)
    assert digest(b.META)==EXPECTED_META and digest(b.WEIGHTS[ARCH])==EXPECTED_EXTERNAL
    # Actual current pixels are rehashed, not inferred from an old success marker.
    assert digest(b.RAW)==EXPECTED_RAW
    net,_=base.build_cpu(ARGS.seed)
    head_sha=prior.state_sha(net[1:].state_dict())
    assert head_sha==ref['expected_head_sha']
    encoder_sha=prior.state_sha(net[0].original.state_dict())
    del net
    m={key:copy.deepcopy(ref[key]) for key in COMMON_FIELDS}
    frozen=base.freeze_sources()
    paths=[b.META,FULL_META,b.WEIGHTS[ARCH],ARGS.split,ARGS.data_preflight,
           ARGS.matched_ssl5_root/'manifest.json',*source_files(),*frozen,
           *(Path(path) for path in evidence['inputs'])]
    m.update(created_utc=base.utc_now().isoformat(),
             initialization='SSL0_EXTERNAL_PRETRAINED',ssl_epochs=0,ssl_steps=0,
             external_checkpoint_path=str(b.WEIGHTS[ARCH]),external_checkpoint_sha=EXPECTED_EXTERNAL,
             external_encoder_tensor_sha=encoder_sha,
             additional_clear_ec_ssl=False,matched_ssl5_root=str(ARGS.matched_ssl5_root),
             matched_ssl5_manifest_sha=digest(ARGS.matched_ssl5_root/'manifest.json'),
             data_preflight_path=str(ARGS.data_preflight),data_preflight_sha256=digest(ARGS.data_preflight),
             data_preflight_evidence=dict(current_pixels_verified=True,slide_disjoint=True,
                  exact_pixel_disjoint=True,slide_exact_connected_disjoint=True,
                  train_n=len(tr),validation_n=len(va),source_sha256=evidence['source_sha256']),
             hashes={str(path):digest(path) for path in paths},
             limits=['Single model; no ensemble, TTA or calibration',
                     'External pretrained CPDINO encoder; zero additional CLEAR-EC SSL updates',
                     'Matched fold0 SSL5 control recipe; initialization is the only intervention',
                     'Slide and exact-pixel groups are disjoint; patient identity/near duplicates remain unresolved',
                     'Terminal20 primary; intermediate minima retrospective diagnostics',
                     'No exact resume; interrupted arms remain incomplete',
                     'H200 training is not T4 deployment certification'],
             torch_version=torch.__version__,cuda_version=torch.version.cuda,official_release_untouched=True)
    for forbidden in ('ssl_root','ssl_manifest_sha','ssl_checkpoint_sha','ssl_encoder_tensor_sha','ssl_seed'):
        assert forbidden not in m
    save(root/'manifest.json',m)
    print(json.dumps(dict(status='prepared',initialization=m['initialization'],root=str(root),
                          recipe=m['recipe'],external_checkpoint_sha=EXPECTED_EXTERNAL)),flush=True)


def verify():
    assert_pinned_sources()
    m=read(ARGS.root/'manifest.json')
    assert m['initialization']=='SSL0_EXTERNAL_PRETRAINED' and m['additional_clear_ec_ssl'] is False
    assert m['ssl_epochs']==0 and m['ssl_steps']==0
    assert m['external_checkpoint_path']==str(b.WEIGHTS[ARCH])
    assert m['external_checkpoint_sha']==EXPECTED_EXTERNAL
    assert m['data_preflight_path']==str(ARGS.data_preflight)
    assert m['matched_ssl5_root']==str(ARGS.matched_ssl5_root)
    assert m['data_preflight_sha256']==digest(ARGS.data_preflight)
    assert m['matched_ssl5_manifest_sha']==digest(ARGS.matched_ssl5_root/'manifest.json')
    for forbidden in ('ssl_root','ssl_manifest_sha','ssl_checkpoint_sha','ssl_encoder_tensor_sha','ssl_seed'):
        assert forbidden not in m
    for path,expected in m['hashes'].items():
        assert digest(path)==expected,f'Frozen input/source changed: {path}'
    for source in source_files():
        assert digest(source)==m['hashes'][str(ARGS.root/'source'/source.name)]
    ids,y,tr,va=base.arrays()
    preflight_evidence(ids,y,tr,va)
    ref=matched_reference(ids,y,tr,va)
    for key in COMMON_FIELDS:
        assert m[key]==ref[key],f'Matched recipe/data changed: {key}'
    return m


def setup(m):
    # b.setup is the original, pinned seed42 implementation. It obtains target
    # statistics and both optimizer LRs from m, not from its historical fold3.
    assert m['seed']==42 and m['initialization']=='SSL0_EXTERNAL_PRETRAINED'
    net,opt,mu,sigma,details=b.setup(ARCH,m)
    assert details['weight_sha256']==m['external_checkpoint_sha']
    encoder_sha=prior.state_sha(net[0].original.state_dict())
    head_sha=prior.state_sha(net[1:].state_dict())
    assert encoder_sha==m['external_encoder_tensor_sha'] and head_sha==m['expected_head_sha']
    assert all(p.requires_grad for p in net.parameters()) and not opt.state
    assert not any(isinstance(module,torch.nn.modules.batchnorm._BatchNorm) for module in net.modules())
    assert [id(p) for p in opt.param_groups[0]['params']]==[id(p) for p in net[0].parameters()]
    assert [id(p) for p in opt.param_groups[1]['params']]==[id(p) for p in net[1:].parameters()]
    assert opt.param_groups[0]['lr']==m['recipe']['encoder_lr']
    assert opt.param_groups[1]['lr']==m['recipe']['head_lr']
    assert all(group['weight_decay']==m['recipe']['weight_decay'] for group in opt.param_groups)
    torch.testing.assert_close(mu.cpu(),torch.tensor(m['log_mean'],dtype=torch.float32),rtol=0,atol=0)
    torch.testing.assert_close(sigma.cpu(),torch.tensor(m['log_std'],dtype=torch.float32),rtol=0,atol=0)
    details.update(initialization='SSL0_EXTERNAL_PRETRAINED',ssl_epochs=0,
                   external_checkpoint_path=m['external_checkpoint_path'],
                   external_checkpoint_sha=m['external_checkpoint_sha'],
                   external_encoder_tensor_sha=encoder_sha,encoder_tensor_sha=encoder_sha,
                   head_initial_tensor_sha=head_sha,projection_or_ssl_optimizer_loaded=False,
                   distributed_execution=m['execution'],mode='full',seed=42,
                   encoder_trainable=sum(p.numel() for p in net[0].parameters()),
                   head_trainable=sum(p.numel() for p in net[1:].parameters()),
                   matched_ssl5_manifest_sha=m['matched_ssl5_manifest_sha'])
    return net,opt,mu,sigma,details


def initialization_cpu_probe():
    """Read-only, manifest-independent check, usable before SSL5 is complete."""
    assert_pinned_sources()
    assert digest(b.WEIGHTS[ARCH])==EXPECTED_EXTERNAL
    # Independently compare the actual model with all external encoder tensors;
    # checking only a checkpoint filename could accidentally permit SSL weights.
    net,_=base.build_cpu(42)
    state=torch.load(b.WEIGHTS[ARCH],map_location='cpu',weights_only=True)
    encoder={key[len('encoder.'):]:value for key,value in state.items() if key.startswith('encoder.')}
    assert encoder and set(encoder)==set(net[0].original.state_dict())
    for key,value in net[0].original.state_dict().items():
        torch.testing.assert_close(value,encoder[key],rtol=0,atol=0)
    report=dict(external_encoder_all_tensors_equal=True,external_checkpoint_sha=EXPECTED_EXTERNAL,
                external_encoder_tensor_sha=prior.state_sha(encoder),
                head_initial_tensor_sha=prior.state_sha(net[1:].state_dict()),seed=42,
                encoder_tensor_count=len(encoder),initialization='SSL0_EXTERNAL_PRETRAINED')
    del net,state,encoder
    return report


def cpu_tests():
    m=verify()
    if (ARGS.root/'cpu_tests.json').exists():
        raise FileExistsError('CPU test evidence already exists')
    initial=initialization_cpu_probe()
    assert initial['external_encoder_tensor_sha']==m['external_encoder_tensor_sha']
    assert initial['head_initial_tensor_sha']==m['expected_head_sha']
    # This runs the same real head/seed, positive-target metric identity, clipped
    # gradients, schedule and partial DDP gradient tests (including the fold0 final batch n=15).
    base.cpu_tests()
    report=read(ARGS.root/'cpu_tests.json')
    report.update(initialization='SSL0_EXTERNAL_PRETRAINED',
                  external_encoder_all_tensors_equal=True,matched_ssl5_head_verified=True,
                  matched_data_recipe_verified=True,shared_pinned_training_implementation=True,
                  data_preflight_binding_verified=True)
    save(ARGS.root/'cpu_tests.json',report)


def audit():
    m=verify();root=ARGS.root
    if (root/'audit.json').exists():
        raise FileExistsError('Audit already exists')
    run=root/'runs'/f'fold{m["fold"]}'/ARCH
    result=prior.validate_result(run,digest(root/'manifest.json'))
    assert digest(run/'predictions.npz')==result['predictions_sha']
    assert len(result['history'])==20 and result['ranks_synchronized'] and result['single_model']
    details=result['details']
    assert details['initialization']=='SSL0_EXTERNAL_PRETRAINED' and details['ssl_epochs']==0
    assert details['external_encoder_tensor_sha']==m['external_encoder_tensor_sha']
    assert details['external_checkpoint_sha']==m['external_checkpoint_sha']
    assert details['head_initial_tensor_sha']==m['expected_head_sha']
    assert details['matched_ssl5_manifest_sha']==m['matched_ssl5_manifest_sha']
    ids,y,tr,va=base.arrays();updates=math.ceil(len(tr)/16);epochs=[]
    for epoch,item in enumerate(result['history'],1):
        assert item['epoch']==epoch and item['updates']==epoch*updates
        assert item['order_sha']==base.array_sha(np.random.default_rng(42+epoch-1).permutation(tr))
        prediction=run/f'predictions_epoch_{epoch:02d}.npz';weight=run/f'epoch_{epoch:02d}.pt'
        with np.load(prediction) as z:
            np.testing.assert_array_equal(z['indices'],va)
            np.testing.assert_array_equal(z['ids'],ids[va])
            np.testing.assert_array_equal(z['target'],y[va])
            score=b.metrics(z['pred'],z['target'])
            historical=base.secondary_score(z['pred'],y,va)
        for key in ('CD','CV','HEX','mean','n'):
            assert abs(score[key]-item['score'][key])<1e-6
            assert abs(historical[key]-item['historical_subset_score'][key])<1e-6
        checkpoint=torch.load(weight,map_location='cpu',weights_only=True)
        assert checkpoint['epoch']==epoch and checkpoint['manifest_sha']==digest(root/'manifest.json')
        del checkpoint
        epochs.append(dict(epoch=epoch,weights_sha=digest(weight),predictions_sha=digest(prediction),score=score))
    with np.load(run/'predictions.npz') as z,np.load(run/'predictions_epoch_20.npz') as final:
        for key in ('indices','ids','target','pred'):
            np.testing.assert_array_equal(z[key],final[key])
        score=b.metrics(z['pred'],z['target'])
    for key in ('CD','CV','HEX','mean','n'):
        assert abs(score[key]-result['score'][key])<1e-6
    checkpoint=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
    final=torch.load(run/'epoch_20.pt',map_location='cpu',weights_only=True)
    assert prior.state_sha(checkpoint['model'])==prior.state_sha(final['model'])
    assert checkpoint['epoch']==20 and checkpoint['manifest_sha']==digest(root/'manifest.json')
    best=min(result['history'],key=lambda item:item['score']['mean'])
    report=dict(status='single_model_crossfold_regression_verified',initialization='SSL0_EXTERNAL_PRETRAINED',
                ssl_epochs=0,fold=0,seed=42,manifest_sha=digest(root/'manifest.json'),score=score,
                recipe=m['recipe'],primary='fixed terminal epoch20',lower_is_better=True,
                retrospective_best=dict(epoch=best['epoch'],score=best['score'],not_primary=True),
                historical_subset_score=result['historical_subset_score'],
                external_checkpoint_sha=m['external_checkpoint_sha'],
                external_encoder_tensor_sha=m['external_encoder_tensor_sha'],
                matched_ssl5_root=m['matched_ssl5_root'],matched_ssl5_manifest_sha=m['matched_ssl5_manifest_sha'],
                data_preflight_sha256=m['data_preflight_sha256'],
                result_sha=digest(run/'result.json'),complete_sha=digest(run/'complete.json'),
                checkpoint_sha=result['checkpoint_sha'],predictions_sha=result['predictions_sha'],
                all_validation_images_scored=True,epoch_artifacts=epochs,single_model=True,limits=m['limits'])
    save(root/'audit.json',report);print(json.dumps(report),flush=True)


def configure(args):
    global ARGS
    ARGS=args
    assert_pinned_sources()
    base.ARGS=args; base.ROOT=args.root; base.SPLIT=args.split; base.SSL=None
    base.source_files=source_files
    base.EXPECTED_REUSED_SOURCES={**base.EXPECTED_REUSED_SOURCES,Path(base.__file__).name:EXPECTED_BASE}
    base.verify=verify
    base.setup=setup


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','cpu_tests','cpu-tests','smoke','train','audit'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--split',type=Path,default=Path(DEFAULT_SPLIT))
    parser.add_argument('--fold',type=int,default=0)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--head-lr',type=float,default=1e-3)
    parser.add_argument('--clip',type=float,default=1.)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--schedule-epochs',type=int,default=20)
    parser.add_argument('--deadline-utc',required=True)
    parser.add_argument('--data-preflight',type=Path,
                        default=Path(DEFAULT_DATA_PREFLIGHT))
    parser.add_argument('--matched-ssl5-root',type=Path,required=True,
                        help='Fresh fold0 SSL5 regression arm root with the matching seed42 control manifest')
    args=parser.parse_args()
    try:
        fixed_control_contract(args.fold,args.seed,args.head_lr,args.clip,args.epochs,args.schedule_epochs)
    except ValueError as error:
        parser.error(str(error))
    base.parse_deadline(args.deadline_utc)
    configure(args)
    command=args.command.replace('-','_')
    if command in ('smoke','train'):
        getattr(base,command)()
    else:
        globals()[command]()


if __name__=='__main__':
    main()
