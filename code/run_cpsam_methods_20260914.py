"""Bounded, matched CPSAM DINO/I-JEPA comparison. All data/compute stay on K8s.

Stage 1: two SSL learning rates per method, fold2 ridge screening.
Stage 2: selected SSL checkpoint per method, original full regression on folds2/3.
Stage 3: optional all-data research candidate if the prespecified gate passes.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path('experiments/cpsam_methods_20260914_v2')
PARENT = Path('experiments/raw_ssl_next_20260913')
sys.path.insert(0, str(PARENT/'source'))
import next_candidate_20260913 as old
from cpsam_ssl_methods import train_ssl, token_forward

TRIALS = {f'{method}_{rate}':dict(method=method, lr=lr, head_multiplier=100 if method=='dino' else 10)
    for method in ('dino', 'ijepa') for rate, lr in (('low', 5e-6), ('high', 1e-5))}


def verify():
    m = old.load(ROOT/'manifest.json')
    for group in ('source_sha256', 'input_sha256'):
        for path, sha in m[group].items():
            assert old.sha(path) == sha, path
    return m


def data():
    with np.load(old.ALLDATA/'meta.npz') as d:
        ids, y = d['ids'].astype(str), d['Y'].astype(np.float32)
    with np.load('datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True) as d:
        old_ids = d['ids'].astype(str)
    lookup = {s:i for i,s in enumerate(ids)}
    mapping = np.array([lookup[s] for s in old_ids])
    return ids, y, mapping


def aggregate(pred, y):
    if pred.shape != y.shape or not np.isfinite(pred).all() or not np.isfinite(y).all():
        raise ValueError('Invalid predictions/targets')
    values, counts = [], []
    for j in range(3):
        valid = y[:, j] > 0
        if not valid.any():
            raise ValueError('No valid targets')
        values.append(float((np.abs(pred[valid, j]-y[valid, j])/y[valid, j]).mean()*100))
        counts.append(int(valid.sum()))
    return dict(CD_CV_HEX=values, mean=float(np.mean(values)), valid_counts=counts, n=len(y))


def prepare():
    assert not (ROOT/'manifest.json').exists()
    previous = old.verify()
    sources = {str(p):old.sha(p) for p in (ROOT/'source').glob('*.py')}
    for directory in (ROOT/'upstream/dino', ROOT/'upstream/ijepa/src'):
        sources.update({str(p):old.sha(p) for p in directory.rglob('*.py')})
    inputs = dict(previous['input_sha256'])
    inputs['cpsam_models/cpsam_v2'] = old.sha('cpsam_models/cpsam_v2')
    inputs[str(PARENT/'manifest.json')] = old.sha(PARENT/'manifest.json')
    # Bind downstream trainer and view loader, including their frozen dependencies.
    sources.update(previous['source_sha256'])
    for directory in ('experiments/crop_dino_screen_20260910/source/frozen',
                      'experiments/specialize_screen_20260908/source/frozen',
                      'experiments/region_search_20260905/source/code'):
        sources.update({str(p):old.sha(p) for p in Path(directory).glob('*.py')})
    shared = old.shared_api()
    net = shared.build_model('cpsam_models/cpsam_v2', device='cpu')
    e = net.encoder
    assert e.pos_embed.shape == (1, 32, 32, 1024)
    assert len(e.blocks) == 24 and all(b.window_size == 0 for b in e.blocks)
    assert all(b.attn.use_rel_pos for b in e.blocks)
    del net; gc.collect()
    # Regression metric test: valid entries in other columns must survive a zero.
    test_y = np.array([[1., 0., 4.], [2., 2., 2.]])
    test_p = np.array([[2., 9., 4.], [2., 1., 4.]])
    score = aggregate(test_p, test_y)
    np.testing.assert_allclose(score['CD_CV_HEX'], [50, 50, 50])
    assert score['valid_counts'] == [2, 1, 2]
    sys.path.insert(0, 'experiments/deploy_region_20260907/source')
    from train_deploy_region import build_regions
    views = build_regions('full_quality')
    assert views.images.shape[0] == 9000
    ids, y, mapping = data()
    folds = {str(f):previous['folds'][str(f)] for f in (2,3)}
    for f, r in folds.items():
        with np.load(r['split_path']) as d:
            tr, va = mapping[d['train']], mapping[d['validation']]
            assert not set(tr) & set(va)
            assert (y[tr] > 0).all()
            assert r['slide_and_exact_duplicate_overlap'] == 0
        for name in ('selected.pt', 'oof.npz', 'result.json', 'config.json'):
            path = Path(r['control_root'])/name
            inputs[str(path)] = old.sha(path)
    manifest = dict(created_at=time.strftime('%FT%T%z'), source_sha256=sources, input_sha256=inputs,
        folds=folds, trials=TRIALS, backbone='existing CPSAM ViT-L/8, 24x1024, 304542720 encoder parameters',
        initialization='same original cpsam_models/cpsam_v2 as current matched single-model baseline',
        no_official_dino_or_ijepa_backbone=True, ssl_steps=6000, ssl_batch=32, ssl_images=9000,
        ssl_labels_used=False, transductive=True, independent_validation=False,
        frozen_probe='pre-neck mean 1024, 5 native256 views; train-standardized ridge alpha10; fold2 only',
        selection='Per-method minimum fold2 ridge mean; then fixed full downstream folds2/3. Fold3 never tunes a setting.',
        regression='existing CandidateNet, raw targets, seed77, K2/batch48, encoder1e-5/head1e-3, fixed5215 updates, 40-view evaluation',
        gate='Choose method by full downstream fold2 mean only. Require >=0.03 better pooled mean vs existing standard SSL and both folds better than both controls before all9000 candidate fit.',
        gpu_stage1_count=4, gpu_stage1_max_seconds=18000, gpu_downstream_count=4,
        gpu_downstream_max_seconds=25200, gpu_final_count=1, gpu_final_max_seconds=25200,
        max_total_launched_gpu_hours=55, no_automatic_retry=True, no_submission=True,
        student_inference_backbones=1, teacher_and_predictor_training_only=True,
        limitations=['Historically reused folds','All internal validation images seen without labels during SSL',
                      'Two learning rates is a bounded search, not a globally optimal setting',
                      'Equal image exposures/updates do not imply equal FLOPs for DINO and I-JEPA'])
    old.save(ROOT/'manifest.json', manifest)
    print(json.dumps(dict(preflight_passed=True, encoder_parameters=304542720, trials=TRIALS, zero_target_test=True)), flush=True)


def fresh_encoder(shared, checkpoint_path=None):
    shared.seed(77)
    net = shared.build_model('cpsam_models/cpsam_v2')
    if checkpoint_path:
        ck = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        net.encoder.load_state_dict(ck['teacher_encoder'], strict=True)
    return net


def check_real_encoder(net, views):
    # Compare the algorithms in full FP32. TF32 may round differently for the
    # equivalent dense and gathered einsum shapes; don't weaken the tolerance.
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    x=torch.from_numpy(views.tile(0,views.selected[0,0],0)[None]).cuda()
    with torch.no_grad():
        original=net.encoder(x)
        tokens=token_forward(net.encoder,x)
        adapted=net.encoder.neck(tokens.reshape(1,32,32,1024).permute(0,3,1,2))
        delta=float((original-adapted).abs().max())
        torch.testing.assert_close(original,adapted,rtol=1e-4,atol=2e-5)
    old.setup_torch()
    print(dict(real_encoder_fp32_max_abs=delta,passed=True),flush=True)
    return delta


def gpu_preflight():
    m=verify();old.setup_torch();shared=old.shared_api()
    sys.path.insert(0,'experiments/deploy_region_20260907/source')
    from train_deploy_region import build_regions
    views=build_regions('full_quality');results={}
    for method in ('dino','ijepa'):
        net=fresh_encoder(shared);delta=check_real_encoder(net,views)
        config=dict(**TRIALS[method+'_high'],batch=32,steps=4,smoke=True,all9000=True)
        result=train_ssl(net.encoder,views,ROOT/'gpu_preflight'/method,config,old.save,old.sha,ROOT/'upstream')
        results[method]=dict(**result,equivalence_max_abs=delta)
        del net;gc.collect();torch.cuda.empty_cache()
    old.save(ROOT/'gpu_preflight_complete.json',dict(results=results,at=time.strftime('%FT%T%z')))


@torch.no_grad()
def extract(encoder, views, out):
    encoder.eval(); values=[]; started=time.monotonic()
    for i in range(9000):
        xs = torch.from_numpy(views.evaluate(i, False)).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            f = token_forward(encoder, xs).float().mean((0,1))
        values.append(f.cpu().numpy())
        if (i+1)%250 == 0:
            old.save(out/'feature_progress.json', dict(stage='features', images=i+1, total=9000,
                elapsed_seconds=time.monotonic()-started, at=time.strftime('%FT%T%z')))
    x = np.stack(values)
    assert x.shape == (9000, 1024) and np.isfinite(x).all()
    np.save(out/'features.npy', x)
    return x


def ridge_fold2(x, manifest):
    ids, y, mapping = data()
    r = manifest['folds']['2']
    with np.load(r['split_path']) as d:
        tr, va = mapping[d['train']], mapping[d['validation']]
    with np.load(Path(r['reference_root'])/'preflight/reference.npz') as d:
        keep = d['primary'][d['folds']==2]
        np.testing.assert_array_equal(ids[va], d['ids'][d['folds']==2].astype(str))
    a, b = x[tr].astype(float), x[va].astype(float)
    mean, sd = a.mean(0), a.std(0).clip(.01)
    a, b = (a-mean)/sd, (b-mean)/sd
    z = np.log(y[tr].astype(float)); zm = z.mean(0)
    w = np.linalg.solve(a.T@a+10*np.eye(a.shape[1]), a.T@(z-zm))
    pred = np.exp(np.clip(b@w+zm, -20, 20))
    return dict(**aggregate(pred[keep], y[va][keep]), fold=2, alpha=10,
        transductive=True, full_finetune=False, used_for_hpo=True)


def ssl_trial(tag):
    m = verify(); old.setup_torch(); shared = old.shared_api()
    sys.path.insert(0, 'experiments/deploy_region_20260907/source')
    from train_deploy_region import build_regions
    views = build_regions('full_quality')
    out = ROOT/'ssl'/tag; out.mkdir(parents=True, exist_ok=False)
    config = dict(**TRIALS[tag], batch=32, steps=6000, smoke=False,
        all9000=True, initial_sha256=m['input_sha256']['cpsam_models/cpsam_v2'],
        neck_frozen=True, original_relative_positions_preserved=True, no_pixel_resize=True)
    net = fresh_encoder(shared)
    # Full real-encoder equivalence once on GPU, including its unchanged neck.
    check_real_encoder(net, views)
    smoke = train_ssl(net.encoder, views, out/'smoke', dict(config, steps=4, smoke=True), old.save, old.sha, ROOT/'upstream')
    assert smoke['strict_reload']
    del net; gc.collect(); torch.cuda.empty_cache()
    old.save(out/'smoke_complete.json', dict(passed=True, at=time.strftime('%FT%T%z')))
    net = fresh_encoder(shared)
    result = train_ssl(net.encoder, views, out/'pretrain', config, old.save, old.sha, ROOT/'upstream')
    del net; gc.collect(); torch.cuda.empty_cache()
    net = fresh_encoder(shared, result['checkpoint'])
    x = extract(net.encoder, views, out)
    report = ridge_fold2(x, m)
    old.save(out/'probe.json', report)
    old.save(out/'complete.json', dict(ssl=result, probe=report, feature_sha256=old.sha(out/'features.npy'),
        finished_at=time.strftime('%FT%T%z')))


def select():
    m = verify(); out = ROOT/'selection.json'
    assert not out.exists(), 'Selection must be frozen once'
    scores = {}
    order = set()
    for tag in TRIALS:
        r = old.load(ROOT/'ssl'/tag/'complete.json')
        assert r['ssl']['steps']==6000 and r['ssl']['unique_images_seen']==9000
        assert old.sha(r['ssl']['checkpoint'])==r['ssl']['checkpoint_sha256']
        assert old.sha(ROOT/'ssl'/tag/'features.npy')==r['feature_sha256']
        scores[tag] = r['probe']['mean']; order.add(r['ssl']['sample_order_sha256'])
    assert len(order)==1, 'Image exposure must be matched across methods/rates'
    chosen = {method:min((t for t in TRIALS if t.startswith(method+'_')), key=lambda t:(scores[t],t)) for method in ('dino','ijepa')}
    old.save(out, dict(chosen=chosen, fold2_probe_scores=scores, criterion='fold2 only, mean of per-metric MAPE',
        source_manifest_sha256=old.sha(ROOT/'manifest.json'), at=time.strftime('%FT%T%z')))
    print(out.read_text(), flush=True)


def downstream(method, fold, smoke=False, final=False):
    m=verify(); old.setup_torch(); shared=old.shared_api()
    chosen=old.load(ROOT/'selection.json')['chosen'][method]
    ssl=old.load(ROOT/'ssl'/chosen/'complete.json')['ssl']
    assert old.sha(ssl['checkpoint'])==ssl['checkpoint_sha256']
    if final:
        gate=old.load(ROOT/'comparison.json')
        assert gate['update_gate_passed'] and gate['chosen_by_fold2']==method
        sys.path.insert(0,'experiments/deploy_region_20260907/source')
        from train_deploy_region import build_regions
        views=build_regions('full_quality');ids,y,_=data()
        tr=np.flatnonzero((y>0).all(1));va=None;updates=5215
        out=ROOT/'candidate'/method
        assert len(tr)==8998
    else:
        r=m['folds'][str(fold)]
        with np.load('datasets/clear_ec_official/cache_native_meta_512.npz',allow_pickle=True) as d:
            ids,y=d['ids'].astype(str),d['Y'].astype(np.float32)
        with np.load(r['split_path']) as d:tr,va=d['train'].copy(),d['validation'].copy()
        images=np.memmap('datasets/clear_ec_official/img_native_512.dat',mode='r',dtype='uint8',shape=(8954,512,512))
        views=shared.NativeRegionViews(images,ids,'full_quality');updates=r['updates']
        out=ROOT/('downstream_smoke' if smoke else 'downstream')/f'{method}_fold{fold}'
    net=fresh_encoder(shared,ssl['checkpoint'])
    for block in net.encoder.blocks:
        fn=block.forward
        block.forward=lambda x,fn=fn:torch.utils.checkpoint.checkpoint(fn,x,use_reentrant=False)
    cfg=old.config_for(shared,2 if smoke else updates,smoke,method=method,fold=fold,
        ssl_tag=chosen,ssl_checkpoint_sha256=ssl['checkpoint_sha256'],ssl_all9000=True,
        transductive=True,encoder_lr=1e-5,head_lr=1e-3,initialization='CPSAM updated by '+method,
        train_rows=len(tr),validation_rows=0 if va is None else len(va),single_model=True)
    result=old.fit(shared,net,views,tr,y,out,cfg,1e-5,None if va is None else (va[:4] if smoke else va),ids)
    if not final and not smoke:
        raw=old.load(Path(m['folds'][str(fold)]['control_root'])/'result.json')
        assert result['training']['first_three_batches_sha256']==raw['training']['first_three_batches_sha256']
    old.save(out/'complete.json',dict(**result, finished_at=time.strftime('%FT%T%z')))


def audit():
    m=verify();targets=[];groups=[];predictions={k:[] for k in ('raw','standard_ssl','dino','ijepa')};hashes={}
    for fold in (2,3):
        r=m['folds'][str(fold)]
        with np.load(Path(r['reference_root'])/'preflight/reference.npz') as ref:
            mask=ref['folds']==fold;keep=ref['primary'][mask]
            targets.append(ref['target'][mask][keep]);groups.extend([fold]*int(keep.sum()))
            paths={'raw':Path(r['control_root']),'standard_ssl':PARENT/'ssl_runs'/f'fold{fold}'/'standard',
                **{s:ROOT/'downstream'/f'{s}_fold{fold}' for s in ('dino','ijepa')}}
            for method,path in paths.items():
                result=old.load(path/'result.json')
                assert old.sha(path/'selected.pt')==result['checkpoint_sha256']
                assert result['strict_reload_passed']
                with np.load(path/'oof.npz') as d:
                    for key in ('ids','indices','target'):np.testing.assert_array_equal(d[key],ref[key][mask])
                    predictions[method].append(d['pred'][keep].astype(float))
                for name in ('selected.pt','oof.npz','result.json','config.json'):
                    hashes[str(path/name)]=old.sha(path/name)
    y=np.concatenate(targets);g=np.array(groups)
    reports={}
    for method,parts in predictions.items():
        p=np.concatenate(parts)
        reports[method]=dict(overall=aggregate(p,y),per_fold={str(f):aggregate(p[g==f],y[g==f]) for f in (2,3)},
            ssl_validation_images_seen=method in ('dino','ijepa'))
    chosen=min(('dino','ijepa'),key=lambda s:(reports[s]['per_fold']['2']['mean'],s))
    a,b=reports[chosen],reports['standard_ssl']
    passed=(a['overall']['mean']<=b['overall']['mean']-.03 and
        all(a['per_fold'][str(f)]['mean']<reports[ref]['per_fold'][str(f)]['mean']
            for f in (2,3) for ref in ('raw','standard_ssl')))
    result=dict(reports=reports,chosen_by_fold2=chosen,update_gate_passed=passed,input_sha256=hashes,
        n=len(y),inference_backbones=1,transductive=True,independent_validation=False,
        standard_ssl_is_historical_fold_clean=True,official_score_claim=False,
        note='New SSL saw all validation images without labels. Differences vs fold-clean controls include that data exposure.',
        finished_at=time.strftime('%FT%T%z'))
    old.save(ROOT/'comparison.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','gpu-preflight','ssl','select','downstream','downstream-run','audit','final-fit'))
    p.add_argument('--tag',choices=TRIALS);p.add_argument('--method',choices=('dino','ijepa'))
    p.add_argument('--fold',type=int,choices=(2,3));p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(4)
    def interrupted(sig, frame):
        raise TimeoutError(f'Job interrupted by signal {sig}; no automatic retry')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if a.command=='gpu-preflight':gpu_preflight()
        elif a.command=='ssl':ssl_trial(a.tag)
        elif a.command=='downstream':downstream(a.method,a.fold,a.smoke)
        elif a.command=='downstream-run':
            downstream(a.method,a.fold,True)
            gc.collect();torch.cuda.empty_cache()
            downstream(a.method,a.fold,False)
        elif a.command=='final-fit':downstream(a.method,None,final=True)
        else:globals()[a.command]()
    except Exception as error:
        errors=ROOT/'failures';errors.mkdir(exist_ok=True)
        name='_'.join(str(x) for x in (a.command,a.tag,a.method,a.fold) if x is not None)
        old.save(errors/(name+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),at=time.strftime('%FT%T%z')))
        raise
