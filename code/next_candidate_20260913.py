"""Separate raw-label release preparation and bounded SSL learning-rate screen.

All compute and data access run on Kubernetes. Existing artifacts stay immutable.
Only the raw all-data checkpoint may enter the user-requested new model archive.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

import numpy as np

ROOT = Path('experiments/raw_ssl_next_20260913')
PARENT = Path('experiments/target_screen_20260912')
REPLICA = Path('experiments/overnight_target_20260912')
FOLD4 = Path('experiments/target_fold4_20260912')
OLD_STAGE = Path('experiments/region_candidate_20260907/experimental_stage')
ALLDATA = Path('datasets/clear_ec_final9000_20260904')
FRAME = Path('datasets/clear_ec_fullframe_20260904')
RAW_NAME = 'full_quality_seed77.pt'
RATES = {'standard': 1e-5, 'half_lr': 5e-6}
FOLDS = (2, 3, 4)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def shared_api():
    sys.path.insert(0, 'experiments/crop_dino_screen_20260910/source/frozen')
    import train_specialize_screen
    return train_specialize_screen


def source_for(fold):
    return REPLICA if fold in (2, 3) else FOLD4


def verify():
    manifest = load(ROOT/'manifest.json')
    for group in ('source_sha256', 'input_sha256'):
        for path, expected in manifest[group].items():
            assert sha(path) == expected, path
    return manifest


def prepare():
    assert not (ROOT/'manifest.json').exists()
    sources, inputs, records = {}, {}, {}
    for parent in (PARENT, REPLICA, FOLD4):
        assert load(parent/'queue_finished.json')['audit_completed']
        assert not load(parent/'complete.json')['failed']
        m = load(parent/'manifest.json')
        for p, h in m['source_sha256'].items():
            assert sha(p) == h, p
            sources[p] = h
        for name in ('manifest.json', 'screen_audit.json', 'preflight/reference.npz'):
            inputs[str(parent/name)] = sha(parent/name)
    extra = list((ROOT/'source').glob('*.py')) + list((ROOT/'app').glob('*.py'))
    extra += list(Path('experiments/deploy_region_20260907/source').glob('*.py'))
    for p in extra:
        assert not p.name.startswith('._')
        sources[str(p)] = sha(p)
    # Recheck every cached native image against the earlier frozen fingerprint.
    fingerprints = Path('experiments/ssl_screen_20260911/preflight/pixel_fingerprints.npz')
    assert sha(fingerprints) == load(PARENT/'manifest.json')['input_sha256'][str(fingerprints)]
    with np.load(fingerprints) as d:
        hashes = d['sha256'].copy()
    fm = load(FRAME/'fullframe_manifest.json')
    images = np.memmap(FRAME/'img_full_raw.dat', mode='r', dtype='uint8', shape=tuple(fm['cache_shape']))
    assert len(images) == len(hashes) == 8954
    for image, expected in zip(images, hashes):
        assert hashlib.sha256(image.tobytes()).hexdigest() == expected
    del images
    for p in (fingerprints, ALLDATA/'meta.npz', ALLDATA/'manifest.json', FRAME/'fullframe_manifest.json',
              FRAME/'fullframe_meta.npz', FRAME/'context_quality.npz', FRAME/'context_views_manifest.json',
              FRAME/'source_metadata/final_train_ids_9000_reconciled.csv',
              Path('datasets/clear_ec_official/cache_native_meta_512.npz'),
              Path('datasets/clear_ec_official/final_train_ids.csv')):
        inputs[str(p)] = sha(p)
    with np.load(ALLDATA/'meta.npz') as d:
        assert len(d['ids']) == len(set(d['ids'])) == 9000
        np.testing.assert_array_equal(d['train_indices'], np.flatnonzero((d['Y']>0).all(1)))
        assert len(d['train_indices']) == 8998 and np.isfinite(d['Y']).all()
    assert sha(FRAME/'source_metadata/final_train_ids_9000_reconciled.csv') == load(ALLDATA/'manifest.json')['source_csv_sha256']
    # The old model inventory binds the five protected members and the foundation.
    inventory = load(ROOT/'runtime_setup/provenance/model_archive.json')
    for name, row in inventory['members'].items():
        p = OLD_STAGE/'model'/name
        assert sha(p) == row['sha256'] and p.stat().st_size == row['bytes'], name
        inputs[str(p)] = row['sha256']
    assert sha('cpsam_models/cpsam_v2') == inventory['members']['cpsam_v2']['sha256']
    inputs['cpsam_models/cpsam_v2'] = inventory['members']['cpsam_v2']['sha256']
    inputs[str(ROOT/'runtime_setup/provenance/model_archive.json')] = sha(ROOT/'runtime_setup/provenance/model_archive.json')
    cfg = load(OLD_STAGE/'model/candidate6_full_quality.json')
    assert cfg['members'][-1] == RAW_NAME and len(cfg['members']) == 6
    for fold in FOLDS:
        parent = source_for(fold)
        row = load(parent/'manifest.json')['folds'][str(fold)]
        audit = load(parent/'screen_audit.json')
        for name in ('config.json', 'selected.pt', 'oof.npz', 'result.json', 'split_indices.npz'):
            p = parent/'runs'/f'raw_fold{fold}'/name
            assert sha(p) == audit['input_sha256'][str(p)], str(p)
            # Checkpoint hashes are verified here, not redundantly read by every fit.
            if name != 'selected.pt': inputs[str(p)] = audit['input_sha256'][str(p)]
        inputs[row['split_path']] = sha(row['split_path'])
        records[str(fold)] = dict(row, reference_root=str(parent),
            control_root=str(parent/'runs'/f'raw_fold{fold}'))
    pre = ROOT/'preflight'; pre.mkdir(exist_ok=False)
    # Validate all-data view mapping and its 46 reconciled images without a GPU.
    sys.path.insert(0, 'experiments/deploy_region_20260907/source')
    from train_deploy_region import build_regions, check_head_construction
    check_head_construction()
    views = build_regions('full_quality'); del views
    manifest = dict(created_at=time.strftime('%FT%T%z'), source_sha256=sources, input_sha256=inputs,
        folds=records, raw_updates=5215, seed=77, raw_train_rows=8998,
        raw_step_rule='Median of fixed terminal raw CV updates [5180,5215,5215,5215,5180]; not validation-best selection',
        raw_batch_rule='Same current raw CV loop: batch48, drop partial minibatch, fixed5215 optimizer updates; not35 all-data epochs',
        raw_calibration='Preserve original candidate6 calibration, equal weights and filename interface; no new fitting',
        ssl_updates=6000, learning_rates=RATES, ssl_folds=list(FOLDS),
        ssl_design='Same fold-clean native SSL followed by two fresh raw-target fits: original vs half encoder/head LR; no scale resizing',
        ssl_gate='Research only: half_lr beats standard and base raw on each of folds2,3,4 and by0.01 mean MAPE in fixed-six aggregate; report all variants',
        no_automatic_ssl_promotion=True, no_automatic_submission=True,
        old_members=inventory['members'], max_raw_seconds=7*3600, max_ssl_seconds=13*3600,
        runtime_python=str(ROOT/'runtime_setup/.venv/bin/python'),
        caveat='Historically reused CV. Packaging request is not proof of score improvement; previous0.02 ensemble gate was not passed.')
    save(ROOT/'manifest.json', manifest)
    print(json.dumps(dict(preflight_passed=True, raw_train_rows=8998, ssl_folds=records,
                         source_files=len(sources), raw_updates=5215)), flush=True)


def setup_torch():
    import torch, cv2
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    return torch


def fresh_net(shared, encoder=None):
    import torch
    import torch.utils.checkpoint as checkpoint
    shared.seed(77)
    net = shared.build_model('cpsam_models/cpsam_v2')
    if encoder is not None:
        ck = torch.load(encoder, map_location='cpu', weights_only=True)
        assert ck['config']['validation_images_used'] == 0
        net.encoder.load_state_dict(ck['teacher_encoder'], strict=True)
        del ck
    for block in net.encoder.blocks:
        original = block.forward
        block.forward = lambda x, fn=original: checkpoint.checkpoint(fn, x, use_reentrant=False)
    return net


def fit(shared, net, views, tr, y, out, config, lr, validation=None, ids=None):
    import torch
    out.mkdir(parents=True, exist_ok=False)
    save(out/'config.json', config)
    np.savez_compressed(out/'split_indices.npz', train=tr,
                        validation=np.array([], dtype=np.int64) if validation is None else validation)
    z = torch.tensor(((np.log(y.clip(1e-6))-shared.LM)/shared.LS).astype(np.float32), device='cuda')
    training = shared.train(net, views, tr, z, None, config['updates'], out, lr, 77)
    torch.save(dict(model=net.state_dict(), config=config), out/'selected.pt')
    ck = torch.load(out/'selected.pt', map_location='cpu', weights_only=True)
    net.load_state_dict(ck['model'], strict=True); del ck, z
    evaluation = None
    if validation is not None:
        evaluation = shared.evaluate(net, views, validation, y, ids, out)
    result = dict(completed_at=time.strftime('%FT%T%z'), training=training, evaluation=evaluation,
                  checkpoint_sha256=sha(out/'selected.pt'), strict_reload_passed=True,
                  deployment_approved=False, smoke=config['smoke'])
    save(out/'result.json', result)
    return result


def config_for(shared, updates, smoke, **extra):
    return dict(seed=77, updates=updates, smoke=smoke, batch_size=48, k_crops=2,
        pool='mean', feature_agg=0, bag_pool='mean', objective='mixed_mape',
        region_policy='full_quality', target_policy='raw', lm=shared.LM.tolist(), ls=shared.LS.tolist(),
        select_mode='fixed_terminal_no_validation_selection', **extra)


def raw_train(smoke):
    manifest = verify(); torch = setup_torch(); shared = shared_api()
    sys.path.insert(0, 'experiments/deploy_region_20260907/source')
    from train_deploy_region import build_regions
    views = build_regions('full_quality')
    with np.load(ALLDATA/'meta.npz') as d:
        ids, y, tr = d['ids'].copy(), d['Y'].copy(), d['train_indices'].copy()
    assert len(tr) == 8998 and (y[tr]>0).all()
    net = fresh_net(shared)
    cfg = config_for(shared, 2 if smoke else manifest['raw_updates'], smoke,
        train_rows=8998, validation_rows=0, encoder_lr=1e-5, head_lr=1e-3,
        initialization='original_cpsam_no_new_ssl', partial_batch='drop',
        raw_label_metadata_sha256=sha(ALLDATA/'meta.npz'),
        initial_sha256=manifest['old_members']['cpsam_v2']['sha256'])
    out = ROOT/('raw_smoke' if smoke else 'raw_deploy')
    result = fit(shared, net, views, tr, y, out, cfg, 1e-5,
                 tr[:4] if smoke else None, ids)
    print(json.dumps(dict(raw_fit_complete=True, smoke=smoke, training=result['training'])), flush=True)


def ssl_worker(fold, smoke):
    manifest = verify(); torch = setup_torch(); shared = shared_api()
    record = manifest['folds'][str(fold)]
    # Read the same metadata and duplicate-clean split as the completed raw controls.
    with np.load('datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True) as d:
        ids, y = d['ids'].astype(str), d['Y'].astype(np.float32)
    with np.load(record['split_path']) as d:
        tr, va = d['train'].copy(), d['validation'].copy()
    assert len(tr) == record['train_rows'] and not set(tr) & set(va)
    image = np.memmap('datasets/clear_ec_official/img_native_512.dat', mode='r', dtype='uint8', shape=(8954,512,512))
    views = shared.NativeRegionViews(image, ids, 'full_quality')
    out = ROOT/('ssl_smoke' if smoke else 'ssl_runs')/f'fold{fold}'
    out.mkdir(parents=True, exist_ok=False)
    pre = out/'pretrain'; pre.mkdir()
    sys.path.insert(0, 'experiments/ssl_screen_20260911/source')
    from ssl_native_core import pretrain
    shared.seed(77); net = shared.build_model('cpsam_models/cpsam_v2')
    ssl = pretrain(net.encoder, views, tr, pre, 4 if smoke else manifest['ssl_updates'], shared)
    del net; gc.collect(); torch.cuda.empty_cache()
    results = {}
    for variant, lr in RATES.items():
        save(out/'status.json', dict(stage='downstream', variant=variant, at=time.strftime('%FT%T%z')))
        net = fresh_net(shared, ssl['final_checkpoint'])
        cfg = config_for(shared, 2 if smoke else record['updates'], smoke,
            fold=fold, variant=variant, encoder_lr=lr, head_lr=lr*100,
            train_rows=len(tr), validation_rows=len(va), split_sha256=record['split_sha256'],
            ssl_checkpoint_sha256=ssl['checkpoint_sha256'], ssl_updates=4 if smoke else 6000)
        result = fit(shared, net, views, tr, y, out/variant, cfg, lr, va[:4] if smoke else va, ids)
        if not smoke:
            control = load(Path(record['control_root'])/'result.json')
            assert result['training']['first_three_batches_sha256'] == control['training']['first_three_batches_sha256']
        results[variant] = result['checkpoint_sha256']
        del net; gc.collect(); torch.cuda.empty_cache()
    a, b = [load(out/v/'result.json')['training']['first_three_batches_sha256'] for v in RATES]
    assert a == b
    save(out/'worker_complete.json', dict(fold=fold, variants=results, smoke=smoke,
                                        finished_at=time.strftime('%FT%T%z')))


def run_children(role):
    manifest = verify()
    lock = (ROOT/f'{role}.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/f'{role}_status.json').exists(), 'Do not restart partial queues'
    import torch
    assert torch.cuda.device_count() == (1 if role=='raw' else 3)
    logdir = ROOT/f'{role}_logs'; logdir.mkdir(exist_ok=False)
    active = {}; started = time.monotonic(); completed = []; failed = {}
    deadline = started + manifest[f'max_{role}_seconds']
    def stop(signum=None, frame=None):
        for item in active.values():
            p=item['p']
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        if signum:
            save(ROOT/f'{role}_interrupted.json',dict(signal=signum,at=time.strftime('%FT%T%z')))
            raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    def launch(gpu, smoke):
        used=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
        assert used<100, 'Assigned GPU unexpectedly occupied'
        name = 'raw' if role=='raw' else f'fold{FOLDS[gpu]}'
        command=[sys.executable,str(ROOT/'source/next_candidate_20260913.py'),'raw-train' if role=='raw' else 'ssl-worker']
        if role=='ssl':command += ['--fold',str(FOLDS[gpu])]
        if smoke:command.append('--smoke')
        log=(logdir/f'{name}_{"smoke" if smoke else "full"}.log').open('x')
        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        log.close();active[gpu]=dict(p=p,name=name,smoke=smoke,start=time.monotonic())
        print(json.dumps(dict(launched=name,gpu=gpu,smoke=smoke)),flush=True)
    try:
        for gpu in range(1 if role=='raw' else 3):launch(gpu,True)
        while active:
            for gpu,item in list(active.items()):
                rc=item['p'].poll()
                if rc is None:
                    if time.monotonic()>deadline or (item['smoke'] and time.monotonic()-item['start']>1800):
                        raise TimeoutError(item['name'])
                    continue
                del active[gpu]
                if rc:raise RuntimeError(f'{item["name"]} exit={rc}')
                marker = ROOT/('raw_smoke' if item['smoke'] else 'raw_deploy')/'result.json' if role=='raw' else ROOT/('ssl_smoke' if item['smoke'] else 'ssl_runs')/item['name']/'worker_complete.json'
                assert marker.is_file()
                if item['smoke']:launch(gpu,False)
                else:completed.append(item['name'])
            save(ROOT/f'{role}_status.json',dict(at=time.strftime('%FT%T%z'),done=completed,failed=failed,
                active={str(g):dict(name=v['name'],pid=v['p'].pid,smoke=v['smoke']) for g,v in active.items()},
                remaining_seconds=max(0,deadline-time.monotonic())))
            if active:time.sleep(10)
        if role=='raw':
            subprocess.run([sys.executable,str(ROOT/'source/next_candidate_20260913.py'),'stage'],check=True,timeout=900)
            # Preserve the venv interpreter symlink path; resolving it selects base Python.
            subprocess.run([os.path.abspath(manifest['runtime_python']),str(ROOT/'source/next_candidate_20260913.py'),'runtime'],check=True,timeout=1800)
            save(ROOT/'raw_ready_for_archive.json',dict(runtime_passed=True,at=time.strftime('%FT%T%z')))
        else:
            subprocess.run([sys.executable,str(ROOT/'source/next_candidate_20260913.py'),'ssl-audit'],check=True,timeout=1200)
    except Exception as e:
        failed['controller']=repr(e);stop()
    save(ROOT/f'{role}_complete.json',dict(done=completed,failed=failed,
        elapsed_seconds=time.monotonic()-started,finished_at=time.strftime('%FT%T%z')))
    if failed:raise RuntimeError(failed)


def stage():
    manifest=verify(); run=ROOT/'raw_deploy'; result=load(run/'result.json');cfg=load(run/'config.json')
    assert not cfg['smoke'] and cfg['target_policy']=='raw' and cfg['train_rows']==8998
    assert cfg['updates']==5215 and result['strict_reload_passed']
    assert sha(run/'selected.pt')==result['checkpoint_sha256']
    dest=ROOT/'model'; dest.mkdir(exist_ok=False); files={}
    for name,row in manifest['old_members'].items():
        src=run/'selected.pt' if name==RAW_NAME else OLD_STAGE/'model'/name
        expected=result['checkpoint_sha256'] if name==RAW_NAME else row['sha256']
        assert sha(src)==expected
        shutil.copyfile(src,dest/name)
        assert sha(dest/name)==expected
        files[name]=dict(bytes=(dest/name).stat().st_size,sha256=expected)
    assert files[RAW_NAME]['sha256'] != manifest['old_members'][RAW_NAME]['sha256']
    save(ROOT/'stage_manifest.json',dict(files=files,only_changed_model=RAW_NAME,
         existing_image_compatible_by_interface=True,calibration_unchanged=True,automatic_submission=False))


def runtime():
    manifest=verify()
    import torch, SimpleITK as sitk
    assert torch.__version__.split('+')[0]=='2.11.0' and torch.version.cuda=='12.6'
    assert torch.cuda.is_available()
    sys.path.insert(0,str(ROOT/'app'))
    import inference
    stage=load(ROOT/'stage_manifest.json')
    for name,row in stage['files'].items():assert sha(ROOT/'model'/name)==row['sha256']
    out=ROOT/'runtime_check';out.mkdir(exist_ok=False)
    with np.load(ALLDATA/'meta.npz') as d:ids=d['ids'].copy()
    records=[]
    for index in np.linspace(0,8999,8,dtype=int):
        image=sitk.GetArrayFromImage(sitk.ReadImage(str(FRAME/'train_mha'/(str(ids[index])+'.mha'))))
        old,old_members=inference.predict(image,OLD_STAGE/'model','candidate6_full_quality.json',torch.float16,True)
        value,members=inference.predict(image,ROOT/'model','candidate6_full_quality.json',torch.float16,True)
        np.testing.assert_array_equal(members[:5],old_members[:5])
        scale=np.asarray(load(ROOT/'model/candidate6_full_quality.json')['calibration'])
        np.testing.assert_allclose(value,members.mean(0)*scale,rtol=0,atol=1e-12)
        bf16=inference.predict(image,ROOT/'model','candidate6_full_quality.json',torch.bfloat16)
        drift=float(np.max(np.abs(value-bf16)/bf16));assert drift<=.005
        assert value[0]>0 and 0<value[1]<=1 and 0<value[2]<=1
        records.append(dict(index=int(index),fp16=value.tolist(),bf16=bf16.tolist(),relative_precision_drift=drift))
        print(json.dumps(dict(runtime_image=int(index),passed=True)),flush=True)
    inp=out/'input'; idir=inp/'images/corneal-specular-microscopy-image';idir.mkdir(parents=True)
    shutil.copyfile(FRAME/'train_mha'/(str(ids[0])+'.mha'),idir/'case.mha')
    save(inp/'inputs.json',[{'socket':{'slug':'corneal-specular-microscopy-image'}}])
    subprocess.run([sys.executable,str(ROOT/'app/inference.py'),'--input',str(inp),'--output',str(out/'output'),
        '--model',str(ROOT/'model'),'--config','candidate6_full_quality.json','--fp16'],check=True,timeout=300)
    names=('cell-density','coefficient-of-variation','hexagonality')
    assert {p.name for p in (out/'output').iterdir()}=={n+'.json' for n in names}
    values=[load(out/'output'/(n+'.json')) for n in names]
    np.testing.assert_allclose(values,records[0]['fp16'],rtol=1e-5,atol=1e-6)
    save(out/'result.json',dict(passed=True,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),records=records,unchanged_first_five_exact=True,
        official_T4_test=False,actual_archived_docker_image=False,
        scope='CUDA12.6 isolated venv, frozen submission app, H200 numerical/interface check; not efficacy evaluation',
        finished_at=time.strftime('%FT%T%z')))


def ssl_audit():
    manifest=verify();sys.path.insert(0,'experiments/region_search_20260905/source/code')
    from audit_submission_ensembles import metrics
    ys=[];folds=[];bases=[];old5s=[];pred={v:[] for v in RATES};hashes={}
    for fold in FOLDS:
        rec=manifest['folds'][str(fold)];parent=Path(rec['reference_root'])
        assert (ROOT/'ssl_runs'/f'fold{fold}'/'worker_complete.json').is_file()
        with np.load(parent/'preflight/reference.npz') as d:
            mask=(d['folds']==fold)&d['primary'];y=d['target'][mask];old5=d['current5'][mask]
            fold_only=d['folds']==fold;primary=d['primary'][fold_only]
            with np.load(Path(rec['control_root'])/'oof.npz') as c:
                for key in ('ids','indices','target'):np.testing.assert_array_equal(c[key],d[key][fold_only])
                bases.append(c['pred'].astype(float)[primary])
            for variant in RATES:
                p=ROOT/'ssl_runs'/f'fold{fold}'/variant
                r=load(p/'result.json');cfg=load(p/'config.json')
                assert not cfg['smoke'] and cfg['target_policy']=='raw' and cfg['fold']==fold
                assert cfg['encoder_lr']==RATES[variant] and cfg['updates']==rec['updates']
                assert cfg['split_sha256']==rec['split_sha256'] and r['strict_reload_passed']
                assert sha(p/'selected.pt')==r['checkpoint_sha256']
                with np.load(rec['split_path']) as expected,np.load(p/'split_indices.npz') as actual:
                    for key in ('train','validation'):np.testing.assert_array_equal(actual[key],expected[key])
                with np.load(p/'oof.npz') as c:
                    for key in ('ids','indices','target'):np.testing.assert_array_equal(c[key],d[key][fold_only])
                    pred[variant].append(c['pred'].astype(float)[primary])
                for name in ('selected.pt','oof.npz','config.json','result.json'):hashes[str(p/name)]=sha(p/name)
        ys.append(y);old5s.append(old5);folds.extend([fold]*len(y))
    y=np.concatenate(ys);old5=np.concatenate(old5s);f=np.asarray(folds)
    values={'base_raw':np.concatenate(bases),**{v:np.concatenate(x) for v,x in pred.items()}}
    reports={}
    for label,space in [('standalone',values),('fixed_six',{k:(5*old5+v)/6 for k,v in values.items()})]:
        overall={k:float(metrics(v,y).mean()) for k,v in space.items()}
        per_fold={str(fold):{k:float(metrics(v[f==fold],y[f==fold]).mean()) for k,v in space.items()} for fold in FOLDS}
        reports[label]=dict(overall=overall,per_fold=per_fold)
    six=reports['fixed_six'];o=six['overall']
    gate=o['half_lr']<=min(o['standard'],o['base_raw'])-.01 and all(
        p['half_lr']<min(p['standard'],p['base_raw']) for p in six['per_fold'].values())
    save(ROOT/'ssl_audit.json',dict(n=len(y),folds=list(FOLDS),reports=reports,half_lr_research_gate=bool(gate),
        input_sha256=hashes,automatic_promotion=False,
        caveat='Fixed two-rate exploratory comparison on historically reused folds. No calibrated leaderboard prediction.',
        finished_at=time.strftime('%FT%T%z')))


def archive():
    deadline=time.monotonic()+11*3600
    while not (ROOT/'raw_ready_for_archive.json').exists():
        for p in (ROOT/'raw_interrupted.json',ROOT/'raw_complete.json'):
            if p.exists() and (p.name.endswith('interrupted.json') or load(p).get('failed')):
                raise RuntimeError('Raw producer failed; do not archive')
        if time.monotonic()>deadline:raise TimeoutError('No successful raw runtime result')
        time.sleep(30)
    assert load(ROOT/'runtime_check/result.json')['passed']
    stage=load(ROOT/'stage_manifest.json');out=ROOT/'artifacts';out.mkdir(exist_ok=False)
    names=list(stage['files']);assert len(names)==8 and {p.name for p in (ROOT/'model').iterdir()}==set(names)
    for name,row in stage['files'].items():assert sha(ROOT/'model'/name)==row['sha256']
    path=out/'model_full_quality77_raw_20260913.tar.gz';partial=out/(path.name+'.partial')
    with tarfile.open(partial,'w:gz',compresslevel=1,format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(names):tar.add(ROOT/'model'/name,arcname=name,recursive=False)
    with tarfile.open(partial,'r:gz') as tar:
        assert set(tar.getnames())==set(names) and len(tar.getmembers())==8
        for member in tar:
            assert member.isfile() and member.name in stage['files']
            h=hashlib.sha256();stream=tar.extractfile(member)
            for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
            assert h.hexdigest()==stage['files'][member.name]['sha256']
    digest=sha(partial);partial.rename(path)
    (out/'SHA256SUMS.txt').write_text(f'{digest}  {path.name}\n')
    save(out/'model_archive.json',dict(name=path.name,bytes=path.stat().st_size,sha256=digest,
        members=stage['files'],archived_hashes_verified=True,only_changed_model=RAW_NAME,
        original_image_reused=True,calibration_unchanged=True,official_platform_test_pending=True))
    save(ROOT/'package_ready.json',dict(path=str(path),sha256=digest,bytes=path.stat().st_size,
        runtime_passed=True,archive_verified=True,automatic_submission=False,
        finished_at=time.strftime('%FT%T%z')))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','raw-train','ssl-worker','raw-run','ssl-run','stage','runtime','ssl-audit','archive'])
    parser.add_argument('--fold',type=int,choices=FOLDS)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.command=='raw-train':raw_train(args.smoke)
    elif args.command=='ssl-worker':
        assert args.fold is not None;ssl_worker(args.fold,args.smoke)
    elif args.command in ('raw-run','ssl-run'):run_children(args.command.split('-')[0])
    else:globals()[args.command.replace('-','_')]()
