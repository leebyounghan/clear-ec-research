"""Finite, predeclared raw-vs-smoothed replication on folds 2/3. No promotion."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(os.environ.get('CLEAR_EC_REPLICATION_ROOT', 'experiments/overnight_target_20260912'))
PARENT = Path('experiments/target_screen_20260912')
PRIOR = Path('experiments/ssl_screen_20260911')
FOLDS = tuple(int(f) for f in os.environ.get('CLEAR_EC_REPLICATION_FOLDS', '2,3').split(','))
assert len(FOLDS)==len(set(FOLDS)) and all(f in (2,3,4) for f in FOLDS)
JOBS = tuple((f,p) for f in FOLDS for p in ('raw','slidesmooth'))
INITIAL = '0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667'
sys.path.insert(0, str(PARENT / 'source'))
from target_screen_core import targets_for_training


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def connected_groups(slides, fingerprints):
    parent = {str(s): str(s) for s in slides}
    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s
    seen = {}
    for slide, digest in zip(slides, fingerprints):
        slide, digest = str(slide), str(digest)
        if digest in seen:
            parent[find(slide)] = find(seen[digest])
        else:
            seen[digest] = slide
    return np.array([find(str(s)) for s in slides])


def clean_split(train, validation, groups):
    train, validation = np.asarray(train), np.asarray(validation)
    assert len(np.unique(train)) == len(train) and len(np.unique(validation)) == len(validation)
    assert not set(train) & set(validation)
    overlap = set(groups[train]) & set(groups[validation])
    clean = train[~np.isin(groups[train], list(overlap))]
    primary = validation[~np.isin(groups[validation], list(overlap))]
    assert not set(groups[clean]) & set(groups[validation])
    assert not set(groups[train]) & set(groups[primary])
    return clean, primary


def metadata():
    data = Path('datasets/clear_ec_official')
    with np.load(data/'cache_native_meta_512.npz', allow_pickle=True) as d:
        ids, y = d['ids'].astype(str), d['Y'].astype(np.float32)
    with (data/'final_train_ids.csv').open() as f:
        mapping = {Path(r['ID']).stem: str(r['slide_id']) for r in csv.DictReader(f)}
    slides = np.array([mapping[Path(i).stem] for i in ids])
    assert len(ids) == len(set(ids)) == 8954
    return ids, y, slides


def prepare():
    pre = ROOT/'preflight'
    pre.mkdir(exist_ok=False)
    parent = json.loads((PARENT/'manifest.json').read_text())
    assert sha('cpsam_models/cpsam_v2') == INITIAL
    sources = dict(parent['source_sha256'])
    for path, digest in sources.items():
        assert sha(path) == digest, path
    sources.update({str(p): sha(p) for p in (ROOT/'source').glob('*.py') if not p.name.startswith('._')})
    ids, y, slides = metadata()
    fp_path = PRIOR/'preflight/pixel_fingerprints.npz'
    assert sha(fp_path) == parent['input_sha256'][str(fp_path)]
    with np.load(fp_path) as d:
        fingerprints = d['sha256'].copy()
    frame = Path('datasets/clear_ec_fullframe_20260904')
    frame_manifest = json.loads((frame/'fullframe_manifest.json').read_text())
    with np.load(frame/'fullframe_meta.npz', allow_pickle=True) as d:
        np.testing.assert_array_equal(ids, d['ids'].astype(str))
        np.testing.assert_array_equal(y, d['Y'])
    images = np.memmap(frame/'img_full_raw.dat', mode='r', dtype=np.uint8,
                       shape=tuple(frame_manifest['cache_shape']))
    for i in range(len(ids)):
        assert hashlib.sha256(images[i].tobytes()).hexdigest() == fingerprints[i], 'Image cache changed'
    groups = connected_groups(slides, fingerprints)
    # Five historical central members have aligned splits on these folds. The
    # historical sixth is seed42, NOT a nonexistent submitted-seed77 fold2/3 fit.
    sys.path.insert(0, 'experiments/region_search_20260905/source/code')
    import audit_region_search as historical
    _, reference = historical.compare('full_quality', FOLDS)
    records, primary = {}, []
    inputs = {str(PARENT/'manifest.json'): sha(PARENT/'manifest.json'), str(fp_path): sha(fp_path)}
    for fold in FOLDS:
        old = Path('experiments/region_search_20260905/runs')/f'full_quality_f{fold}_seed42_c256k2_mean_multitask_slidesmooth'
        split_path = old/'split_indices.npz'
        with np.load(split_path) as d:
            tr, va = d['train'].copy(), d['validation'].copy()
        assert (y[tr] > 0).all() and not set(slides[tr]) & set(slides[va])
        clean, primary_va = clean_split(tr, va, groups)
        expected = reference['folds'] == fold
        np.testing.assert_array_equal(ids[va], reference['ids'][expected])
        np.testing.assert_array_equal(y[va], reference['target'][expected])
        assert len(clean) > 7000 and len(primary_va) > 1700
        destination = pre/f'fold{fold}.npz'
        np.savez_compressed(destination, train=clean, validation=va,
                            primary_validation=primary_va, historical_train=tr)
        for p in (split_path, destination):
            inputs[str(p)] = sha(p)
        primary.extend(primary_va.tolist())
        records[str(fold)] = dict(train_rows=len(clean), validation_rows=len(va),
            primary_rows=len(primary_va), removed_train_rows=len(tr)-len(clean),
            split_path=str(destination), split_sha256=sha(destination),
            updates=35*(len(clean)//48), slide_and_exact_duplicate_overlap=0)
    for p in (Path('datasets/clear_ec_official/cache_native_meta_512.npz'),
              Path('datasets/clear_ec_official/final_train_ids.csv'),
              frame/'fullframe_manifest.json', frame/'fullframe_meta.npz',
              frame/'context_quality.npz', frame/'context_views_manifest.json'):
        inputs[str(p)] = sha(p)
    primary_mask = np.isin(reference['indices'], primary)
    np.savez_compressed(pre/'reference.npz', **reference, primary=primary_mask)
    inputs[str(pre/'reference.npz')] = sha(pre/'reference.npz')
    manifest = dict(created_at=time.strftime('%FT%T%z'), initial_sha256=INITIAL,
        source_sha256=sources, input_sha256=inputs, folds=records,
        jobs=[dict(gpu=g, fold=f, policy=p) for g, (f,p) in enumerate(JOBS)],
        gpu_count=len(JOBS), max_worker_seconds=5.5*3600, max_queue_seconds=6*3600,
        smoke_timeout_seconds=900, epochs=35, seed=77,
        base_only=True, new_ssl_pretraining=False, external_data=False,
        selection=f'Raw-versus-smoothed on folds {FOLDS}; launched after parent results, not prospectively registered',
        inference='full_quality native256 K2, fixed40view BF16, equal replacement weight1/6, no calibration fit',
        prerequisite='Parent queue_finished.json with successful audit and no failed workers',
        gate='Raw improves matched smoothed on BOTH folds and aggregate >=0.02 points; standalone and fixed-six separately',
        reference_caveat='Historical six uses regional seed42, not the submitted seed77 sixth. Matched NEW smoothed seed77 is primary comparator.',
        limitations=['Historically reused slide-group folds, not untouched test data',
                     'No patient identity established; near duplicates not exhaustively audited',
                     f'Does not confirm SSL-specific interactions; evaluated folds {FOLDS} only'],
        deployment_approved=False, automatic_submission=False)
    save(ROOT/'manifest.json', manifest)
    print(json.dumps(dict(preflight_passed=True, folds=records, source_files=len(sources))), flush=True)


def verify(manifest):
    for records in ('source_sha256', 'input_sha256'):
        for path, digest in manifest[records].items():
            assert sha(path) == digest, path
    assert sha('cpsam_models/cpsam_v2') == manifest['initial_sha256']


def train(args):
    import torch
    import torch.utils.checkpoint as checkpoint
    import cv2
    sys.path.insert(0, 'experiments/crop_dino_screen_20260910/source/frozen')
    import train_specialize_screen as shared
    manifest = json.loads((ROOT/'manifest.json').read_text())
    verify(manifest)
    out = ROOT/('smoke' if args.smoke else 'runs')/f'{args.policy}_fold{args.fold}'
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    ids, y, slides = metadata()
    record = manifest['folds'][str(args.fold)]
    with np.load(record['split_path']) as d:
        tr, va = d['train'].copy(), d['validation'].copy()
    assert len(tr) == record['train_rows'] and not set(slides[tr]) & set(slides[va])
    images = np.memmap('datasets/clear_ec_official/img_native_512.dat', mode='r',
                       dtype=np.uint8, shape=(len(ids),512,512))
    views = shared.NativeRegionViews(images, ids, 'full_quality')
    targets = targets_for_training(y, slides, tr, args.policy)
    np.testing.assert_array_equal(targets[va], y[va])
    z = torch.tensor(((np.log(targets.clip(1e-6))-shared.LM)/shared.LS).astype(np.float32), device='cuda')
    shared.seed(77)
    net = shared.build_model('cpsam_models/cpsam_v2')
    for block in net.encoder.blocks:
        original = block.forward
        block.forward = lambda x, fn=original: checkpoint.checkpoint(fn,x,use_reentrant=False)
    updates = 2 if args.smoke else record['updates']
    config = dict(fold=args.fold, target_policy=args.policy, seed=77, smoke=args.smoke,
        updates=updates, train_rows=len(tr), validation_rows=len(va), batch_size=48,
        k_crops=2, region_policy='full_quality', objective='mixed_mape',
        encoder_lr=1e-5, head_lr=1e-3, select_mode='fixed_terminal_no_validation_selection',
        initial_sha256=INITIAL, split_sha256=record['split_sha256'],
        lm=shared.LM.tolist(), ls=shared.LS.tolist(), initialization='fresh_original_cpsam')
    save(out/'config.json', config)
    np.savez_compressed(out/'split_indices.npz', train=tr, validation=va)
    training = shared.train(net,views,tr,z,None,updates,out,1e-5,77)
    torch.save(dict(model=net.state_dict(),config=config),out/'selected.pt')
    ck = torch.load(out/'selected.pt', map_location='cpu', weights_only=True)
    net.load_state_dict(ck['model'],strict=True); del ck
    evaluation = shared.evaluate(net,views,va[:4] if args.smoke else va,y,ids,out)
    result = dict(completed_at=time.strftime('%FT%T%z'), training=training,
        evaluation=evaluation, checkpoint_sha256=sha(out/'selected.pt'),
        strict_reload_passed=True, smoke=args.smoke, deployment_approved=False)
    save(out/'result.json',result)
    print(json.dumps(result),flush=True)


def gate(deltas):
    return bool(len(deltas['per_fold'])>=2 and deltas['aggregate'] <= -.02 and all(v < 0 for v in deltas['per_fold']))


def audit():
    manifest=json.loads((ROOT/'manifest.json').read_text()); verify(manifest)
    completed=json.loads((ROOT/'complete.json').read_text())
    assert not completed['failed'] and len(completed['done']) == len(JOBS)
    sys.path.insert(0,'experiments/region_search_20260905/source/code')
    from audit_submission_ensembles import metrics
    with np.load(ROOT/'preflight/reference.npz') as d:
        ref={k:d[k].copy() for k in d.files}
    predictions={}; timings={}; hashes={}; samples={}
    for policy in ('raw','slidesmooth'):
        chunks=[]
        for fold in FOLDS:
            path=ROOT/'runs'/f'{policy}_fold{fold}'
            result=json.loads((path/'result.json').read_text())
            config=json.loads((path/'config.json').read_text())
            assert result['strict_reload_passed'] and not config['smoke']
            assert config['updates']==manifest['folds'][str(fold)]['updates']
            assert config['initial_sha256']==INITIAL and config['seed']==77
            assert config['target_policy']==policy and config['fold']==fold
            record=manifest['folds'][str(fold)]
            assert config['split_sha256']==record['split_sha256']
            with np.load(path/'split_indices.npz') as actual, np.load(record['split_path']) as expected:
                for key in ('train','validation'):
                    np.testing.assert_array_equal(actual[key],expected[key])
            assert sha(path/'selected.pt')==result['checkpoint_sha256']
            with np.load(path/'oof.npz') as d:
                mask=ref['folds']==fold
                for key in ('indices','ids','target'):
                    np.testing.assert_array_equal(d[key],ref[key][mask])
                prediction=d['pred'].astype(float)
                assert np.isfinite(prediction).all() and (prediction>0).all()
                chunks.append(prediction)
            samples[(fold,policy)]=result['training']['first_three_batches_sha256']
            timings[f'{policy}_fold{fold}']=dict(train_seconds=result['training']['elapsed_seconds'],
                                               eval_seconds=result['evaluation']['elapsed_seconds'])
            for name in ('config.json','result.json','oof.npz','selected.pt','split_indices.npz'):
                hashes[str(path/name)]=sha(path/name)
        predictions[policy]=np.concatenate(chunks)[ref['primary']]
    for fold in FOLDS:
        assert samples[(fold,'raw')]==samples[(fold,'slidesmooth')], 'Sampling mismatch'
    y=ref['target'][ref['primary']]; f=ref['folds'][ref['primary']]
    base5=ref['current5'][ref['primary']]
    replace={p:(5*base5+v)/6 for p,v in predictions.items()}
    reports={}
    for name,space in (('standalone',predictions),('fixed_six',replace)):
        deltas=dict(aggregate=float((metrics(space['raw'],y)-metrics(space['slidesmooth'],y)).mean()),
            per_fold=[float((metrics(space['raw'][f==fold],y[f==fold])-metrics(space['slidesmooth'][f==fold],y[f==fold])).mean()) for fold in FOLDS])
        reports[name]=dict(metrics={p:metrics(v,y).tolist() for p,v in space.items()},
                           raw_minus_smoothed=deltas,passes_replication_gate=gate(deltas))
    report=dict(completed_at=time.strftime('%FT%T%z'),folds=list(FOLDS),n=len(y),
        comparisons=reports,compute=timings,input_sha256=hashes,
        parent_screen=json.loads((PARENT/'screen_audit.json').read_text())['comparisons'],
        current5_metrics=metrics(base5,y).tolist(), reference_caveat=manifest['reference_caveat'],
        limitations=manifest['limitations'],deployment_approved=False,automatic_submission=False)
    save(ROOT/'screen_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'}),flush=True)


def run():
    import fcntl
    started=time.monotonic()
    lock=(ROOT/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'status.json').exists(), 'Never restart a partial queue'
    prerequisite=json.loads((PARENT/'queue_finished.json').read_text())
    assert prerequisite['audit_completed']
    assert not json.loads((PARENT/'complete.json').read_text())['failed']
    assert (PARENT/'screen_audit.json').is_file()
    manifest=json.loads((ROOT/'manifest.json').read_text()); verify(manifest)
    import torch
    assert torch.cuda.device_count()==manifest['gpu_count']==len(JOBS)
    subprocess.run([sys.executable,str(ROOT/'source/test_overnight_target.py')],check=True,timeout=120)
    deadline=started+manifest['max_queue_seconds']
    active={}; done=[]; smokes=[]; failed={}; full_started=False
    (ROOT/'logs').mkdir(exist_ok=False)
    def stop(signum=None,frame=None):
        for item in active.values():
            p=item['process']
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        if signum:
            save(ROOT/'interrupted.json',dict(signal=signum,at=time.strftime('%FT%T%z')))
            raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    def launch(gpu,smoke):
        fold,policy=JOBS[gpu]; name=f'{policy}_fold{fold}'
        used=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        assert used<100, 'Assigned GPU unexpectedly occupied'
        cmd=[sys.executable,str(ROOT/'source/overnight_target.py'),'train','--fold',str(fold),'--policy',policy]
        if smoke:cmd.append('--smoke')
        with (ROOT/'logs'/f'{name}_{"smoke" if smoke else "full"}.log').open('x') as log:
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
        active[gpu]=dict(process=p,name=name,smoke=smoke,start=time.monotonic())
        print(json.dumps(dict(launched=name,gpu=gpu,smoke=smoke)),flush=True)
    try:
        for gpu in range(len(JOBS)):launch(gpu,True)
        while active:
            for gpu,item in list(active.items()):
                rc=item['process'].poll()
                limit=manifest['smoke_timeout_seconds'] if item['smoke'] else manifest['max_worker_seconds']
                if rc is None:
                    if time.monotonic()>deadline or time.monotonic()-item['start']>limit:
                        failed[item['name']]='timeout'; stop(); break
                    continue
                del active[gpu]
                path=ROOT/('smoke' if item['smoke'] else 'runs')/item['name']/'result.json'
                if rc or not path.exists():
                    failed[item['name']]=f'exit={rc}; result_exists={path.exists()}';stop();break
                assert json.loads(path.read_text())['strict_reload_passed']
                (smokes if item['smoke'] else done).append(item['name'])
            if failed:break
            if not active and len(smokes)==len(JOBS) and not full_started:
                for fold in FOLDS:
                    pair=[json.loads((ROOT/'smoke'/f'{p}_fold{fold}'/'result.json').read_text()) for p in ('raw','slidesmooth')]
                    assert pair[0]['training']['first_three_batches_sha256']==pair[1]['training']['first_three_batches_sha256']
                full_started=True
                for gpu in range(len(JOBS)):launch(gpu,False)
            save(ROOT/'status.json',dict(updated_at=time.strftime('%FT%T%z'),done=done,smoke_done=smokes,failed=failed,
                active={str(g):dict(name=v['name'],pid=v['process'].pid,smoke=v['smoke']) for g,v in active.items()},
                remaining_queue_seconds=max(0,deadline-time.monotonic())))
            if active:time.sleep(10)
    except Exception as e:
        failed['controller']=repr(e);stop()
    save(ROOT/'complete.json',dict(completed_at=time.strftime('%FT%T%z'),done=done,failed=failed,smoke_done=smokes))
    if failed:raise RuntimeError(failed)
    try:
        remaining=int(deadline-time.monotonic())
        assert remaining>0
        subprocess.run([sys.executable,str(ROOT/'source/overnight_target.py'),'audit'],check=True,timeout=min(1200,remaining))
    except Exception as e:
        save(ROOT/'audit_failed.json',dict(error=repr(e)));raise
    save(ROOT/'queue_finished.json',dict(finished_at=time.strftime('%FT%T%z'),audit_completed=True,
        gpu_container_will_exit=True,original_pvc_retained=True,automatic_submission=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','train','audit','run'))
    parser.add_argument('--fold',type=int,choices=FOLDS)
    parser.add_argument('--policy',choices=('raw','slidesmooth'))
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.command=='train':
        assert args.fold is not None and args.policy is not None
        train(args)
    else:globals()[args.command]()
