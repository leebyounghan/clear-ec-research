"""Single-model SSL-only ablations; bounded queue, no ensemble or deployment."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np

ROOT=Path('experiments/representation_ssl_20260913')
PARENT=Path('experiments/raw_ssl_next_20260913')
sys.path.insert(0,str(PARENT/'source'))
import next_candidate_20260913 as old
sys.path.insert(0,'experiments/ssl_screen_20260911/source')
from representation_ssl_core import VARIANTS
FOLDS=(2,3)
TASKS=[(fold,v) for fold in FOLDS for v in VARIANTS]


def gate(candidate,reference,perfold):
    return candidate<=reference-.03 and all(p['candidate']<p['reference'] for p in perfold.values())


def verify():
    old.verify();m=old.load(ROOT/'manifest.json')
    for group in ('source_sha256','input_sha256'):
        for p,h in m[group].items(): assert old.sha(p)==h,p
    return m


def prepare():
    assert not (ROOT/'manifest.json').exists()
    previous=old.verify();assert not old.load(PARENT/'ssl_complete.json')['failed']
    audit=old.load(PARENT/'ssl_audit.json');inputs={}
    for p,h in audit['input_sha256'].items():
        if '/standard/' in p and any(f'/fold{f}/' in p for f in FOLDS):
            assert old.sha(p)==h
            if not p.endswith('selected.pt'):inputs[p]=h
    for p in (PARENT/'manifest.json',PARENT/'ssl_audit.json',PARENT/'ssl_complete.json'):
        inputs[str(p)]=old.sha(p)
    # Validate all native pixels against the frozen duplicate audit again.
    with np.load('experiments/ssl_screen_20260911/preflight/pixel_fingerprints.npz') as d: hashes=d['sha256'].copy()
    fm=old.load(old.FRAME/'fullframe_manifest.json')
    images=np.memmap(old.FRAME/'img_full_raw.dat',mode='r',dtype='uint8',shape=tuple(fm['cache_shape']))
    import hashlib
    assert len(images)==len(hashes)==8954
    for image,h in zip(images,hashes):assert hashlib.sha256(image.tobytes()).hexdigest()==h
    folds={str(f):previous['folds'][str(f)] for f in FOLDS}
    for f,r in folds.items():
        with np.load(r['split_path']) as d:
            assert not set(d['train'])&set(d['validation'])
            assert len(d['train'])==r['train_rows'] and r['slide_and_exact_duplicate_overlap']==0
        assert old.sha(r['split_path'])==r['split_sha256']
    sources={str(p):old.sha(p) for p in (ROOT/'source').glob('*.py')}
    m=dict(created_at=time.strftime('%FT%T%z'),source_sha256=sources,input_sha256=inputs,
        folds=folds,variants=VARIANTS,ssl_updates=6000,primary_endpoint='single_model_mean_MAPE',
        downstream='raw labels, full_quality native256, seed77, K2/batch48, encoder1e-5/head1e-3; frozen terminal updates',
        selection='>=0.03 single-model improvement vs existing standard SSL, both folds better; no auto promotion',
        max_seconds=10*3600,no_ensemble=True,no_submission=True,no_all_data_fit=True,
        caveat='Reused development folds; no untouched validation or leaderboard prediction')
    old.save(ROOT/'manifest.json',m)
    print(json.dumps(dict(preflight_passed=True,folds=folds,variants=VARIANTS)),flush=True)


def worker(fold,variant,smoke):
    m=verify();torch=old.setup_torch();shared=old.shared_api();r=m['folds'][str(fold)]
    with np.load('datasets/clear_ec_official/cache_native_meta_512.npz',allow_pickle=True) as d:
        ids,y=d['ids'].astype(str),d['Y'].astype(np.float32)
    with np.load(r['split_path']) as d:tr,va=d['train'].copy(),d['validation'].copy()
    images=np.memmap('datasets/clear_ec_official/img_native_512.dat',mode='r',dtype='uint8',shape=(8954,512,512))
    views=shared.NativeRegionViews(images,ids,'full_quality')
    out=ROOT/('smoke' if smoke else 'runs')/f'{variant}_fold{fold}';out.mkdir(parents=True,exist_ok=False)
    pre=out/'pretrain';pre.mkdir()
    from representation_ssl_core import pretrain
    shared.seed(77);net=shared.build_model('cpsam_models/cpsam_v2')
    ssl=pretrain(net.encoder,views,tr,pre,4 if smoke else 6000,shared,variant)
    del net;gc.collect();torch.cuda.empty_cache()
    net=old.fresh_net(shared,ssl['final_checkpoint'])
    cfg=old.config_for(shared,2 if smoke else r['updates'],smoke,fold=fold,variant=variant,
        encoder_lr=1e-5,head_lr=1e-3,train_rows=len(tr),validation_rows=len(va),
        ssl_checkpoint_sha256=ssl['checkpoint_sha256'],ssl_updates=4 if smoke else 6000,
        split_sha256=r['split_sha256'])
    result=old.fit(shared,net,views,tr,y,out/'downstream',cfg,1e-5,va[:4] if smoke else va,ids)
    if not smoke:
        control=old.load(PARENT/'ssl_runs'/f'fold{fold}'/'standard/result.json')
        assert result['training']['first_three_batches_sha256']==control['training']['first_three_batches_sha256']
    old.save(out/'complete.json',dict(smoke=smoke,finished_at=time.strftime('%FT%T%z'),
        checkpoint_sha256=result['checkpoint_sha256'],strict_reload_passed=True))


def audit():
    m=verify();sys.path.insert(0,'experiments/region_search_20260905/source/code')
    from audit_submission_ensembles import metrics
    targets=[];groups=[];preds={v:[] for v in ('raw','standard_ssl',*VARIANTS)};hashes={}
    for fold in FOLDS:
        r=m['folds'][str(fold)]
        with np.load(Path(r['reference_root'])/'preflight/reference.npz') as ref:
            fmask=ref['folds']==fold;primary=ref['primary'][fmask]
            targets.append(ref['target'][fmask][primary]);groups.extend([fold]*int(primary.sum()))
            paths={'raw':Path(r['control_root']), 'standard_ssl':PARENT/'ssl_runs'/f'fold{fold}'/'standard'}
            paths.update({v:ROOT/'runs'/f'{v}_fold{fold}'/'downstream' for v in VARIANTS})
            for v,p in paths.items():
                if v in VARIANTS:
                    cfg=old.load(p/'config.json');res=old.load(p/'result.json')
                    assert not cfg['smoke'] and cfg['updates']==r['updates'] and cfg['target_policy']=='raw'
                    assert cfg['split_sha256']==r['split_sha256'] and res['strict_reload_passed']
                    assert old.sha(p/'selected.pt')==res['checkpoint_sha256']
                    with np.load(p/'split_indices.npz') as actual,np.load(r['split_path']) as expected:
                        for key in ('train','validation'):np.testing.assert_array_equal(actual[key],expected[key])
                    for name in ('selected.pt','oof.npz','result.json','config.json'):hashes[str(p/name)]=old.sha(p/name)
                with np.load(p/'oof.npz') as d:
                    for k in ('ids','indices','target'):np.testing.assert_array_equal(d[k],ref[k][fmask])
                    preds[v].append(d['pred'].astype(float)[primary])
    y=np.concatenate(targets);g=np.array(groups);values={v:np.concatenate(p) for v,p in preds.items()}
    reports={v:dict(overall=float(metrics(p,y).mean()),per_metric=metrics(p,y).tolist(),
        per_fold={str(f):float(metrics(p[g==f],y[g==f]).mean()) for f in FOLDS}) for v,p in values.items()}
    gates={v:gate(reports[v]['overall'],reports['standard_ssl']['overall'],
        {str(f):dict(candidate=reports[v]['per_fold'][str(f)],reference=reports['standard_ssl']['per_fold'][str(f)]) for f in FOLDS}) for v in VARIANTS}
    old.save(ROOT/'screen_audit.json',dict(n=len(y),folds=list(FOLDS),reports=reports,
        research_gates=gates,input_sha256=hashes,primary_endpoint='single_model',
        automatic_promotion=False,finished_at=time.strftime('%FT%T%z')))


def run():
    m=verify();lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'status.json').exists(),'Never restart partial queue'
    import torch
    assert torch.cuda.device_count()==4
    logs=ROOT/'logs';logs.mkdir(exist_ok=False)
    started=time.monotonic();active={};done=[];failed={}
    def stop(sig=None,frame=None):
        for row in active.values():
            p=row['p']
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        if sig:
            old.save(ROOT/'interrupted.json',dict(signal=sig,at=time.strftime('%FT%T%z')))
            raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(gpu,smoke):
        used=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
        assert used<100,'Assigned GPU unexpectedly in use'
        fold,v=TASKS[gpu];name=f'{v}_fold{fold}'
        cmd=[sys.executable,str(ROOT/'source/run_representation_ssl_20260913.py'),'worker','--fold',str(fold),'--variant',v]
        if smoke:cmd+=['--smoke']
        with (logs/f'{name}_{"smoke" if smoke else "full"}.log').open('x') as log:
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        active[gpu]=dict(p=p,smoke=smoke,name=name,start=time.monotonic())
        print(json.dumps(dict(launched=name,gpu=gpu,smoke=smoke)),flush=True)
    try:
        for gpu in range(4):launch(gpu,True)
        while active:
            for gpu,row in list(active.items()):
                rc=row['p'].poll()
                if rc is None:
                    if time.monotonic()-started>m['max_seconds'] or (row['smoke'] and time.monotonic()-row['start']>1800):raise TimeoutError(row['name'])
                    continue
                del active[gpu]
                if rc:raise RuntimeError(f'{row["name"]}:exit{rc}')
                assert (ROOT/('smoke' if row['smoke'] else 'runs')/row['name']/'complete.json').is_file()
                if row['smoke']:launch(gpu,False)
                else:done.append(row['name'])
            old.save(ROOT/'status.json',dict(at=time.strftime('%FT%T%z'),done=done,failed=failed,
                active={str(g):dict(name=r['name'],smoke=r['smoke'],pid=r['p'].pid) for g,r in active.items()},
                elapsed_seconds=time.monotonic()-started))
            if active:time.sleep(10)
        subprocess.run([sys.executable,str(ROOT/'source/run_representation_ssl_20260913.py'),'audit'],check=True,timeout=900)
    except Exception as e:failed['controller']=repr(e);stop()
    old.save(ROOT/'complete.json',dict(done=done,failed=failed,audit_completed=not failed,
        elapsed_seconds=time.monotonic()-started,finished_at=time.strftime('%FT%T%z')))
    if failed:raise RuntimeError(failed)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','worker','run','audit'))
    p.add_argument('--fold',type=int,choices=FOLDS);p.add_argument('--variant',choices=VARIANTS)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.command=='worker':worker(a.fold,a.variant,a.smoke)
    else:globals()[a.command]()
