"""Fold-clean SSL checkpoint-duration ablation; research only, no release writes."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np

ROOT = Path('experiments/ssl_duration_20260914')
PARENT = Path('experiments/raw_ssl_next_20260913')
sys.path.insert(0, str(PARENT/'source'))
import next_candidate_20260913 as old
sys.path.insert(0, 'experiments/region_search_20260905/source/code')
from audit_submission_ensembles import metrics
FOLDS = (2, 3)
STEPS = (2000, 4000)


def score(p, y):
    m = metrics(p, y)
    return dict(mean=float(m.mean()), CD_CV_HEX=m.tolist(), n=len(y),
                valid_counts=((y>0)&np.isfinite(y)).sum(0).tolist())


def tests():
    p=np.array([[2.,9.,4.],[2.,1.,4.]])
    y=np.array([[1.,0.,4.],[2.,2.,2.]])
    np.testing.assert_allclose(score(p,y)['CD_CV_HEX'], [50,50,50])
    assert score(p,y)['valid_counts']==[2,1,2]
    y[0,1]=np.nan
    np.testing.assert_allclose(score(p,y)['CD_CV_HEX'], [50,50,50])
    y[:,1]=0
    try: score(p,y)
    except ValueError: pass
    else: raise AssertionError('Empty metric accepted')
    y=np.ones((2,3))
    np.testing.assert_allclose(metrics(p,y), (abs(p-y)/y).mean(0)*100)


def verify():
    m=old.load(ROOT/'manifest.json')
    for group in ('sources','inputs'):
        for path,h in m[group].items(): assert old.sha(path)==h,path
    return m


def prepare():
    import torch
    assert not (ROOT/'manifest.json').exists()
    tests(); previous=old.verify()
    assert not old.load(PARENT/'ssl_complete.json')['failed']
    sources=dict(previous['source_sha256']); inputs=dict(previous['input_sha256'])
    for directory in (ROOT/'source', Path('experiments/crop_dino_screen_20260910/source/frozen'),
                      Path('experiments/specialize_screen_20260908/source/frozen'),
                      Path('experiments/region_search_20260905/source/code')):
        sources.update({str(p):old.sha(p) for p in directory.glob('*.py') if not p.name.startswith('._')})
    folds={str(f):previous['folds'][str(f)] for f in FOLDS}; checkpoints={}
    for f,r in folds.items():
        with np.load(r['split_path']) as d:
            assert not set(d['train'])&set(d['validation'])
            assert len(d['train'])==r['train_rows'] and r['slide_and_exact_duplicate_overlap']==0
        pre=PARENT/'ssl_runs'/f'fold{f}'/'pretrain'
        for s in (*STEPS,6000):
            p=pre/f'encoder_step{s:04d}.pt'; ck=torch.load(p,map_location='cpu',weights_only=True)
            assert ck['step']==s and ck['config']['updates']==6000
            assert ck['config']['validation_images_used']==0
            assert ck['config']['clear_ec_train_rows']==r['train_rows']
            del ck
            checkpoints[f'{f}_{s}']=str(p);inputs[str(p)]=old.sha(p)
        for base in (Path(r['control_root']),PARENT/'ssl_runs'/f'fold{f}'/'standard'):
            result=old.load(base/'result.json');assert result['strict_reload_passed']
            assert old.sha(base/'selected.pt')==result['checkpoint_sha256']
            for name in ('selected.pt','config.json','result.json','oof.npz','split_indices.npz'):
                inputs[str(base/name)]=old.sha(base/name)
        assert inputs[checkpoints[f'{f}_6000']]==old.load(PARENT/'ssl_runs'/f'fold{f}'/'standard/config.json')['ssl_checkpoint_sha256']
    m=dict(created_at=time.strftime('%FT%T%z'),sources=sources,inputs=inputs,folds=folds,
        checkpoints=checkpoints,ssl_checkpoints=list(STEPS),baseline_ssl_checkpoint=6000,
        primary_endpoint='single-model mean of per-metric MAPE',
        downstream='raw targets, seed77, native full_quality256, K2/batch48, lr1e-5/head1e-3, 5215 updates,40view TTA',
        selection='Choose checkpoint using fold2 only; require pooled improvement >=0.03 vs6000 and both folds better vs6000 and raw.',
        caveat='Early checkpoints of the SAME6000-update cosine trajectory, not independently shortened cosine runs. Reused development folds, no untouched validation.',
        ssl_validation_images_used=0,all9000_ssl=False,no_new_pretraining=True,
        no_automatic_promotion=True,no_submission=True,no_all_data_fit=True,
        concurrent_research_gpus=2,max_seconds_per_fold=39600,maximum_research_runtime_gpu_hours=22,
        cpu_metric_tests_passed=True)
    old.save(ROOT/'manifest.json',m)
    print(json.dumps(dict(preflight_passed=True,folds=list(folds),checkpoints=list(STEPS))),flush=True)


def worker(fold,step,smoke):
    m=verify();torch=old.setup_torch();shared=old.shared_api();r=m['folds'][str(fold)]
    with np.load('datasets/clear_ec_official/cache_native_meta_512.npz',allow_pickle=True) as d:
        ids,y=d['ids'].astype(str),d['Y'].astype(np.float32)
    with np.load(r['split_path']) as d:tr,va=d['train'].copy(),d['validation'].copy()
    assert (y[tr]>0).all() and np.isfinite(y[tr]).all()
    images=np.memmap('datasets/clear_ec_official/img_native_512.dat',mode='r',dtype='uint8',shape=(8954,512,512))
    views=shared.NativeRegionViews(images,ids,'full_quality')
    checkpoint=m['checkpoints'][f'{fold}_{step}'];net=old.fresh_net(shared,checkpoint)
    cfg=old.config_for(shared,2 if smoke else r['updates'],smoke,fold=fold,
        encoder_lr=1e-5,head_lr=1e-3,train_rows=len(tr),validation_rows=len(va),
        split_sha256=r['split_sha256'],ssl_checkpoint_sha256=m['inputs'][checkpoint],
        ssl_updates=step,ssl_schedule_total=6000,ssl_validation_images_used=0)
    out=ROOT/('smoke' if smoke else 'runs')/f'step{step}_fold{fold}'
    result=old.fit(shared,net,views,tr,y,out,cfg,1e-5,va[:4] if smoke else va,ids)
    if not smoke:
        control=old.load(PARENT/'ssl_runs'/f'fold{fold}'/'standard/result.json')
        assert result['training']['first_three_batches_sha256']==control['training']['first_three_batches_sha256']
    old.save(out/'complete.json',dict(smoke=smoke,strict_reload_passed=True,
        checkpoint_sha256=result['checkpoint_sha256'],at=time.strftime('%FT%T%z')))


def run(fold):
    verify();out=ROOT/f'queue_fold{fold}';out.mkdir(exist_ok=False)
    started=time.monotonic();completed=[]
    # Child timeout plus process-group termination keeps the per-fold queue bounded.
    child=None
    def stop(sig=None,frame=None):
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if sig: raise TimeoutError(f'Interrupted by {sig}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for step in STEPS:
            for smoke in (True,False):
                old.save(out/'status.json',dict(step=step,smoke=smoke,completed=completed,at=time.strftime('%FT%T%z')))
                args=[sys.executable,str(ROOT/'source/ssl_duration_20260914.py'),'worker','--fold',str(fold),'--step',str(step)]
                if smoke:args+=['--smoke']
                with (out/f'{step}_{"smoke" if smoke else "full"}.log').open('x') as log:
                    child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    remaining=39600-(time.monotonic()-started)
                    assert remaining>0
                    rc=child.wait(timeout=min(1800 if smoke else 19800,remaining))
                assert rc==0,f'checkpoint{step},smoke{smoke},exit{rc}'
                assert (ROOT/('smoke' if smoke else 'runs')/f'step{step}_fold{fold}'/'complete.json').exists()
            completed.append(step)
        old.save(out/'complete.json',dict(completed=completed,elapsed_seconds=time.monotonic()-started,at=time.strftime('%FT%T%z')))
    except BaseException as error:
        stop();old.save(out/'failure.json',dict(error=repr(error),completed=completed,at=time.strftime('%FT%T%z')))
        raise


def audit():
    m=verify();assert not (ROOT/'comparison.json').exists()
    labels=('raw','ssl6000','ssl2000','ssl4000');parts={k:[] for k in labels};targets=[];groups=[];hashes={}
    for fold in FOLDS:
        assert old.load(ROOT/f'queue_fold{fold}'/'complete.json')['completed']==list(STEPS)
        r=m['folds'][str(fold)]
        with np.load(Path(r['reference_root'])/'preflight/reference.npz') as ref:
            fm=ref['folds']==fold;keep=ref['primary'][fm]
            targets.append(ref['target'][fm][keep]);groups.extend([fold]*int(keep.sum()))
            paths={'raw':Path(r['control_root']),'ssl6000':PARENT/'ssl_runs'/f'fold{fold}'/'standard',
                   **{f'ssl{s}':ROOT/'runs'/f'step{s}_fold{fold}' for s in STEPS}}
            for label,path in paths.items():
                result=old.load(path/'result.json');cfg=old.load(path/'config.json')
                assert result['strict_reload_passed'] and not cfg['smoke'] and cfg['updates']==r['updates']
                assert cfg['target_policy']=='raw' and cfg['split_sha256']==r['split_sha256']
                assert old.sha(path/'selected.pt')==result['checkpoint_sha256']
                with np.load(path/'split_indices.npz') as a,np.load(r['split_path']) as b:
                    for k in ('train','validation'):np.testing.assert_array_equal(a[k],b[k])
                with np.load(path/'oof.npz') as d:
                    for k in ('ids','indices','target'):np.testing.assert_array_equal(d[k],ref[k][fm])
                    parts[label].append(d['pred'][keep].astype(float))
                for name in ('selected.pt','result.json','config.json','oof.npz'):hashes[str(path/name)]=old.sha(path/name)
    y=np.concatenate(targets);g=np.array(groups);reports={}
    for label in labels:
        p=np.concatenate(parts[label]);reports[label]=dict(overall=score(p,y),per_fold={str(f):score(p[g==f],y[g==f]) for f in FOLDS})
    chosen=min(('ssl2000','ssl4000'),key=lambda k:(reports[k]['per_fold']['2']['mean'],k))
    passed=reports[chosen]['overall']['mean']<=reports['ssl6000']['overall']['mean']-.03 and all(
        reports[chosen]['per_fold'][str(f)]['mean']<reports[ref]['per_fold'][str(f)]['mean'] for f in FOLDS for ref in ('raw','ssl6000'))
    result=dict(reports=reports,chosen_by_fold2=chosen,research_gate_passed=bool(passed),
        n=len(y),artifact_sha256=hashes,automatic_promotion=False,ssl_validation_images_used=0,
        independent_validation=False,caveat=m['caveat'],finished_at=time.strftime('%FT%T%z'))
    old.save(ROOT/'comparison.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','tests','run','worker','audit'))
    parser.add_argument('--fold',type=int,choices=FOLDS);parser.add_argument('--step',type=int,choices=STEPS)
    parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    if a.command=='worker':worker(a.fold,a.step,a.smoke)
    elif a.command=='run':run(a.fold)
    else:globals()[a.command]()
