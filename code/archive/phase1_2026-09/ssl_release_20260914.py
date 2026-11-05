"""All-data standard masked EMA SSL + raw sixth-member release, no submission.

Reuses immutable standard SSL/training/application code. Original artifacts stay
unchanged. CPU archive is separate so compression does not hold a GPU.
"""
import argparse
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

ROOT=Path('experiments/ssl_release_20260914')
PARENT=Path('experiments/raw_ssl_next_20260913')
sys.path.insert(0,str(PARENT/'source'))
import next_candidate_20260913 as old
BASE_VERIFY=old.verify
RUNTIME=PARENT/'runtime_setup/.venv/bin/python'
NAME='model_full_quality77_ssl_raw_20260914.tar.gz'


def verify():
    previous=old.ROOT;old.ROOT=PARENT
    try:BASE_VERIFY()
    finally:old.ROOT=previous
    m=old.load(ROOT/'manifest.json')
    for group in ('source_sha256','input_sha256'):
        for p,h in m[group].items():assert old.sha(p)==h,p
    return m


def bind():
    old.ROOT=ROOT;old.verify=verify


def prepare():
    assert not (ROOT/'manifest.json').exists()
    m=BASE_VERIFY()
    assert not old.load(PARENT/'ssl_complete.json')['failed']
    assert old.load(PARENT/'runtime_check/result.json')['passed']
    # Preserve submitted application source independently in this release folder.
    shutil.copytree(PARENT/'app',ROOT/'app')
    sys.path.insert(0,'experiments/deploy_region_20260907/source')
    from train_deploy_region import build_regions
    views=build_regions('full_quality');assert views.images.shape[0]==9000
    with np.load(old.ALLDATA/'meta.npz') as d:
        assert len(d['ids'])==9000 and len(d['train_indices'])==8998
        assert (d['Y'][d['train_indices']]>0).all()
    assert RUNTIME.is_file()
    sources={str(p):old.sha(p) for p in (ROOT/'source').glob('*.py')}
    sources.update({str(p):old.sha(p) for p in (ROOT/'app').glob('*.py')})
    sources.update({str(p):old.sha(p) for p in Path('experiments/ssl_screen_20260911/source').glob('*.py')})
    sources.update({str(p):old.sha(p) for p in Path('experiments/deploy_region_20260907/source').glob('*.py')})
    inputs={str(PARENT/n):old.sha(PARENT/n) for n in ('manifest.json','ssl_audit.json','ssl_complete.json','runtime_check/result.json')}
    out=dict(created_at=time.strftime('%FT%T%z'),source_sha256=sources,input_sha256=inputs,
        old_members=m['old_members'],runtime_python=str(RUNTIME),
        ssl_method='unchanged standard custom masked EMA online variance .1 covariance .01',
        ssl_images=9000,ssl_updates=6000,ssl_batch=32,ssl_lr=1e-5,
        downstream_rows=8998,downstream_updates=5215,encoder_lr=1e-5,head_lr=1e-3,
        seed=77,raw_labels=True,full_quality_native256=True,ssl_no_resize=True,
        independent_validation=False,full_data_final_fit=True,no_automatic_submission=True,
        preserved_first_five_and_calibration=True,archive_name=NAME,
        no_dino=True,no_new_hyperparameter_search=True,max_gpu_seconds=9*3600)
    old.save(ROOT/'manifest.json',out);print(json.dumps(out),flush=True)


def worker(smoke):
    verify();torch=old.setup_torch();shared=old.shared_api();shared.seed(77)
    sys.path.insert(0,'experiments/deploy_region_20260907/source')
    from train_deploy_region import build_regions
    sys.path.insert(0,'experiments/ssl_screen_20260911/source')
    from ssl_native_core import pretrain
    views=build_regions('full_quality')
    with np.load(old.ALLDATA/'meta.npz') as d:ids,y,tr=d['ids'].copy(),d['Y'].copy(),d['train_indices'].copy()
    ssl_indices=np.arange(9000,dtype=np.int64)
    pre=ROOT/('ssl_smoke' if smoke else 'pretrain');pre.mkdir(exist_ok=False)
    net=shared.build_model('cpsam_models/cpsam_v2')
    seen=set();original_train=views.train
    def recording_train(index,k=2):
        seen.add(int(index));return original_train(index,k)
    views.train=recording_train
    result=pretrain(net.encoder,views,ssl_indices,pre,4 if smoke else 6000,shared)
    views.train=original_train
    if not smoke:assert len(seen)==9000,'Not all SSL images sampled'
    old.save(pre/'coverage.json',dict(eligible_images=9000,unique_images_seen=len(seen),labels_used=False))
    del net;gc.collect();torch.cuda.empty_cache()
    net=old.fresh_net(shared,result['final_checkpoint'])
    cfg=old.config_for(shared,2 if smoke else 5215,smoke,train_rows=8998,validation_rows=0,
        encoder_lr=1e-5,head_lr=1e-3,initialization='standard_SSL_all9000_terminal_EMA',
        ssl_images=9000,ssl_updates=4 if smoke else 6000,ssl_checkpoint_sha256=result['checkpoint_sha256'],
        full_data_final_fit=True,independent_validation=False)
    out=ROOT/('raw_smoke' if smoke else 'raw_deploy')
    old.fit(shared,net,views,tr,y,out,cfg,1e-5,tr[:4] if smoke else None,ids)


def run():
    m=verify();bind()
    assert not (ROOT/'gpu_complete.json').exists() and not (ROOT/'gpu_status.json').exists()
    logs=ROOT/'logs';logs.mkdir(exist_ok=False);started=time.monotonic();active=None;failed={}
    def stop(sig=None,frame=None):
        if active is not None and active.poll() is None:
            os.killpg(active.pid,signal.SIGTERM)
            try:active.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
        if sig:
            old.save(ROOT/'interrupted.json',dict(signal=sig,at=time.strftime('%FT%T%z')))
            raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    stages=[('smoke',sys.executable,['worker','--smoke'],1800),
            ('full_train',sys.executable,['worker'],8*3600),
            ('stage',sys.executable,['stage'],900),
            ('runtime',str(RUNTIME.absolute()),['runtime'],1800)]
    try:
        for name,python,args,limit in stages:
            old.save(ROOT/'gpu_status.json',dict(stage=name,at=time.strftime('%FT%T%z'),elapsed_seconds=time.monotonic()-started))
            with (logs/(name+'.log')).open('x') as log:
                active=subprocess.Popen([python,str(ROOT/'source/ssl_release_20260914.py'),*args],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active.wait(timeout=min(limit,max(1,m['max_gpu_seconds']-(time.monotonic()-started))))
            assert active.returncode==0,f'{name} exit {active.returncode}'
        old.save(ROOT/'ready_for_archive.json',dict(runtime_passed=True,at=time.strftime('%FT%T%z')))
    except Exception as e:failed['controller']=repr(e);stop()
    old.save(ROOT/'gpu_complete.json',dict(failed=failed,elapsed_seconds=time.monotonic()-started,at=time.strftime('%FT%T%z')))
    if failed:raise RuntimeError(failed)


def stage():
    m=verify();cfg=old.load(ROOT/'raw_deploy/config.json');ssl=old.load(ROOT/'pretrain/result.json')
    assert cfg['ssl_images']==9000 and cfg['ssl_updates']==6000
    assert old.sha(ROOT/'pretrain/encoder_step6000.pt')==cfg['ssl_checkpoint_sha256']==ssl['checkpoint_sha256']
    bind();old.stage()


def runtime():
    bind();old.runtime()


def archive():
    started=time.monotonic()
    while not (ROOT/'ready_for_archive.json').exists():
        assert not (ROOT/'interrupted.json').exists(),'GPU interrupted'
        if (ROOT/'gpu_complete.json').exists():assert not old.load(ROOT/'gpu_complete.json')['failed']
        assert time.monotonic()-started<10*3600,'Archive wait timeout'
        time.sleep(30)
    verify();assert old.load(ROOT/'runtime_check/result.json')['passed']
    inventory=old.load(ROOT/'stage_manifest.json')['files'];assert len(inventory)==8
    out=ROOT/'artifacts';out.mkdir(exist_ok=False);partial=out/(NAME+'.partial')
    for n,r in inventory.items():assert old.sha(ROOT/'model'/n)==r['sha256']
    with tarfile.open(partial,'w:gz',compresslevel=1,format=tarfile.PAX_FORMAT) as tar:
        for n in sorted(inventory):tar.add(ROOT/'model'/n,arcname=n,recursive=False)
    with tarfile.open(partial,'r:gz') as tar:
        assert set(tar.getnames())==set(inventory) and len(tar.getmembers())==8
        for member in tar:
            assert member.isfile();h=hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
            assert h.hexdigest()==inventory[member.name]['sha256']
    digest=old.sha(partial);path=out/NAME;partial.rename(path)
    (out/'SHA256SUMS.txt').write_text(f'{digest}  {NAME}\n')
    old.save(out/'model_archive.json',dict(name=NAME,bytes=path.stat().st_size,sha256=digest,members=inventory,
        only_changed_model=old.RAW_NAME,ssl_images=9000,supervised_images=8998,archived_hashes_verified=True,
        calibration_unchanged=True,official_platform_test_pending=True))
    old.save(ROOT/'package_ready.json',dict(path=str(path),bytes=path.stat().st_size,sha256=digest,
        archive_verified=True,runtime_passed=True,automatic_submission=False,at=time.strftime('%FT%T%z')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','worker','run','stage','runtime','archive'));p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.command=='worker':worker(a.smoke)
    else:globals()[a.command]()
