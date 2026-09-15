"""Exactly two GPU workers, two bounded branches each, then CPU audit. No retries."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime,timezone

ROOT=Path('experiments/statres_screen_20260908')
sys.path.insert(0,str(ROOT/'source/frozen'))
from final_candidate_core import save_json


def main():
    if (ROOT/'manifest.json').exists(): raise FileExistsError('already launched')
    assert int(subprocess.check_output(['nvidia-smi','--query-gpu=count','--format=csv,noheader,nounits'],text=True).splitlines()[0])==2
    deadline=datetime(2026,9,9,1,0,tzinfo=timezone.utc).timestamp()
    assert time.time()<deadline-3.5*3600, 'insufficient time for this finite screen'
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'source').rglob('*.py'))}
    prior=json.loads(Path('experiments/specialize_screen_20260908/manifest.json').read_text())
    for p,h in prior['source_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
        hashes[p]=h
    # The historical audit dependencies are also frozen and hashed before launch.
    for name in ('audit_region_search.py','audit_method_screen.py','audit_submission_ensembles.py'):
        p=Path('experiments/region_search_20260905/source/code')/name
        hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (ROOT/'logs').mkdir(exist_ok=False)
    save_json(ROOT/'manifest.json',dict(started_at=time.strftime('%FT%T%z'),folds=[0,1],gpu_count=2,
        source_sha256=hashes,common_reused_updates=2980,branch_updates=1490,branches=['mean','std'],
        full_worker_timeout_seconds=16200,smoke_timeout_seconds=1800,absolute_deadline_utc='2026-09-09T01:00:00Z',
        max_full_gpu_hours=9,auto_retry=False,automatic_confirmation=False,
        screen_gate='std CV/HEX mean MAPE beats capacity-matched mean control in both folds and >=0.03 aggregate; fixed 50/50 CV/HEX blend with current6 improves both folds and >=0.01 overall aggregate; blend beats control blend in both folds. CD unchanged in blend.',
        scope='Same fold TRAIN-ONLY fixed terminal common checkpoints; mean residual vs std residual; fixed terminal evaluation only. No candidate writes.'))
    active,done,failed={},[],{}
    def launch(fold,smoke):
        used=int(subprocess.check_output(['nvidia-smi',f'--id={fold}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        assert used<100,'GPU occupied'
        for p,h in hashes.items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
        stage='smoke' if smoke else 'full'
        cmd=[sys.executable,str(ROOT/'source/train_statres_screen.py'),'--fold',str(fold)]
        if smoke: cmd+=['--smoke']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(fold),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        with (ROOT/'logs'/f'fold{fold}_{stage}.log').open('x') as log:
            p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[fold]=dict(process=p,smoke=smoke,start=time.time())
        print(fold,stage,p.pid,flush=True)
    for f in (0,1):
        try: launch(f,True)
        except Exception as error: failed[str(f)]=repr(error)
    while active:
        for fold,item in list(active.items()):
            p=item['process']; rc=p.poll()
            if rc is None and (time.time()>deadline or time.time()-item['start']>(1800 if item['smoke'] else 16200)):
                os.killpg(p.pid,signal.SIGTERM)
                try: p.wait(timeout=20)
                except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL); p.wait()
                rc=-999
            if rc is None: continue
            del active[fold]
            try:
                assert rc==0,f'worker exit {rc}'
                r=json.loads((ROOT/('smoke' if item['smoke'] else 'runs')/f'fold{fold}/result.json').read_text())
                assert set(r['branches'])=={'mean','std'}
                if item['smoke']: launch(fold,False)
                else: done.append(fold)
            except Exception as error: failed[str(fold)]=repr(error)
        save_json(ROOT/'status.json',dict(updated_at=time.strftime('%FT%T%z'),done=done,failed=failed,
            active={str(f):dict(pid=v['process'].pid,gpu=f,smoke=v['smoke']) for f,v in active.items()}))
        if active: time.sleep(10)
    completion=dict(completed_at=time.strftime('%FT%T%z'),done=done,failed=failed,
        deployment_approved=False,audit_in_progress=not failed,release_idle_pod_after_persistent_artifacts_check=True)
    save_json(ROOT/'complete.json',completion)
    if not failed:
        try:
            with (ROOT/'logs/audit.log').open('x') as log:
                subprocess.run([sys.executable,str(ROOT/'source/audit_statres_screen.py')],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            completion['audit_completed']=True
        except Exception as error: completion['audit_error']=repr(error)
    completion['audit_in_progress']=False
    completion['controller_completed_at']=time.strftime('%FT%T%z')
    save_json(ROOT/'complete.json',completion)


if __name__=='__main__': main()
