"""Finite four-GPU target-policy screen; all parity smokes precede full fits."""
import json
import os
import signal
import subprocess
import sys
import time
from prepare_target_screen import ROOT,sha


def save(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert not (ROOT/'status.json').exists(),'Never restart an existing queue'
    for p,h in manifest['source_sha256'].items():assert sha(p)==h,p
    import torch
    assert torch.cuda.device_count()==manifest['gpu_count']==4
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.used','--format=csv,noheader'],text=True)
    save(ROOT/'gpu_inventory.json',dict(inventory=inventory,torch_version=torch.__version__))
    subprocess.run([sys.executable,str(ROOT/'source/test_target_screen.py')],check=True,timeout=120)
    jobs={0:('base',0),1:('base',1),2:('ssl',0),3:('ssl',1)}
    active={};done=[];smoke_done=[];failed={}
    started=time.time();deadline=started+manifest['max_queue_seconds']
    (ROOT/'logs').mkdir(exist_ok=False)
    def stop(signum=None,frame=None):
        for item in active.values():
            p=item['process']
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        if signum:
            save(ROOT/'interrupted.json',dict(signal=signum,at=time.strftime('%FT%T%z')))
            raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(gpu,policy,smoke):
        initialization,fold=jobs[gpu];name=f'{initialization}_{policy}_fold{fold}'
        used=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        assert used<100,'Assigned GPU unexpectedly occupied'
        command=[sys.executable,str(ROOT/'source/train_target_screen.py'),'--fold',str(fold),
                 '--initialization',initialization,'--policy',policy]
        if smoke:command.append('--smoke')
        with (ROOT/'logs'/f'{name}_{"smoke" if smoke else "full"}.log').open('x') as log:
            p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
        active[gpu]=dict(process=p,name=name,policy=policy,smoke=smoke,start=time.time())
        print(json.dumps(dict(launched=name,gpu=gpu,smoke=smoke,pid=p.pid)),flush=True)
    try:
        for gpu in jobs:launch(gpu,'slidesmooth',True)
        full_started=False
        while active:
            for gpu,item in list(active.items()):
                rc=item['process'].poll()
                limit=manifest['smoke_timeout_seconds'] if item['smoke'] else manifest['max_worker_seconds']
                if rc is None and (time.time()>deadline or time.time()-item['start']>limit):
                    failed[item['name']]='timeout';stop();break
                if rc is None:continue
                del active[gpu]
                folder=ROOT/('smoke' if item['smoke'] else 'runs')/item['name']
                if rc!=0 or not (folder/'result.json').exists():
                    failed[item['name']]=f'exit={rc}, result_exists={(folder/"result.json").exists()}';stop();break
                result=json.loads((folder/'result.json').read_text())
                assert result['strict_reload_passed']
                if item['smoke']:
                    smoke_done.append(item['name'])
                    if item['policy']=='slidesmooth':launch(gpu,'raw',True)
                else:done.append(item['name'])
            if failed:break
            if not active and len(smoke_done)==8 and not full_started:
                for initialization,fold in jobs.values():
                    pair=[json.loads((ROOT/'smoke'/f'{initialization}_{policy}_fold{fold}'/'result.json').read_text()) for policy in ('raw','slidesmooth')]
                    assert pair[0]['training']['first_three_batches_sha256']==pair[1]['training']['first_three_batches_sha256']
                    if initialization=='base':assert pair[1]['smoothed_base_smoke_parity']
                full_started=True
                for gpu in jobs:launch(gpu,'raw',False)
            save(ROOT/'status.json',dict(updated_at=time.strftime('%FT%T%z'),started_unix=started,
                deadline_unix=deadline,done=done,failed=failed,smoke_done=smoke_done,
                active={str(g):dict(pid=v['process'].pid,name=v['name'],smoke=v['smoke']) for g,v in active.items()}))
            if active:time.sleep(10)
    except Exception as error:
        failed['controller']=repr(error);stop()
    save(ROOT/'complete.json',dict(completed_at=time.strftime('%FT%T%z'),done=done,failed=failed,
                                  smoke_done=smoke_done,elapsed_seconds=time.time()-started,deployment_approved=False))
    if failed:raise RuntimeError(failed)
    try:
        subprocess.run([sys.executable,str(ROOT/'source/audit_target_screen.py')],check=True,timeout=1800)
    except Exception as error:
        save(ROOT/'audit_failed.json',dict(error=repr(error),at=time.strftime('%FT%T%z')))
        raise
    save(ROOT/'queue_finished.json',dict(finished_at=time.strftime('%FT%T%z'),audit_completed=True,
                                        gpu_container_will_exit=True,original_pvc_retained=True,auto_promotion=False))


if __name__=='__main__':main()
