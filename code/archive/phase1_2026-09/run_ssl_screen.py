"""Finite four-GPU controller. Smoke barrier, no retry, stop all on failure."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from prepare_ssl_screen import ROOT, sha


def save(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)); tmp.replace(path)


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert not (ROOT/'status.json').exists(), 'Refusing an existing queue restart'
    for path,digest in manifest['source_sha256'].items():
        assert sha(path)==digest,path
    import torch
    assert torch.cuda.device_count()==4,'Four CUDA-visible GPUs required'
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.used','--format=csv,noheader'],text=True)
    save(ROOT/'gpu_inventory.json',dict(inventory=inventory,torch_version=torch.__version__))
    subprocess.run([sys.executable,str(ROOT/'source/test_ssl_native_core.py')],check=True,timeout=120)
    jobs={0:('ssl35',0),1:('ssl35',1),2:('supervised55',0),3:('supervised55',1)}
    (ROOT/'logs').mkdir(exist_ok=False)
    active={}; done=[]; smoke_done=[]; failed={}
    started=time.time(); deadline=started+manifest['max_queue_seconds']
    def stop(signum=None,frame=None):
        for item in active.values():
            process=item['process']
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
        if signum:
            save(ROOT/'interrupted.json',dict(signal=signum,at=time.strftime('%FT%T%z')))
            raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(gpu,smoke):
        recipe,fold=jobs[gpu];name=f'{recipe}_fold{fold}'
        # UUIDs are visible only for assigned GPUs; do not touch other workloads.
        used=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        assert used<100,'Assigned GPU unexpectedly occupied'
        command=[sys.executable,str(ROOT/'source/train_ssl_screen.py'),'--fold',str(fold),'--recipe',recipe]
        if smoke:command.append('--smoke')
        stage='smoke' if smoke else 'full'
        with (ROOT/'logs'/f'{name}_{stage}.log').open('x') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
        active[gpu]=dict(process=process,start=time.time(),name=name,smoke=smoke)
        print(json.dumps(dict(launched=name,gpu=gpu,stage=stage,pid=process.pid)),flush=True)
    try:
        for gpu in jobs:launch(gpu,True)
        full_started=False
        while active:
            for gpu,item in list(active.items()):
                rc=item['process'].poll()
                limit=manifest['smoke_timeout_seconds'] if item['smoke'] else manifest['max_worker_seconds']
                if rc is None and (time.time()>deadline or time.time()-item['start']>limit):
                    failed[item['name']]='timeout';stop();break
                if rc is None:continue
                del active[gpu]
                result_path=ROOT/('smoke' if item['smoke'] else 'runs')/item['name']/'result.json'
                if rc!=0 or not result_path.exists():
                    failed[item['name']]=f'exit={rc}, result_exists={result_path.exists()}';stop();break
                result=json.loads(result_path.read_text())
                assert result['strict_reload_passed']
                if item['name'].startswith('supervised55'):
                    assert (result_path.parent/'worker_complete.json').exists()
                (smoke_done if item['smoke'] else done).append(item['name'])
            if failed:break
            if not active and len(smoke_done)==4 and not full_started:
                full_started=True
                for gpu in jobs:launch(gpu,False)
            save(ROOT/'status.json',dict(updated_at=time.strftime('%FT%T%z'),started_unix=started,
                deadline_unix=deadline,done=done,failed=failed,smoke_done=smoke_done,
                active={str(g):dict(pid=v['process'].pid,name=v['name'],smoke=v['smoke']) for g,v in active.items()}))
            if active:time.sleep(10)
    except Exception as error:
        failed['controller']=repr(error);stop()
    save(ROOT/'complete.json',dict(completed_at=time.strftime('%FT%T%z'),done=done,
        failed=failed,smoke_done=smoke_done,elapsed_seconds=time.time()-started,deployment_approved=False))
    if failed:raise RuntimeError(failed)
    subprocess.run([sys.executable,str(ROOT/'source/audit_ssl_screen.py')],check=True,timeout=1800)
    save(ROOT/'queue_finished.json',dict(finished_at=time.strftime('%FT%T%z'),audit_completed=True,
        gpu_container_will_exit=True,original_pvc_retained=True,auto_promotion=False))


if __name__=='__main__':
    main()
