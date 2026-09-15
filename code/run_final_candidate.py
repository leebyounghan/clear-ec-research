"""Finite three fits and five inference audits with automatic same-scope postflight."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from final_candidate_core import save_json
from run_submission_sweep import gpu_busy

ROOT=Path('experiments/final_candidate_20260904')
LOG=Path('logs/final_candidate_20260904')


def main():
    ROOT.mkdir(parents=True,exist_ok=True); LOG.mkdir(parents=True,exist_ok=True)
    with (ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'complete.json').exists(): print('already complete'); return
        if (ROOT/'status.json').exists() and json.loads((ROOT/'status.json').read_text()).get('active'):
            raise RuntimeError('Prior active state: inspect workers before restart')
        for cmd in ['code/prepare_final9000_candidate.py','code/plan_final_candidate.py']:
            subprocess.run([sys.executable,cmd],check=True,timeout=900)
        pending=[dict(key=f'seed{s}',kind='train',seed=s,gpu=g) for g,s in enumerate([42,77,2026])]
        pending += [dict(key=f'fold{f}',kind='audit',fold=f,gpu=f+3) for f in range(5)]
        files=['code/final_candidate_core.py','code/train_final_candidate.py','code/audit_candidate_fold.py',
               'code/prepare_final9000_candidate.py','code/plan_final_candidate.py','code/run_final_candidate.py',
               'code/package_final_candidate.py','submission_candidate_20260904/inference.py']
        manifest=dict(jobs=pending,source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files})
        path=ROOT/'manifest.json'
        if path.exists(): assert json.loads(path.read_text())==manifest
        else: save_json(path,manifest)
        active={}; done=[]; failed={}; start=time.time()
        def snapshot():
            save_json(ROOT/'status.json',dict(updated_at=time.strftime('%FT%T%z'),elapsed_seconds=time.time()-start,
                done=done,failed=failed,pending=[j['key'] for j in pending],active={str(g):dict(
                key=v['job']['key'],pid=v['p'].pid,stage=v['stage'],log=v['log'],started=v['started']) for g,v in active.items()}))
        while pending or active:
            for j in list(pending):
                g=j['gpu']
                if g in active or gpu_busy(g): continue
                if j['kind']=='train':
                    smoke=not (ROOT/'smoke'/j['key']/'result.json').exists()
                    stage='smoke' if smoke else 'train'
                    cmd=[sys.executable,'code/train_final_candidate.py','--seed',str(j['seed'])]+(['--smoke'] if smoke else [])
                else: stage='audit'; cmd=[sys.executable,'code/audit_candidate_fold.py','--fold',str(j['fold'])]
                env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='4',PYTHONNOUSERSITE='1')
                log=LOG/f"{j['key']}_{stage}.log"; stream=log.open('a')
                p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                active[g]=dict(job=j,p=p,stage=stage,log=str(log),stream=stream,started=time.time(),term=None)
                pending.remove(j); print(f'{time.strftime("%FT%T%z")} gpu={g} {j["key"]} {stage} pid={p.pid}',flush=True); snapshot()
            time.sleep(10)
            for g,v in list(active.items()):
                p=v['p']; limit=600 if v['stage']=='smoke' else 8*3600
                if p.poll() is None and time.time()-v['started']>limit:
                    if v['term'] is None: os.killpg(p.pid,signal.SIGTERM); v['term']=time.time()
                    elif time.time()-v['term']>60: os.killpg(p.pid,signal.SIGKILL)
                rc=p.poll()
                if rc is None: continue
                v['stream'].close(); j=v['job']
                directory=ROOT/('smoke' if v['stage']=='smoke' else 'runs')/j['key'] if j['kind']=='train' else ROOT/'inference_audit'/j['key']
                good=rc==0 and (directory/'result.json').exists()
                if good:
                    if v['stage']=='smoke': pending.append(j)
                    else: done.append(j['key'])
                else: failed[j['key']]=f'{v["stage"]} exit {rc}; inspect {v["log"]}'
                print(f'{j["key"]} {v["stage"]} complete={good}',flush=True); del active[g]
            snapshot()
        if not failed:
            with (LOG/'postflight.log').open('a') as stream:
                try:
                    subprocess.run([sys.executable,'code/package_final_candidate.py'],stdout=stream,
                        stderr=subprocess.STDOUT,check=True,timeout=1800,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4'))
                except Exception as exc: failed['postflight']=repr(exc)
        snapshot(); save_json(ROOT/'complete.json',dict(done=done,failed=failed,completed_at=time.strftime('%FT%T%z')))
        if failed: raise SystemExit(1)


if __name__=='__main__': main()
