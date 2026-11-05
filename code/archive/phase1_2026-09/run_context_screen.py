#!/usr/bin/env python3
"""Eight bounded ablations; use idle GPUs once the prior sweep has dispatched all jobs."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from run_submission_sweep import gpu_busy

ROOT=Path('experiments/context_20260904')
LOGS=Path('logs/context_20260904')
GROUPS={
    'central_native': ('central','uniform',1,'mean'),
    'central_context': ('central','uniform',2,'attn'),
    'full_context_uniform': ('full','uniform',2,'attn'),
    'full_context_focus': ('full','focus',2,'attn'),
}


def submission_reserved(root=Path('experiments/submission_20260904'), now=None):
    """None means wait; otherwise reserve every GPU owned by the prior controller.

    Waiting until its pending list is empty prevents competing GPU launches.
    Reserving active entries also covers worker initialization before CUDA exists.
    A stale status never authorizes a new launch.
    """
    if (root/'complete.json').exists():
        return set()
    path=root/'status.json'
    if not path.exists(): return None
    if (time.time() if now is None else now)-path.stat().st_mtime > 120: return None
    state=json.loads(path.read_text())
    if state.get('pending', ['unknown']): return None
    if not state.get('active') and not state.get('done'): return None
    return {int(g) for g in state.get('active',{})}


def save(path,value):
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(value,indent=2,allow_nan=False)); temp.replace(path)


def jobs():
    out=[]
    for tag,(scope,quality,k,bag_pool) in GROUPS.items():
        for fold in [0,1]:
            env=dict(ROOT='datasets/clear_ec_official',OUT_ROOT=str(ROOT/'runs'),TAG=tag,FOLD=fold,
                SEED=42,FOLD_SEED=2026,NFOLDS=5,YSRC='slidesmooth',EPOCHS=25,BS=24,CROP=256,
                K_CROPS=k,CROP_MODE='random',POOL='mean',BAG_POOL=bag_pool,FEATURE_AGG=int(k>1),
                VIEW_SCOPE=scope,QUALITY_MODE=quality,FULLFRAME_ROOT='datasets/clear_ec_fullframe_20260904',
                LOSS_MODE='multitask',MAX_TRAIN_STEPS=0,MAX_VAL_IMAGES=0,EVAL_D8=0,FINAL_D8=1,
                EVAL_IMAGE_BS=2,ENC_LR=1e-5,HEAD_LR=1e-3,WEIGHT_DECAY=1e-4,SELECT_MODE='best',
                EXCLUDE_FOLD=-1,NORM_MODE='fixed',ENCODER_INIT='',SAVE_CHECKPOINT=1,
                CPSAM_MODEL='cpsam_models/cpsam_v2',PYTHONNOUSERSITE=1)
            suffix='' if bag_pool=='mean' else '_random_'+bag_pool
            name=f'{tag}_f{fold}_seed42_c256k{k}_mean_multitask_slidesmooth{suffix}'
            out.append(dict(key=f'{tag}_f{fold}',group=tag,fold=fold,env=env,run=str(ROOT/'runs'/name)))
    return out


def valid(job,smoke=False):
    run=Path(job['run'])
    if smoke: run=ROOT/'smoke'/run.name
    if not (run/'result.json').exists(): return False
    r=json.loads((run/'result.json').read_text())
    assert np.isfinite(r['final_metrics']).all()
    with np.load(run/'best_oof.npz',allow_pickle=True) as d,np.load(run/'split_indices.npz') as s:
        assert len(d['indices']) >= (2 if smoke else 1700)
        assert np.isfinite(d['pred']).all()
        assert np.array_equal(d['indices'],s['validation'])
        assert not np.intersect1d(d['indices'],s['train']).size
    return True


def summarize(all_jobs,done):
    results={}; references={}
    for group in GROUPS:
        group_jobs=[j for j in all_jobs if j['group']==group]
        if not all(j['key'] in done for j in group_jobs): continue
        pred=[]; target=[]; fold_scores=[]
        for j in group_jobs:
            with np.load(Path(j['run'])/'best_oof.npz',allow_pickle=True) as d:
                if j['fold'] in references: assert np.array_equal(d['ids'],references[j['fold']])
                else: references[j['fold']]=d['ids'].copy()
                p,y=d['pred'].astype(float),d['target'].astype(float)
                keep=(y>0)&np.isfinite(y)
                err=np.where(keep,100*np.abs(p-y)/np.where(keep,y,1),0)
                fold_scores.append(float((err.sum(0)/keep.sum(0)).mean()))
                pred.append(p); target.append(y)
        p,y=np.concatenate(pred),np.concatenate(target); keep=(y>0)&np.isfinite(y)
        mm=np.where(keep,100*np.abs(p-y)/np.where(keep,y,1),0).sum(0)/keep.sum(0)
        results[group]=dict(metrics=mm.tolist(),overall=float(mm.mean()),per_fold=fold_scores)
    save(ROOT/'screen_summary.json',dict(methods=results,
        scope='Two-fold screening, not final validation or leaderboard. Same seed, folds, BS24 and 25 full epochs. Context uses two encoder views versus one in native control. Focus is unlabeled image quality, never ROI GT.'))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--dry-run',action='store_true'); args=parser.parse_args()
    all_jobs=jobs()
    if args.dry_run: print(json.dumps(all_jobs,indent=2)); return
    ROOT.mkdir(parents=True,exist_ok=True); LOGS.mkdir(parents=True,exist_ok=True)
    with (ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=ROOT/'status.json'
        if old.exists():
            for v in json.loads(old.read_text()).get('active',{}).values():
                p=Path(f"/proc/{v['pid']}/cmdline")
                if p.exists() and b'train_context_oof.py' in p.read_bytes():
                    raise RuntimeError('Existing context worker alive; inspect before restart')
        source=['code/train_context_oof.py','code/fullframe_views.py','code/run_context_screen.py']
        manifest=dict(jobs=all_jobs,source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source})
        if (ROOT/'manifest.json').exists(): assert json.loads((ROOT/'manifest.json').read_text())==manifest
        else: save(ROOT/'manifest.json',manifest)
        wait_start=time.time()
        dependencies=[Path('datasets/clear_ec_fullframe_20260904/fullframe_manifest.json'),
            Path('datasets/clear_ec_fullframe_20260904/context_views_manifest.json')]
        while not all(p.exists() for p in dependencies) or submission_reserved() is None:
            save(ROOT/'status.json',dict(stage='waiting_dependencies',updated_at=time.strftime('%FT%T%z'),
                missing=[str(p) for p in dependencies if not p.exists()]+(
                    ['submission sweep must dispatch all pending jobs and have fresh status']
                    if submission_reserved() is None else []),
                active={},done=[],failed={},pending=[j['key'] for j in all_jobs]))
            if time.time()-wait_start>18*3600: raise TimeoutError('dependencies unavailable for 18 hours')
            time.sleep(15)
        done=[j['key'] for j in all_jobs if valid(j)]
        pending=[j for j in all_jobs if j['key'] not in done]
        active={}; failed={}; started=time.time()
        def snapshot():
            save(ROOT/'status.json',dict(stage='running',updated_at=time.strftime('%FT%T%z'),
                elapsed_seconds=time.time()-started,done=done,failed=failed,pending=[j['key'] for j in pending],
                active={str(g):dict(key=v['job']['key'],pid=v['p'].pid,stage=v['stage'],log=v['log'],started=v['started']) for g,v in active.items()}))
        while pending or active:
            reserved=submission_reserved()
            for gpu in range(8):
                if reserved is None or gpu in reserved or gpu in active or not pending or gpu_busy(gpu): continue
                j=pending.pop(0); smoke=not valid(j,smoke=True)
                env=os.environ.copy(); env.pop('WAIT_PID',None)
                env.update({k:str(v) for k,v in j['env'].items()})
                env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4')
                if smoke: env.update(OUT_ROOT=str(ROOT/'smoke'),BS='4',EPOCHS='1',MAX_TRAIN_STEPS='1',MAX_VAL_IMAGES='2')
                stage='smoke' if smoke else 'train'
                log=LOGS/f"{j['key']}_{stage}.log"; stream=log.open('a')
                p=subprocess.Popen([sys.executable,'code/train_context_oof.py'],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=dict(job=j,p=p,stream=stream,stage=stage,log=str(log),started=time.time(),term=None)
                print(f"gpu={gpu} {stage} {j['key']} pid={p.pid}",flush=True); snapshot()
            time.sleep(10)
            for gpu,v in list(active.items()):
                p=v['p']; limit=600 if v['stage']=='smoke' else 12*3600
                if p.poll() is None and time.time()-v['started']>limit:
                    if v['term'] is None: os.killpg(p.pid,signal.SIGTERM); v['term']=time.time()
                    elif time.time()-v['term']>60: os.killpg(p.pid,signal.SIGKILL)
                rc=p.poll()
                if rc is None: continue
                v['stream'].close(); j=v['job']
                try: good=rc==0 and valid(j,smoke=v['stage']=='smoke')
                except Exception as exc: good=False; failed[j['key']]=repr(exc)
                if good:
                    if v['stage']=='smoke': pending.insert(0,j)
                    else: done.append(j['key'])
                else: failed.setdefault(j['key'],f"{v['stage']} exit {rc}")
                print(f"finished {j['key']} {v['stage']} valid={good}",flush=True)
                del active[gpu]
            snapshot(); summarize(all_jobs,done)
        save(ROOT/'complete.json',dict(completed_at=time.strftime('%FT%T%z'),done=done,failed=failed))
        summarize(all_jobs,done)
        if failed: raise SystemExit(1)


if __name__=='__main__': main()
