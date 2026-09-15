"""Up to twenty supervised fits plus one external pretrain, GPUs1-7 only."""
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

from audit_region_search import ROOT,RECIPES,run_path,audit
from final_candidate_core import save_json
from run_submission_sweep import gpu_busy

CODE=Path(__file__).resolve().parent


def job(recipe,fold):
    ssl=recipe=="native_ssl"; k=1 if ssl else 2
    env=dict(ROOT="datasets/clear_ec_official",OUT_ROOT=str(ROOT/"runs"),TAG=recipe,FOLD=fold,SEED=42,FOLD_SEED=2026,NFOLDS=5,
        YSRC="slidesmooth",EPOCHS=35,BS=48,CROP=256,K_CROPS=k,CROP_MODE="random",FEATURE_AGG=0,POOL="mean",BAG_POOL="mean",
        LOSS_MODE="multitask",OBJECTIVE="mixed_mape",SAM_RHO=0,MAX_TRAIN_STEPS=0,MAX_VAL_IMAGES=0,EVAL_D8=0,FINAL_D8=1,
        EVAL_IMAGE_BS=4,ENC_LR=1e-5,HEAD_LR=1e-3,WEIGHT_DECAY=1e-4,SELECT_MODE="best",EXCLUDE_FOLD=-1,NORM_MODE="fixed",
        REGION_POLICY="central" if ssl else recipe,ENCODER_INIT="",ENCODER_SHA256="",SAVE_CHECKPOINT=1,
        CPSAM_MODEL="cpsam_models/cpsam_v2",OMP_NUM_THREADS=4,PYTHONNOUSERSITE=1,PYTHONUNBUFFERED=1)
    return dict(key=f"{recipe}_f{fold}",recipe=recipe,fold=fold,env=env)


def valid(j,smoke=False):
    path=run_path(j["recipe"],j["fold"],smoke)
    if not (path/"result.json").exists(): return False
    r=json.loads((path/"result.json").read_text())
    if not np.isfinite(r["final_metrics"]).all(): raise ValueError("Nonfinite final score")
    with np.load(path/"best_oof.npz",allow_pickle=True) as p,np.load(path/"split_indices.npz") as s:
        assert len(p["indices"])==(2 if smoke else (1790 if j["fold"]==4 else 1791))
        assert np.isfinite(p["pred"]).all() and (p["pred"]>0).all()
        np.testing.assert_array_equal(p["indices"],s["validation"])
        assert not np.intersect1d(p["indices"],s["train"]).size
    return True


def main():
    ROOT.mkdir(parents=True,exist_ok=True); (ROOT/"logs").mkdir(exist_ok=True)
    with (ROOT/"controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/"status.json").exists(): raise RuntimeError("Existing region queue; inspect before restart")
        planned=[job(r,f) for r in RECIPES for f in (0,1)]
        save_json(ROOT/"manifest.json",dict(screen_jobs=planned,confirmation_folds=[2,3,4],maximum_supervised_jobs=20,
            allowed_gpus=list(range(1,8)),gpu0_reserved=True,pretrain_steps=6000,pretrain_batch=32,
            gate="add6 raw delta <= -0.01 and both folds improve OR replace4 raw delta <= -0.005 and both folds improve; confirm each passing recipe on folds2-4; no automatic deployment",
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.glob("*.py")}))
        pending=[j for j in planned if j["recipe"]!="native_ssl"]
        pending.append(dict(key="external_ssl",recipe="external_ssl",fold=-1,env={}))
        active={}; done=[]; failed={}; screened={};confirmed={};ssl_released=False;started=time.time()
        def snapshot(stage="running"):
            save_json(ROOT/"status.json",dict(stage=stage,updated_at=time.strftime("%FT%T%z"),elapsed_seconds=time.time()-started,
                pending=[j["key"] for j in pending],done=done,failed=failed,screened=screened,confirmed=confirmed,
                active={str(g):dict(key=x["job"]["key"],pid=x["process"].pid,stage=x["stage"],log=str(x["log"])) for g,x in active.items()}))
        snapshot()
        while pending or active:
            for gpu in range(1,8):
                if gpu in active or gpu_busy(gpu) or not pending: continue
                j=pending.pop(0); pre=j["recipe"]=="external_ssl"
                smoke=not ((ROOT/"pretrain_smoke/complete.json").exists() if pre else valid(j,True))
                env=dict(os.environ); env.update({k:str(v) for k,v in j["env"].items()})
                env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",PYTHONNOUSERSITE="1",PYTHONUNBUFFERED="1")
                if pre:
                    command=[sys.executable,str(CODE/"train_native_external_ssl.py")]+(["--smoke"] if smoke else [])
                    path=ROOT/("pretrain_smoke" if smoke else "pretrain")
                else:
                    if j["recipe"]=="native_ssl":
                        cp=json.loads((ROOT/"pretrain/complete.json").read_text())
                        env.update(ENCODER_INIT=str(ROOT/"pretrain/encoder_final.pt"),ENCODER_SHA256=cp["encoder_sha256"])
                    if smoke: env.update(OUT_ROOT=str(ROOT/"smoke"),BS="4",EPOCHS="1",MAX_TRAIN_STEPS="1",MAX_VAL_IMAGES="2")
                    command=[sys.executable,str(CODE/"train_region_method.py")]; path=run_path(j["recipe"],j["fold"],smoke)
                if path.exists():
                    failed[j["key"]]=f"Partial path exists: {path}";continue
                stage="smoke" if smoke else "train";log=ROOT/"logs"/f"{j['key']}_{stage}.log";stream=log.open("a")
                p=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=dict(job=j,process=p,stream=stream,stage=stage,log=log,started=time.time(),terminated=None)
                print(f"gpu={gpu} {j['key']} {stage} pid={p.pid}",flush=True)
            for gpu,x in list(active.items()):
                p=x["process"];limit=900 if x["stage"]=="smoke" else 12*3600
                if p.poll() is None and time.time()-x["started"]>limit:
                    if x["terminated"] is None: os.killpg(p.pid,signal.SIGTERM);x["terminated"]=time.time()
                    elif time.time()-x["terminated"]>60: os.killpg(p.pid,signal.SIGKILL)
                if p.poll() is None: continue
                x["stream"].close();j=x["job"];smoke=x["stage"]=="smoke";pre=j["recipe"]=="external_ssl"
                try:
                    good=p.returncode==0 and ((ROOT/("pretrain_smoke" if smoke else "pretrain")/"complete.json").exists() if pre else valid(j,smoke))
                except Exception as e: good=False;failed[j["key"]]=repr(e)
                if good:
                    if smoke: pending.insert(0,j)
                    else: done.append(j["key"])
                else: failed.setdefault(j["key"],f"{x['stage']} exit={p.returncode}, log={x['log']}")
                del active[gpu]
            if "external_ssl" in done and not ssl_released:
                pending.extend(job("native_ssl",f) for f in (0,1));ssl_released=True
            for recipe in RECIPES:
                if recipe not in screened and all(f"{recipe}_f{f}" in done for f in (0,1)):
                    try:
                        result=audit(recipe,[0,1],"screen");screened[recipe]=bool(result["passes_screen"])
                        if result["passes_screen"]: pending.extend(job(recipe,f) for f in (2,3,4))
                    except Exception as e: failed[f"audit_{recipe}"]=repr(e);screened[recipe]=False
                if recipe not in confirmed and all(f"{recipe}_f{f}" in done for f in range(5)):
                    try:
                        audit(recipe,[2,3,4],"confirmation");result=audit(recipe,list(range(5)),"five_fold")
                        confirmed[recipe]=dict(delta_add6=result["delta_add6"],delta_replace4=result["delta_replace4"],deployment_approved=False)
                    except Exception as e: failed[f"confirmation_{recipe}"]=repr(e);confirmed[recipe]=dict(error=str(e))
            snapshot()
            if pending or active: time.sleep(10)
        snapshot("complete")
        save_json(ROOT/"complete.json",dict(done=done,failed=failed,screened=screened,confirmed=confirmed,
            elapsed_seconds=time.time()-started,completed_at=time.strftime("%FT%T%z"),deployment_approved=False))
        if failed: raise SystemExit(1)


if __name__=="__main__": main()
