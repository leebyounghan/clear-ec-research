"""Finite 2-GPU queue: all smokes before four fits; exit to release resources.

No retries, candidate replacement, new data download or scored submission.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path("experiments/crop_dino_screen_20260910")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
    return h.hexdigest()


def save(path,obj,exclusive=False):
    if exclusive:
        with path.open("x") as f:json.dump(obj,f,indent=2,allow_nan=False)
    else:
        tmp=path.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)


def prepare():
    pre=json.loads((ROOT/"preflight/complete.json").read_text())
    assert pre["ready_for_crop_screen"]
    assert sha(ROOT/"preflight/guarded_selection.npz")==pre["cache_sha256"]
    assert pre["initial_weights_sha256"]=="0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667"
    paths=list((ROOT/"source").rglob("*.py"))
    # Shared helper and audit imports resolve these frozen historical dependencies.
    paths+=list(Path("experiments/specialize_screen_20260908/source/frozen").glob("*.py"))
    paths+=list(Path("experiments/region_search_20260905/source/code").glob("*.py"))
    hashes={str(p):sha(p) for p in sorted(set(paths))}
    subprocess.run([sys.executable,str(ROOT/"source/test_guarded_regions.py")],check=True)
    sys.path.insert(0,"experiments/region_search_20260905/source/code")
    import audit_region_search as historical
    import numpy as np
    from audit_submission_ensembles import metrics
    historical.run_path=lambda recipe,fold:Path("experiments/region_seedcheck_20260906/runs")/f"full_quality_f{fold}_seed77_c256k2_mean_multitask_slidesmooth"
    _,baseline=historical.compare("full_quality",(0,1))
    previous=json.loads(Path("experiments/specialize_screen_20260908/screen_audit.json").read_text())
    for path,h in previous["input_sha256"].items():assert sha(path)==h,path
    np.testing.assert_allclose(metrics(baseline["add6"],baseline["target"]),previous["current6_metrics"],atol=1e-10,rtol=0)
    save(ROOT/"manifest.json",dict(created_at=time.strftime("%FT%T%z"),source_sha256=hashes,
        folds=[0,1],policies=["control","guarded"],gpu_count=2,epochs=35,seed=77,
        max_queue_seconds=14*3600,max_worker_seconds=7*3600,smoke_timeout_seconds=900,
        absolute_deadline_utc="2026-09-11T15:00:00Z",auto_retry=False,
        screen_gate="Fixed add7 improves both folds and >=0.02 overall vs current6; beats matched control blend in both folds.",
        baseline_current6_metrics=previous["current6_metrics"],selection_sha256=pre["cache_sha256"],
        dino_status="blocked_pending_challenge_external_weight_rule; no DINO training queued",
        deployment_approved=False,scope="crop two-fold screen only; no automatic confirmation/all-data fits"),True)


def run():
    manifest=json.loads((ROOT/"manifest.json").read_text())
    assert not (ROOT/"status.json").exists(),"No restart/retry of an existing queue"
    for path,h in manifest["source_sha256"].items():assert sha(path)==h,path
    devices=subprocess.check_output(["nvidia-smi","--query-gpu=index","--format=csv,noheader"],text=True).splitlines()
    assert len(devices)==2
    deadline=min(time.time()+manifest["max_queue_seconds"],datetime(2026,9,11,15,tzinfo=timezone.utc).timestamp())
    assert deadline-time.time()>12*3600,"Insufficient deadline budget"
    (ROOT/"logs").mkdir(exist_ok=False)
    active={};done=[];failed={};smoke_done=[]
    started=time.time()

    def terminate_all(signum=None,frame=None):
        for item in active.values():
            p=item["process"]
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        if signum:raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate_all)
    signal.signal(signal.SIGINT,terminate_all)

    def launch(gpu,policy,smoke):
        used=int(subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip())
        assert used<100,"GPU already occupied; no unrelated process may be killed"
        for path,h in manifest["source_sha256"].items():assert sha(path)==h,path
        stage="smoke" if smoke else "full"
        name=f"{policy}_fold{gpu}"
        cmd=[sys.executable,str(ROOT/"source/train_guarded_screen.py"),"--fold",str(gpu),"--policy",policy]
        if smoke:cmd.append("--smoke")
        with (ROOT/"logs"/f"{name}_{stage}.log").open("x") as log:
            p=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",PYTHONNOUSERSITE="1",PYTHONUNBUFFERED="1"),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[gpu]=dict(process=p,start=time.time(),policy=policy,smoke=smoke,name=name)
        print(json.dumps(dict(launched=name,stage=stage,pid=p.pid,gpu=gpu)),flush=True)

    try:
        for gpu in (0,1):launch(gpu,"control",True)
        full_started=False
        while active:
            for gpu,item in list(active.items()):
                p=item["process"];rc=p.poll()
                limit=manifest["smoke_timeout_seconds"] if item["smoke"] else manifest["max_worker_seconds"]
                if rc is None and (time.time()>deadline or time.time()-item["start"]>limit):
                    failed[item["name"]]="timeout";terminate_all();break
                if rc is None:continue
                del active[gpu]
                folder=ROOT/("smoke" if item["smoke"] else "runs")/item["name"]
                if rc!=0 or not (folder/"result.json").exists():
                    failed[item["name"]]=f"exit={rc}, result_exists={(folder/'result.json').exists()}";terminate_all();break
                result=json.loads((folder/"result.json").read_text())
                assert result["strict_reload_passed"]
                if item["smoke"]:
                    smoke_done.append(item["name"])
                    if item["policy"]=="control":launch(gpu,"guarded",True)
                else:
                    done.append(item["name"])
                    if item["policy"]=="control":launch(gpu,"guarded",False)
            if failed:break
            if not active and len(smoke_done)==4 and not full_started:
                full_started=True
                for gpu in (0,1):launch(gpu,"control",False)
            save(ROOT/"status.json",dict(updated_at=time.strftime("%FT%T%z"),started_unix=started,deadline_unix=deadline,
                done=done,failed=failed,smoke_done=smoke_done,
                active={str(g):dict(pid=v["process"].pid,policy=v["policy"],smoke=v["smoke"],name=v["name"]) for g,v in active.items()}))
            if active:time.sleep(10)
    except Exception as error:
        failed["controller"]=repr(error);terminate_all()
    save(ROOT/"complete.json",dict(completed_at=time.strftime("%FT%T%z"),done=done,failed=failed,
        smoke_done=smoke_done,elapsed_seconds=time.time()-started,deployment_approved=False,
        needs_screen_audit=not failed),True)
    if failed:raise RuntimeError(failed)
    subprocess.run([sys.executable,str(ROOT/"source/audit_guarded_screen.py")],check=True,timeout=1200)
    save(ROOT/"queue_finished.json",dict(finished_at=time.strftime("%FT%T%z"),audit_completed=True,
        gpu_container_will_exit=True,original_pvc_retained=True),True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true")
    (prepare if parser.parse_args().prepare else run)()
