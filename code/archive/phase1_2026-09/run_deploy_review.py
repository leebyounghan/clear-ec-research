"""Bounded corrected fit and five inference audits; existing experiments are immutable."""
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

ROOT = Path("experiments/deploy_review_20260905")
CODE = Path(__file__).resolve().parent


def main():
    ROOT.mkdir(parents=True, exist_ok=True); (ROOT/"logs").mkdir(exist_ok=True)
    with (ROOT/"controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT/"complete.json").exists():
            print("already complete",flush=True); return
        if (ROOT/"status.json").exists():
            raise RuntimeError("Previous review exists; inspect processes and partial results before restart")
        files = list(CODE.glob("*.py")) + [p for p in (CODE.parent/"submission_candidate_20260905").iterdir() if p.is_file() and not p.name.startswith(".")]
        save_json(ROOT/"manifest.json",dict(source_sha256={str(p.relative_to(CODE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            scope="One corrected all-data predbag fit, five full OOF numerical audits, safe staging. No new recipe, pod lifecycle or upload."))
        pending = [dict(key="fit", script="train_deploy_methods.py", args=["--recipe","predbag2_35","--output-root",str(ROOT)],
            result=ROOT/"runs/predbag2_35/result.json", gpu=0)]
        pending += [dict(key=f"audit{f}",script="audit_deploy_methods_fold.py",args=["--fold",str(f)],
                        result=ROOT/f"inference_audit/fold{f}/result.json",gpu=None) for f in range(5)]
        active, done, failed = {}, [], {}; started = time.time()
        def snapshot(stage="running"):
            save_json(ROOT/"status.json",dict(stage=stage, updated_at=time.strftime("%FT%T%z"), elapsed_seconds=time.time()-started,
                pending=[j["key"] for j in pending], done=done, failed=failed,
                active={str(g):dict(key=x["job"]["key"],pid=x["process"].pid,log=str(x["log"])) for g,x in active.items()}))
        snapshot()
        while pending or active:
            for gpu in range(8):
                if gpu in active or gpu_busy(gpu):
                    continue
                candidates = [j for j in pending if (j["gpu"] == gpu or (j["gpu"] is None and gpu != 0))]
                if not candidates:
                    continue
                j=candidates[0]; pending.remove(j)
                if j["result"].exists():
                    raise RuntimeError(f"Unexpected prior output {j['result']}")
                log=ROOT/"logs"/(j["key"]+".log"); stream=log.open("a")
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",PYTHONUNBUFFERED="1",PYTHONNOUSERSITE="1")
                p=subprocess.Popen([sys.executable,str(CODE/j["script"]),*j["args"]],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=dict(job=j,process=p,stream=stream,log=log,started=time.time(),terminated=None)
                print(f"gpu={gpu} {j['key']} pid={p.pid}",flush=True)
            for gpu,x in list(active.items()):
                p=x["process"]
                if p.poll() is None and time.time()-x["started"]>8*3600:
                    if x["terminated"] is None:
                        os.killpg(p.pid,signal.SIGTERM);x["terminated"]=time.time()
                    elif time.time()-x["terminated"]>60:
                        os.killpg(p.pid,signal.SIGKILL)
                if p.poll() is None:
                    continue
                x["stream"].close(); j=x["job"]
                if p.returncode == 0 and j["result"].exists():
                    done.append(j["key"])
                else:
                    failed[j["key"]]=dict(exit=p.returncode,log=str(x["log"]))
                del active[gpu]
            snapshot()
            if pending or active:
                time.sleep(10)
        if not failed:
            snapshot("postflight")
            with (ROOT/"logs/postflight.log").open("a") as stream:
                try:
                    subprocess.run([sys.executable,str(CODE/"package_deploy_review.py")],check=True,timeout=3600,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",OMP_NUM_THREADS="4"),stdout=stream,stderr=subprocess.STDOUT)
                except Exception as e:
                    failed["postflight"]=str(e)
        snapshot("complete")
        save_json(ROOT/"complete.json",dict(done=done,failed=failed,completed_at=time.strftime("%FT%T%z")))
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
