"""Six finite GPU audit jobs, smoke before full; no training or deployment."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path("experiments/region_inference_20260907")
FROZEN = Path("experiments/region_search_20260905/source/code")
sys.path.append(str(FROZEN))
from final_candidate_core import save_json


def main():
    if (ROOT / "manifest.json").exists():
        raise FileExistsError("audit already launched")
    completion = json.loads(Path("experiments/deploy_region_20260907/complete.json").read_text())
    assert not completion["failed"] and len(completion["done"]) == 2
    jobs = [dict(key=f"fold{f}", gpu=f, args=["--fold", str(f)]) for f in range(5)]
    jobs.append(dict(key="all_data", gpu=5, args=["--all-data"]))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
              list(FROZEN.glob("*.py")) + list((ROOT / "source").glob("*.py"))}
    for d in ("logs", "smoke", "audits"):
        (ROOT / d).mkdir(exist_ok=False)
    save_json(ROOT / "manifest.json", dict(jobs=jobs, source_sha256=hashes,
        started_at=time.strftime("%FT%T%z"), budget_gpu_hours=18, full_timeout_seconds=10800, smoke_timeout_seconds=1200,
        gates="Exact native40-view equality on every OOF image; BF16 max relative<=1e-5; FP16 max<=.05, mean<=.005, full-fold absolute MAPE drift<=.01",
        scope="Five folds both seed77 policies BF16/FP16; eight fixed native all-data images including four extras; no model or candidate changes"))
    active, done, failed = {}, [], {}

    def launch(job, smoke):
        usage = int(subprocess.check_output(["nvidia-smi", f'--id={job["gpu"]}', "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
        if usage > 100:
            raise RuntimeError("GPU occupied; never interrupt")
        for path, digest in hashes.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
        cmd = [sys.executable, str(ROOT / "source/audit_region_inference.py"), *job["args"]]
        if smoke:
            cmd.append("--smoke")
        stage = "smoke" if smoke else "full"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job["gpu"]), OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1")
        with (ROOT / "logs" / f'{job["key"]}_{stage}.log').open("x") as log:
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active[job["key"]] = dict(job=job, process=p, smoke=smoke, start=time.time(), command=cmd)
        print(job["key"], stage, p.pid, flush=True)

    for job in jobs:
        try:
            launch(job, True)
        except Exception as e:
            failed[job["key"]] = repr(e)
    while active:
        for key, e in list(active.items()):
            p = e["process"]
            rc = p.poll()
            if rc is None and time.time()-e["start"] > (1200 if e["smoke"] else 10800):
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                rc = -999
            if rc is None:
                continue
            del active[key]
            try:
                assert rc == 0, f"process exit {rc}"
                result = json.loads((ROOT / ("smoke" if e["smoke"] else "audits") / key / "result.json").read_text())
                assert all(v["passed"] for v in result["comparisons"].values())
                if e["smoke"]:
                    launch(e["job"], False)
                else:
                    done.append(key)
            except Exception as error:
                failed[key] = repr(error)
        save_json(ROOT / "status.json", dict(updated_at=time.strftime("%FT%T%z"), done=done, failed=failed,
            active={k: dict(pid=e["process"].pid, gpu=e["job"]["gpu"], stage="smoke" if e["smoke"] else "full", command=e["command"]) for k,e in active.items()}))
        if active:
            time.sleep(10)
    save_json(ROOT / "complete.json", dict(done=done, failed=failed, completed_at=time.strftime("%FT%T%z"),
        deployment_approved=False, container_and_entrypoint_validation_still_required=True))


if __name__ == "__main__":
    main()
