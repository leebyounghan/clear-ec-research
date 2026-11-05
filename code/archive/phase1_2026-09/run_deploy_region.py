"""Finite two-GPU all-data preparation; never promotes or submits models."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path("experiments/deploy_region_20260907")
SOURCE = ROOT / "source"
FROZEN = Path("experiments/region_search_20260905/source/code")
sys.path.insert(0, str(FROZEN))
from final_candidate_core import save_json


def main():
    if (ROOT / "manifest.json").exists():
        raise FileExistsError("already launched; inspect, never duplicate")
    audit = Path("experiments/region_seedconfirm_20260906/manual_fivefold_audit_20260907.json")
    evidence = json.loads(audit.read_text())
    jobs = []
    for gpu, policy in enumerate(("anchor_quality", "full_quality")):
        assert evidence["policies"][policy]["both_new_seeds_pass"]
        epochs = []
        for fold in range(5):
            root = "region_seedcheck_20260906" if fold < 2 else "region_seedconfirm_20260906"
            run = Path("experiments") / root / "runs" / f"{policy}_f{fold}_seed77_c256k2_mean_multitask_slidesmooth"
            epochs.append(json.loads((run / "result.json").read_text())["best_epoch"])
        target = sorted(epochs)[2] * 149
        assert target == (4470 if policy == "anchor_quality" else 4917)
        jobs.append(dict(policy=policy, gpu=gpu, seed=77, oof_best_epochs=epochs, selected_steps=target))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in list(FROZEN.glob("*.py")) + list(SOURCE.glob("*.py"))}
    for directory in ("logs", "smoke", "runs"):
        (ROOT / directory).mkdir(exist_ok=False)
    save_json(ROOT / "manifest.json", dict(jobs=jobs, source_sha256=hashes,
        audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
        seed_rule="Use first replication seed77 consistently, not lowest observed score seed2026",
        step_rule="Median five OOF best epochs times149, same5215-update cosine schedule",
        budget_gpu_hours=11, per_fit_timeout_seconds=21600, smoke_timeout_seconds=1200,
        scope="All-data fit and native smoke preparation only; no validation score, staged changes, upload or submission",
        started_at=time.strftime("%FT%T%z")))
    active, done, failed = {}, [], {}

    def launch(job, smoke):
        gpu = job["gpu"]
        usage = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        if int(usage.strip()) > 100:
            raise RuntimeError(f"GPU{gpu} occupied; do not interrupt")
        for path, digest in hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source changed: {path}")
        stage = "smoke" if smoke else "train"
        cmd = [sys.executable, str(SOURCE / "train_deploy_region.py"), "--recipe", job["policy"], "--output-root", str(ROOT)]
        if smoke:
            cmd.append("--smoke")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1")
        with (ROOT / "logs" / f'{job["policy"]}_{stage}.log').open("x") as log:
            process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(job["policy"], stage, process.pid, flush=True)
        active[job["policy"]] = dict(job=job, process=process, smoke=smoke, start=time.time(), command=cmd)

    for job in jobs:
        try:
            launch(job, True)
        except Exception as error:
            failed[job["policy"]] = repr(error)
    started = time.time()
    while active:
        for policy, entry in list(active.items()):
            process, smoke = entry["process"], entry["smoke"]
            rc = process.poll()
            timeout = 1200 if smoke else 21600
            if rc is None and time.time() - entry["start"] > timeout:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                rc = -999
            if rc is None:
                continue
            del active[policy]
            try:
                if rc:
                    raise RuntimeError(f"exit {rc}")
                run = ROOT / ("smoke" if smoke else "runs") / policy
                result = json.loads((run / "result.json").read_text())
                config = json.loads((run / "config.json").read_text())
                assert result["selected_steps"] == (1 if smoke else entry["job"]["selected_steps"])
                assert config["region_policy"] == policy and config["seed"] == 77 and config["train_rows"] == 8998
                assert (run / "selected.pt").stat().st_size > 1000000000
                if smoke:
                    launch(entry["job"], False)
                else:
                    done.append(policy)
            except Exception as error:
                failed[policy] = repr(error)
        save_json(ROOT / "status.json", dict(updated_at=time.strftime("%FT%T%z"), done=done, failed=failed,
            active={key: dict(pid=e["process"].pid, gpu=e["job"]["gpu"], stage="smoke" if e["smoke"] else "train", command=e["command"])
                    for key, e in active.items()}))
        if active:
            time.sleep(10)
    save_json(ROOT / "complete.json", dict(done=done, failed=failed, elapsed_seconds=time.time()-started,
        completed_at=time.strftime("%FT%T%z"), deployment_approved=False, requires_native_inference_audit=True))


if __name__ == "__main__":
    main()
