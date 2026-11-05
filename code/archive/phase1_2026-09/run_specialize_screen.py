"""Two GPUs only, one fold per GPU; smoke then fixed finite training; no retries."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path("experiments/specialize_screen_20260908")
sys.path.insert(0, str(ROOT / "source/frozen"))
from final_candidate_core import save_json


def main():
    if (ROOT / "manifest.json").exists():
        raise FileExistsError("queue already launched")
    assert int(subprocess.check_output(["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"], text=True).splitlines()[0]) == 2
    paths = sorted((ROOT / "source").rglob("*.py"))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    deadline = datetime(2026, 9, 8, 23, 30, tzinfo=timezone.utc).timestamp()
    assert time.time() < deadline - 4*3600, "insufficient screening time; do not start a long queue"
    (ROOT / "logs").mkdir(exist_ok=False)
    save_json(ROOT / "manifest.json", dict(started_at=time.strftime("%FT%T%z"),
        folds=[0,1], gpu_count=2, source_sha256=hashes, common_updates=2980,
        branch_updates=745, branches=["multitask", "cd", "cv", "hex"],
        max_gpu_hours=18, worker_timeout_seconds=9*3600, smoke_timeout_seconds=1800,
        absolute_deadline_utc="2026-09-08T23:30:00Z", auto_retry=False,
        screening_gate="Each target specialist improves matched multitask target in BOTH folds and aggregate target MAPE by >=0.05 points; fixed 50/50 blend of that target with current6 improves both folds and aggregate overall by >=0.01. No calibration or fitted weights. Confirm folds2-4 before promotion.",
        caveat="Historically reused slide-group folds; fixed terminal weights avoid new checkpoint selection but this is not an untouched test.",
        scope="fresh common train-only CPSAM per fold; four matched continuations; protected candidates unchanged"))
    active, done, failed = {}, [], {}

    def launch(fold, smoke):
        used = int(subprocess.check_output(["nvidia-smi", f"--id={fold}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
        assert used < 100, "GPU occupied: never interrupt unrelated work"
        for p, h in hashes.items():
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h, p
        stage = "smoke" if smoke else "full"
        cmd = [sys.executable, str(ROOT / "source/train_specialize_screen.py"), "--fold", str(fold)]
        if smoke:
            cmd += ["--smoke"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(fold), OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1")
        with (ROOT / "logs" / f"fold{fold}_{stage}.log").open("x") as log:
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active[fold] = dict(process=p, smoke=smoke, start=time.time())
        print(fold, stage, p.pid, flush=True)

    for fold in (0,1):
        try:
            launch(fold, True)
        except Exception as error:
            failed[str(fold)] = repr(error)
    while active:
        for fold, item in list(active.items()):
            p = item["process"]; rc = p.poll()
            if rc is None and (time.time() > deadline or time.time()-item["start"] > (1800 if item["smoke"] else 9*3600)):
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL); p.wait()
                rc = -999
            if rc is None:
                continue
            del active[fold]
            try:
                assert rc == 0, f"worker exit {rc}"
                result = json.loads((ROOT / ("smoke" if item["smoke"] else "runs") / f"fold{fold}/result.json").read_text())
                assert set(result["branches"]) == {"multitask", "cd", "cv", "hex"}
                if item["smoke"]:
                    launch(fold, False)
                else:
                    done.append(fold)
            except Exception as error:
                failed[str(fold)] = repr(error)
        save_json(ROOT / "status.json", dict(updated_at=time.strftime("%FT%T%z"), done=done, failed=failed,
            active={str(f): dict(pid=v["process"].pid, gpu=f, smoke=v["smoke"]) for f,v in active.items()}))
        if active:
            time.sleep(10)
    save_json(ROOT / "complete.json", dict(completed_at=time.strftime("%FT%T%z"), done=done,
        failed=failed, deployment_approved=False, needs_screen_audit=True,
        release_idle_pod_after_persistent_artifacts_check=True))


if __name__ == "__main__":
    main()
