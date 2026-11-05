#!/usr/bin/env python3
"""Finite two-fit deployment queue with automatic staging and native smoke tests."""
from __future__ import annotations

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
from train_deploy_methods import RECIPES, ROOT


LOG = Path("logs/deploy_methods_20260905")


def result_path(recipe, smoke):
    return ROOT / ("smoke" if smoke else "runs") / recipe / "result.json"


def valid(recipe, smoke):
    path = result_path(recipe, smoke)
    if not path.exists():
        return False
    result = json.loads(path.read_text())
    expected = 1 if smoke else RECIPES[recipe]["selected_steps"]
    assert result["selected_steps"] == expected
    checkpoint = path.parent / "selected.pt"
    assert checkpoint.exists() and checkpoint.stat().st_size > 1_000_000_000
    return True


def main():
    ROOT.mkdir(parents=True, exist_ok=True); LOG.mkdir(parents=True, exist_ok=True)
    with (ROOT / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / "complete.json").exists():
            print("already complete", flush=True); return
        if (ROOT / "status.json").exists():
            old = json.loads((ROOT / "status.json").read_text())
            for value in old.get("active", {}).values():
                process = Path(f"/proc/{value['pid']}/cmdline")
                if process.exists() and b"train_deploy_methods.py" in process.read_bytes():
                    raise RuntimeError("previous worker is still alive")
        source_files = ("code/train_deploy_methods.py", "code/run_deploy_methods.py",
            "code/package_deploy_methods.py", "code/method_training_core.py",
            "code/final_candidate_core.py", "submission_candidate_20260905/inference.py")
        manifest = dict(recipes=RECIPES, jobs=["metric35", "predbag2_35"],
            source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_files},
            scope="Two all-data fits, 4/5-member staging and native smoke; no pod lifecycle or challenge submission.")
        if (ROOT / "manifest.json").exists():
            assert json.loads((ROOT / "manifest.json").read_text()) == manifest
        else:
            save_json(ROOT / "manifest.json", manifest)
        pending = [{"recipe": recipe, "smoke": not valid(recipe, True)} for recipe in RECIPES]
        done, failed, active = [], {}, {}
        started = time.time()

        def snapshot(stage="training"):
            save_json(ROOT / "status.json", dict(updated_at=time.strftime("%FT%T%z"), stage=stage,
                elapsed_seconds=time.time() - started, done=done, failed=failed,
                pending=[f"{x['recipe']}:{'smoke' if x['smoke'] else 'fit'}" for x in pending],
                active={str(gpu): {"recipe": item["job"]["recipe"],
                    "stage": "smoke" if item["job"]["smoke"] else "fit", "pid": item["process"].pid,
                    "started": item["started"], "log": item["log"]} for gpu, item in active.items()}))

        snapshot()
        while pending or active:
            for gpu in range(2):
                if gpu in active or not pending or gpu_busy(gpu):
                    continue
                job = pending.pop(0)
                if valid(job["recipe"], job["smoke"]):
                    if job["smoke"]:
                        pending.append({"recipe": job["recipe"], "smoke": False})
                    else:
                        done.append(job["recipe"])
                    continue
                run_dir = result_path(job["recipe"], job["smoke"]).parent
                if run_dir.exists():
                    failed[job["recipe"]] = f"partial directory exists: {run_dir}"
                    continue
                stage = "smoke" if job["smoke"] else "fit"
                command = [sys.executable, "code/train_deploy_methods.py", "--recipe", job["recipe"]]
                if job["smoke"]:
                    command.append("--smoke")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",
                    PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
                log = LOG / f"{job['recipe']}_{stage}.log"
                stream = log.open("a")
                process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True)
                active[gpu] = dict(job=job, process=process, stream=stream,
                    started=time.time(), log=str(log), terminated=None)
                print(f"{time.strftime('%FT%T%z')} gpu={gpu} {job['recipe']} {stage} pid={process.pid}", flush=True)
                snapshot()
            time.sleep(10)
            for gpu, item in list(active.items()):
                process = item["process"]
                limit = 900 if item["job"]["smoke"] else 8 * 3600
                if process.poll() is None and time.time() - item["started"] > limit:
                    if item["terminated"] is None:
                        os.killpg(process.pid, signal.SIGTERM); item["terminated"] = time.time()
                    elif time.time() - item["terminated"] > 60:
                        os.killpg(process.pid, signal.SIGKILL)
                code = process.poll()
                if code is None:
                    continue
                item["stream"].close(); job = item["job"]
                try:
                    good = code == 0 and valid(job["recipe"], job["smoke"])
                except Exception as exc:
                    good = False; failed[job["recipe"]] = repr(exc)
                if good and job["smoke"]:
                    pending.append({"recipe": job["recipe"], "smoke": False})
                elif good:
                    done.append(job["recipe"])
                else:
                    failed.setdefault(job["recipe"], f"{('smoke' if job['smoke'] else 'fit')} exit {code}")
                print(f"finished {job['recipe']} smoke={job['smoke']} good={good}", flush=True)
                del active[gpu]
                snapshot()
            snapshot()
        if not failed:
            snapshot("postflight")
            with (LOG / "postflight.log").open("a") as stream:
                try:
                    subprocess.run([sys.executable, "code/package_deploy_methods.py"], check=True,
                        stdout=stream, stderr=subprocess.STDOUT, timeout=3600,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4"))
                except Exception as exc:
                    failed["postflight"] = repr(exc)
        snapshot("complete")
        save_json(ROOT / "complete.json", dict(done=done, failed=failed,
            completed_at=time.strftime("%FT%T%z"), elapsed_seconds=time.time() - started))
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
