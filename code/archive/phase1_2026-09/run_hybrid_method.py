#!/usr/bin/env python3
"""Finite five-fold hybrid queue pinned to GPUs2-6; no pod or submission actions."""
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

from audit_hybrid_method import ROOT, TAG, run_path
from audit_method_screen import save
from run_submission_sweep import gpu_busy


LOG = Path("logs/hybrid_20260905")


def job(fold):
    env = dict(ROOT="datasets/clear_ec_official", OUT_ROOT=str(ROOT / "runs"),
        CPSAM_MODEL="cpsam_models/cpsam_v2", TAG=TAG, FOLD=fold, NFOLDS=5,
        FOLD_SEED=2026, SEED=42, EPOCHS=35, BS=48, CROP=256, K_CROPS=2,
        CROP_MODE="random", BAG_POOL="mean", POOL="mean", FEATURE_AGG=0,
        LOSS_MODE="multitask", YSRC="slidesmooth", OBJECTIVE="mixed_mape", SAM_RHO=0,
        MAX_TRAIN_STEPS=0, MAX_VAL_IMAGES=0, EVAL_D8=0, FINAL_D8=1,
        EVAL_IMAGE_BS=4, ENC_LR=1e-5, HEAD_LR=1e-3, WEIGHT_DECAY=1e-4,
        SELECT_MODE="best", EXCLUDE_FOLD=-1, NORM_MODE="fixed", ENCODER_INIT="",
        SAVE_CHECKPOINT=1, PYTHONNOUSERSITE=1, PYTHONUNBUFFERED=1, OMP_NUM_THREADS=4)
    return {"fold": fold, "key": f"{TAG}_f{fold}", "env": env, "run": str(run_path(fold))}


def valid(item):
    run = Path(item["run"])
    if not (run / "result.json").exists():
        return False
    result = json.loads((run / "result.json").read_text())
    assert result["objective"] == "mixed_mape" and result["prediction_bag"] == 2
    with np.load(run / "best_oof.npz") as data, np.load(run / "split_indices.npz") as split:
        assert np.array_equal(data["indices"], split["validation"])
        assert not np.intersect1d(data["indices"], split["train"]).size
        assert len(data["indices"]) > 1700 and np.isfinite(data["pred"]).all()
    return True


def main():
    ROOT.mkdir(parents=True, exist_ok=True); LOG.mkdir(parents=True, exist_ok=True)
    jobs = [job(fold) for fold in range(5)]
    with (ROOT / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / "complete.json").exists():
            print("already complete", flush=True); return
        if (ROOT / "status.json").exists():
            old = json.loads((ROOT / "status.json").read_text())
            for value in old.get("active", {}).values():
                process = Path(f"/proc/{value['pid']}/cmdline")
                if process.exists() and b"train_method_h200.py" in process.read_bytes():
                    raise RuntimeError("previous hybrid worker is alive")
        files = ("code/train_method_h200.py", "code/method_training_core.py",
                 "code/run_hybrid_method.py", "code/audit_hybrid_method.py")
        manifest = dict(jobs=jobs, source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
            scope="Five hybrid OOF jobs on GPUs2-6. No all-data fit, pod lifecycle, upload or submission.")
        if (ROOT / "manifest.json").exists():
            assert json.loads((ROOT / "manifest.json").read_text()) == manifest
        else:
            save(ROOT / "manifest.json", manifest)
        done = [x["key"] for x in jobs if valid(x)]
        pending = [x for x in jobs if x["key"] not in done]
        active, failed = {}, {}
        started = time.time()

        def snapshot(stage="training"):
            save(ROOT / "status.json", dict(updated_at=time.strftime("%FT%T%z"), stage=stage,
                elapsed_seconds=time.time() - started, done=done, failed=failed,
                pending=[x["key"] for x in pending], active={str(gpu): dict(key=v["job"]["key"],
                    pid=v["process"].pid, started=v["started"], log=v["log"]) for gpu,v in active.items()}))

        snapshot()
        while pending or active:
            for gpu in range(2, 7):
                if gpu in active or not pending or gpu_busy(gpu):
                    continue
                item = pending.pop(0)
                if Path(item["run"]).exists():
                    failed[item["key"]] = "partial run exists; inspect before retry"; continue
                env = os.environ.copy(); env.update({k: str(v) for k,v in item["env"].items()})
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                log = LOG / f"{item['key']}.log"; stream = log.open("a")
                process = subprocess.Popen([sys.executable, "code/train_method_h200.py"], env=env,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = dict(job=item, process=process, stream=stream, started=time.time(),
                                   log=str(log), terminated=None)
                print(f"{time.strftime('%FT%T%z')} gpu={gpu} start {item['key']} pid={process.pid}", flush=True)
                snapshot()
            time.sleep(10)
            for gpu, value in list(active.items()):
                process = value["process"]
                if process.poll() is None and time.time() - value["started"] > 10 * 3600:
                    if value["terminated"] is None:
                        os.killpg(process.pid, signal.SIGTERM); value["terminated"] = time.time()
                    elif time.time() - value["terminated"] > 60:
                        os.killpg(process.pid, signal.SIGKILL)
                code = process.poll()
                if code is None: continue
                value["stream"].close(); item = value["job"]
                try: good = code == 0 and valid(item)
                except Exception as exc: good = False; failed[item["key"]] = repr(exc)
                if good: done.append(item["key"])
                else: failed.setdefault(item["key"], f"exit {code}")
                del active[gpu]; snapshot()
            snapshot()
        if not failed:
            snapshot("audit")
            result = subprocess.run([sys.executable, "code/audit_hybrid_method.py"], timeout=600)
            if result.returncode: failed["audit"] = f"exit {result.returncode}"
        snapshot("complete")
        save(ROOT / "complete.json", dict(done=done, failed=failed,
            completed_at=time.strftime("%FT%T%z"), elapsed_seconds=time.time() - started))
        if failed: raise SystemExit(1)


if __name__ == "__main__":
    main()
