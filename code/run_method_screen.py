#!/usr/bin/env python3
"""Finite 8-job screen -> at most 6 confirmation jobs, independently on the pod."""
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

from audit_method_screen import METHODS, ROOT, confirmation, load_fold, run_path, save, screen
from audit_method_screen import aligned
from audit_submission_ensembles import sources

TRAINER = "code/train_method_h200.py"


def make_job(method, fold, smoke=False):
    root = Path("experiments/methods_20260904_smoke") if smoke else ROOT
    env = dict(ROOT="datasets/clear_ec_official", OUT_ROOT=str(root / "runs"),
        CPSAM_MODEL="cpsam_models/cpsam_v2", TAG=method, FOLD=fold, NFOLDS=5,
        FOLD_SEED=2026, SEED=42, EPOCHS=1 if smoke else 35, BS=4 if smoke else 48,
        CROP=256, K_CROPS=2 if method == "predbag2_35" else 1, CROP_MODE="random",
        BAG_POOL="mean", POOL="mean", FEATURE_AGG=0, LOSS_MODE="multitask", YSRC="slidesmooth",
        OBJECTIVE="mixed_mape" if method == "metric35" else "log_l1",
        SAM_RHO=0.02 if method == "sam35" else 0, MAX_TRAIN_STEPS=1 if smoke else 0,
        MAX_VAL_IMAGES=8 if smoke else 0, EVAL_D8=0, FINAL_D8=1, EVAL_IMAGE_BS=4,
        ENC_LR=1e-5, HEAD_LR=1e-3, WEIGHT_DECAY=1e-4, SELECT_MODE="best",
        EXCLUDE_FOLD=-1, NORM_MODE="fixed", ENCODER_INIT="", SAVE_CHECKPOINT=1,
        PYTHONNOUSERSITE=1, PYTHONUNBUFFERED=1, OMP_NUM_THREADS=4,
        OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=4)
    return dict(key=f"{method}_f{fold}", method=method, fold=fold, env=env,
                run=str(run_path(method, fold, root)))


def valid(job, smoke):
    run = Path(job["run"])
    if not (run / "result.json").exists():
        return False
    data = load_fold(run)
    assert len(data["indices"]) == 8 if smoke else len(data["indices"]) > 1700
    config = json.loads((run / "result.json").read_text())
    assert config["epochs"] == job["env"]["EPOCHS"]
    assert config["objective"] == job["env"]["OBJECTIVE"]
    assert config["sam_rho"] == job["env"]["SAM_RHO"]
    return True


def gpu_busy(gpu):
    result = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return bool(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    jobs = [make_job(m, f, args.smoke) for m in METHODS for f in ((0,) if args.smoke else (0, 1))]
    if args.dry_run:
        print(json.dumps(jobs, indent=2)); return
    root = Path("experiments/methods_20260904_smoke") if args.smoke else ROOT
    logs = Path("logs") / root.name
    root.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "complete.json").exists():
            raise RuntimeError("finite queue already complete; do not restart")
        if (root / "status.json").exists():
            old = json.loads((root / "status.json").read_text())
            for value in old.get("active", {}).values():
                proc = Path(f"/proc/{value['pid']}/cmdline")
                if proc.exists() and TRAINER.encode() in proc.read_bytes():
                    raise RuntimeError("orphan training process is alive; inspect before restart")
        source_paths = [TRAINER, "code/method_training_core.py", "code/audit_method_screen.py",
            "code/audit_submission_ensembles.py", "code/train_oof_h200.py", __file__]
        baseline_hashes = {}
        if not args.smoke:
            historical = sources()
            for fold in range(5):
                reference = None
                for name in ("fixed_base", "base_seed77", "base_seed2026"):
                    path = historical[name][fold]
                    data = load_fold(path)
                    if reference is not None:
                        aligned(reference, data)
                    reference = data
                    for filename in ("best_oof.npz", "split_indices.npz", "result.json"):
                        file = path / filename
                        baseline_hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        manifest = dict(screen_jobs=jobs, max_total_jobs=4 if args.smoke else 14,
            source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths},
            baseline_sha256=baseline_hashes,
            scope="No final-candidate changes, all-data initialization, challenge upload, or pod lifecycle operations.")
        if (root / "manifest.json").exists():
            assert json.loads((root / "manifest.json").read_text()) == manifest, "source/config changed"
        else:
            save(root / "manifest.json", manifest)
        done = [j["key"] for j in jobs if valid(j, args.smoke)]
        pending = [j for j in jobs if j["key"] not in done]
        active, failed = {}, {}
        phase = "smoke" if args.smoke else "screen"
        promoted = []
        started = time.time()

        def status():
            save(root / "status.json", dict(updated_at=time.strftime('%FT%T%z'),
                started_at=started, elapsed_seconds=time.time() - started, phase=phase,
                done=done, failed=failed, promoted=promoted,
                pending=[j["key"] for j in pending],
                active={str(g): dict(key=v["job"]["key"], pid=v["process"].pid,
                    started=v["started"], log=v["log"]) for g, v in active.items()}))

        status()
        while True:
            for gpu in range(8):
                if gpu in active or not pending or gpu_busy(gpu):
                    continue
                job = pending.pop(0)
                if valid(job, args.smoke):
                    done.append(job["key"]); continue
                # Do not silently overwrite a partial checkpoint on controller restart.
                if Path(job["run"]).exists():
                    failed[job["key"]] = "partial run exists; manual inspection required"
                    continue
                env = os.environ.copy()
                env.update({k: str(v) for k, v in job["env"].items()})
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                log = logs / f"{job['key']}.log"
                stream = log.open("a")
                proc = subprocess.Popen([sys.executable, TRAINER], env=env,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = dict(job=job, process=proc, stream=stream, log=str(log),
                    started=time.time(), term_at=None)
                print(f"{time.strftime('%FT%T%z')} gpu={gpu} start {job['key']} pid={proc.pid}", flush=True)
                status()
            time.sleep(10)
            for gpu, value in list(active.items()):
                proc = value["process"]
                if proc.poll() is None and time.time() - value["started"] > (900 if args.smoke else 10 * 3600):
                    if value["term_at"] is None:
                        os.killpg(proc.pid, signal.SIGTERM); value["term_at"] = time.time()
                    elif time.time() - value["term_at"] > 60:
                        os.killpg(proc.pid, signal.SIGKILL)
                rc = proc.poll()
                if rc is None:
                    continue
                value["stream"].close(); job = value["job"]
                try:
                    good = rc == 0 and valid(job, args.smoke)
                except Exception as exc:
                    good = False; failed[job["key"]] = repr(exc)
                if good:
                    done.append(job["key"])
                else:
                    failed.setdefault(job["key"], f"exit {rc}; inspect log, no automatic retry")
                print(f"finished {job['key']} rc={rc} valid={good}", flush=True)
                del active[gpu]
            status()
            if pending or active:
                continue
            if phase == "screen":
                failed_methods = [m for m in METHODS if any(f"{m}_f{f}" not in done for f in (0, 1))]
                # Freeze a screen decision before starting confirmation jobs.
                report = screen(failed_methods)
                promoted = report["promoted"]
                phase = "confirmation"
                pending = [make_job(m, f) for m in promoted for f in (2, 3, 4)]
                jobs += pending
                save(root / "confirmation_plan.json", dict(promoted=promoted, jobs=pending))
                print(f"screen complete; promoted={promoted}", flush=True)
                status()
                if pending:
                    continue
            if not args.smoke:
                confirmation(promoted)
            phase = "complete"
            status()
            save(root / "complete.json", dict(completed_at=time.strftime('%FT%T%z'),
                done=done, failed=failed, promoted=promoted, elapsed_seconds=time.time()-started))
            print(f"complete: {len(done)} jobs; failures={failed}", flush=True)
            if failed:
                raise SystemExit(1)
            return


if __name__ == "__main__":
    main()
