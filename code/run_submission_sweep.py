#!/usr/bin/env python3
"""Finite 20-job seed/pooling/target sweep; never manages pods or submits models."""
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

ROOT = Path("experiments/submission_20260904")
LOGS = Path("logs/submission_20260904")


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def make_job(group, fold):
    seed = {"base_seed77": 77, "base_seed2026": 2026}.get(group, 42)
    pool = "attn" if group == "attention" else "mean"
    ysrc = "raw" if group == "raw_targets" else "slidesmooth"
    env = dict(ROOT="datasets/clear_ec_official", OUT_ROOT=str(ROOT / "runs"),
        CPSAM_MODEL="cpsam_models/cpsam_v2", TAG=group, FOLD=fold,
        NFOLDS=5, FOLD_SEED=2026, SEED=seed, YSRC=ysrc,
        EPOCHS=25, BS=48, CROP=256, K_CROPS=1, CROP_MODE="random",
        BAG_POOL="mean", POOL=pool, FEATURE_AGG=0, LOSS_MODE="multitask",
        MAX_TRAIN_STEPS=0, MAX_VAL_IMAGES=0, EVAL_D8=0, FINAL_D8=1,
        EVAL_IMAGE_BS=4, ENC_LR=1e-5, HEAD_LR=1e-3, WEIGHT_DECAY=1e-4,
        SELECT_MODE="best", EXCLUDE_FOLD=-1, NORM_MODE="fixed",
        ENCODER_INIT="", SAVE_CHECKPOINT=1, PYTHONNOUSERSITE=1)
    run = ROOT / "runs" / f"{group}_f{fold}_seed{seed}_c256k1_{pool}_multitask_{ysrc}"
    return dict(key=f"{group}_f{fold}", group=group, fold=fold, env=env, run=str(run))


def build_jobs():
    order = [("base_seed77", f) for f in range(5)]
    order += [("attention", 0), ("attention", 1), ("raw_targets", 0)]
    order += [("raw_targets", f) for f in range(1, 5)]
    order += [("attention", f) for f in range(2, 5)]
    order += [("base_seed2026", f) for f in range(5)]
    return [make_job(g, f) for g, f in order]


def valid_result(job):
    run = Path(job["run"])
    if not (run / "result.json").exists():
        return False
    value = json.loads((run / "result.json").read_text())
    if not value.get("final_metrics") or not np.isfinite(value["final_metrics"]).all():
        raise ValueError(f"invalid completed result: {run}")
    with np.load(run / "best_oof.npz", allow_pickle=True) as data, np.load(run / "split_indices.npz") as split:
        assert np.array_equal(data["indices"], split["validation"])
        assert not np.intersect1d(data["indices"], split["train"]).size
        assert len(data["indices"]) > 1700 and np.isfinite(data["pred"]).all()
    return True


def gpu_busy(gpu):
    result = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return bool(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    jobs = build_jobs()
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    ROOT.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    with (ROOT / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / "status.json").exists():
            old = json.loads((ROOT / "status.json").read_text())
            for value in old.get("active", {}).values():
                proc = Path(f"/proc/{value['pid']}/cmdline")
                if proc.exists() and b"train_oof_h200.py" in proc.read_bytes():
                    raise RuntimeError("Previous training process still alive; inspect/adopt before restart")
        scripts = ["code/train_oof_h200.py", "code/audit_submission_ensembles.py", __file__]
        manifest = dict(jobs=jobs, source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in scripts},
            scope="20 finite central512 training jobs. No ROI GT, no submission, no pod lifecycle changes.")
        if (ROOT / "manifest.json").exists():
            assert json.loads((ROOT / "manifest.json").read_text()) == manifest, "manifest/source changed"
        else:
            save(ROOT / "manifest.json", manifest)
        done = [j["key"] for j in jobs if valid_result(j)]
        pending = [j for j in jobs if j["key"] not in done]
        active, failed, analyzed = {}, {}, set()
        started = time.time()

        def audit():
            complete_groups = tuple(sorted(g for g in {j["group"] for j in jobs}
                if all(j["key"] in done for j in jobs if j["group"] == g)))
            if complete_groups in analyzed:
                return
            with (LOGS / "ensemble_audit.log").open("a") as stream:
                try:
                    r = subprocess.run([sys.executable, "code/audit_submission_ensembles.py"],
                        stdout=stream, stderr=subprocess.STDOUT, timeout=300,
                        env=dict(os.environ, OMP_NUM_THREADS="2"))
                    if r.returncode: failed["analysis:" + repr(complete_groups)] = f"exit {r.returncode}"
                except subprocess.TimeoutExpired:
                    failed["analysis:" + repr(complete_groups)] = "timeout"
            analyzed.add(complete_groups)

        def status():
            save(ROOT / "status.json", dict(updated_at=time.strftime('%FT%T%z'),
                started_at=started, elapsed_seconds=time.time() - started,
                done=done, failed=failed, pending=[j["key"] for j in pending],
                active={str(g): dict(key=v["job"]["key"], pid=v["process"].pid,
                    started=v["started"], log=v["log"]) for g, v in active.items()}))

        audit()
        while pending or active:
            for gpu in range(8):
                if gpu in active or not pending or gpu_busy(gpu):
                    continue
                job = pending.pop(0)
                if valid_result(job):
                    done.append(job["key"]); continue
                env = os.environ.copy()
                for name in ("WAIT_PID", "MAX_IMAGES", "EXTRACT_SPLIT"):
                    env.pop(name, None)
                env.update({k: str(v) for k, v in job["env"].items()})
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4")
                log = LOGS / f"{job['key']}.log"
                stream = log.open("a")
                proc = subprocess.Popen([sys.executable, "code/train_oof_h200.py"],
                    env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = dict(job=job, process=proc, stream=stream, log=str(log), started=time.time(), term_at=None)
                print(f"{time.strftime('%FT%T%z')} gpu={gpu} start {job['key']} pid={proc.pid}", flush=True)
                status()
            time.sleep(10)
            for gpu, value in list(active.items()):
                proc = value["process"]
                if proc.poll() is None and time.time() - value["started"] > 12 * 3600:
                    if value["term_at"] is None:
                        os.killpg(proc.pid, signal.SIGTERM); value["term_at"] = time.time()
                    elif time.time() - value["term_at"] > 60:
                        os.killpg(proc.pid, signal.SIGKILL)
                rc = proc.poll()
                if rc is None: continue
                value["stream"].close(); job = value["job"]
                try:
                    good = rc == 0 and valid_result(job)
                except Exception as exc:
                    good = False; failed[job["key"]] = repr(exc)
                if good:
                    done.append(job["key"])
                else:
                    failed.setdefault(job["key"], f"exit {rc}; inspect log")
                print(f"finished {job['key']} rc={rc} valid={good}", flush=True)
                del active[gpu]
            status(); audit()
        audit(); status()
        save(ROOT / "complete.json", dict(completed_at=time.strftime('%FT%T%z'), done=done, failed=failed))
        print(f"complete: {len(done)} jobs; failures={failed}", flush=True)
        if failed: raise SystemExit(1)


if __name__ == "__main__":
    main()
