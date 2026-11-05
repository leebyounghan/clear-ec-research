#!/usr/bin/env python3
"""Bounded, resumable 8-GPU validation queue. Never manages pod lifecycle."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("experiments/followup_20260902")
LOGS = Path("logs/followup_20260902")
PYTHON = sys.executable
ROOT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
lock = (ROOT / "controller.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def training(key, tag, fold, encoder="", **extra):
    values = dict(ROOT="datasets/clear_ec_official", OUT_ROOT=str(ROOT / "runs"),
                  CPSAM_MODEL="cpsam_models/cpsam_v2", TAG=tag, FOLD=fold,
                  NFOLDS=5, FOLD_SEED=2026, SEED=42, YSRC="slidesmooth",
                  EPOCHS=25, BS=48, CROP=256, K_CROPS=1, CROP_MODE="random",
                  BAG_POOL="mean", POOL="mean", FEATURE_AGG=0, LOSS_MODE="multitask",
                  MAX_TRAIN_STEPS=0, MAX_VAL_IMAGES=0, EVAL_D8=0, FINAL_D8=1,
                  EVAL_IMAGE_BS=4, ENC_LR=1e-5, HEAD_LR=1e-3,
                  SELECT_MODE="best", EXCLUDE_FOLD=-1, NORM_MODE="fixed",
                  ENCODER_INIT=encoder, SAVE_CHECKPOINT=1)
    values.update(extra)
    suffix = "" if values["CROP_MODE"] == "random" and values["BAG_POOL"] == "mean" else f"_{values['CROP_MODE']}_{values['BAG_POOL']}"
    name = f"{tag}_f{fold}_seed42_c256k{values['K_CROPS']}_mean_multitask_slidesmooth{suffix}"
    run = ROOT / "runs" / name
    stages = [dict(script="code/train_oof_h200.py", env=values, output=str(run / "result.json"))]
    if values["K_CROPS"] == 1:
        stages.append(dict(script="code/extract_oof_crop_predictions.py",
                           env=dict(RUN_DIR=str(run), D8=1, EVAL_IMAGE_BS=4), output=str(run / "crop_tiles.npz")))
        if values["EXCLUDE_FOLD"] >= 0:
            stages.append(dict(script="code/extract_oof_crop_predictions.py",
                               env=dict(RUN_DIR=str(run), D8=1, EVAL_IMAGE_BS=4, EXTRACT_SPLIT="holdout"),
                               output=str(run / "crop_holdout.npz")))
    return dict(key=key, run=str(run), stages=stages)


jobs = []
for fold in range(5):
    run = f"experiments/oof_h200/jepa_small_mean_f{fold}_seed42_c256k1_mean_multitask_slidesmooth"
    jobs.append(dict(key=f"extract_jepa_f{fold}", run=run, stages=[dict(
        script="code/extract_oof_crop_predictions.py", env=dict(RUN_DIR=run, D8=1, EVAL_IMAGE_BS=4),
        output=f"{run}/crop_tiles.npz")]))
jepa = "experiments/domain_jepa_small/encoder_latest.pt"
mil_settings = dict(K_CROPS=4, CROP_MODE="cover4", BAG_POOL="meanstd", FEATURE_AGG=1, BS=24)
jobs.extend([
    training("fixed_base_f0", "fixed_base", 0),
    training("fixed_jepa_f0", "fixed_jepa", 0, jepa),
    training("mil_f0", "fixed_mil", 0, jepa, **mil_settings),
])
# Train/calibration/test are distinct slide groups. Checkpoint epoch is fixed,
# and target normalization is fitted only on the training portion.
for outer in range(5):
    jobs.append(training(f"nested_o{outer}", f"nested_o{outer}", (outer + 1) % 5,
                         EXCLUDE_FOLD=outer, SELECT_MODE="last", NORM_MODE="train"))
jobs.extend([
    training("matched_k1_f0", "matched_k1", 0, jepa, BS=24),
    training("mil_f1", "fixed_mil", 1, jepa, **mil_settings),
    training("matched_k1_f1", "matched_k1", 1, jepa, BS=24),
])
for fold in range(1, 5):
    jobs.extend([training(f"fixed_base_f{fold}", "fixed_base", fold),
                 training(f"fixed_jepa_f{fold}", "fixed_jepa", fold, jepa)])

scripts = sorted({s["script"] for job in jobs for s in job["stages"]})
save(ROOT / "manifest.json", dict(jobs=jobs, source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in scripts},
                                  note="Finite validation queue; does not submit algorithms or stop/delete pods."))


def valid_output(path):
    path = Path(path)
    if not path.exists():
        return False
    try:
        if path.suffix == ".json":
            import math
            value = json.loads(path.read_text())
            return bool(value.get("final_metrics")) and all(math.isfinite(x) for x in value["final_metrics"])
        import numpy as np
        with np.load(path, allow_pickle=True) as data:
            return len(data["indices"]) > 0 and np.isfinite(data["z_tile"]).all() and np.isfinite(data["metric_mean"]).all()
    except Exception:
        return False


active, done, failed, pending = {}, [], [], list(jobs)
stages = {job["key"]: 0 for job in jobs}
started = time.time()
analysis_done = set()


def ready_analyses():
    actions = [
        ("jepa_crop_summary", [f"extract_jepa_f{i}" for i in range(5)], "code/summarize_crop_oof.py", dict(TAG="jepa_small_mean")),
        ("nested_report", [f"nested_o{i}" for i in range(5)], "code/validate_crop_holdouts.py", {}),
        ("fixed_base_summary", [f"fixed_base_f{i}" for i in range(5)], "code/summarize_crop_oof.py", dict(TAG="fixed_base", OOF_ROOT=str(ROOT / "runs"))),
        ("fixed_jepa_summary", [f"fixed_jepa_f{i}" for i in range(5)], "code/summarize_crop_oof.py", dict(TAG="fixed_jepa", OOF_ROOT=str(ROOT / "runs"))),
    ]
    for key, dependencies, script, overrides in actions:
        if key in analysis_done or not set(dependencies).issubset(done):
            continue
        env = os.environ.copy(); env.update(overrides); env["OMP_NUM_THREADS"] = "2"
        with (LOGS / f"{key}.log").open("w") as stream:
            result = subprocess.run([PYTHON, script], env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
        analysis_done.add(key)
        if result.returncode:
            failed.append(f"analysis:{key}")
        print(f"analysis {key} rc={result.returncode}", flush=True)


while pending or active:
    for gpu in range(8):
        if gpu in active or not pending:
            continue
        job = pending.pop(0)
        key = job["key"]
        while stages[key] < len(job["stages"]) and valid_output(job["stages"][stages[key]]["output"]):
            stages[key] += 1
        if stages[key] == len(job["stages"]):
            done.append(key)
            continue
        stage = job["stages"][stages[key]]
        env = os.environ.copy()
        for name in ("WAIT_PID", "MAX_IMAGES", "EXTRACT_SPLIT", "ENCODER_INIT", "EXCLUDE_FOLD", "SELECT_MODE", "NORM_MODE"):
            env.pop(name, None)
        env.update({k: str(v) for k, v in stage["env"].items()})
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4")
        log = LOGS / f"{key}_s{stages[key]}.log"
        stream = log.open("a")
        process = subprocess.Popen([PYTHON, stage["script"]], env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        active[gpu] = dict(job=job, process=process, stream=stream, log=str(log), started=time.time())
        print(f"{time.strftime('%FT%T%z')} gpu={gpu} start {key} stage={stages[key]} pid={process.pid}", flush=True)
    time.sleep(10)
    for gpu, value in list(active.items()):
        process = value["process"]
        if time.time() - value["started"] > 12 * 3600 and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        code = process.poll()
        if code is None:
            continue
        value["stream"].close()
        job = value["job"]; key = job["key"]
        output = job["stages"][stages[key]]["output"]
        if code or not valid_output(output):
            failed.append(key)
            print(f"FAILED {key} rc={code} output={output}", flush=True)
        else:
            stages[key] += 1
            if stages[key] == len(job["stages"]):
                done.append(key)
            else:
                pending.insert(0, job)
        del active[gpu]
    ready_analyses()
    save(ROOT / "status.json", dict(updated_at=time.strftime('%FT%T%z'), elapsed_seconds=time.time()-started,
        done=done, failed=failed, pending=[j["key"] for j in pending],
        active={str(g): dict(key=v["job"]["key"], pid=v["process"].pid, log=v["log"],
                            started=v["started"], stage=stages[v["job"]["key"]]) for g,v in active.items()}))

ready_analyses()
save(ROOT / "complete.json", dict(completed_at=time.strftime('%FT%T%z'), done=done, failed=failed))
print(f"queue complete: {len(done)} succeeded, {len(failed)} failed", flush=True)
raise SystemExit(bool(failed))
