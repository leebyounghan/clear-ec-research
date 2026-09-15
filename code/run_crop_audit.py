#!/usr/bin/env python3
"""Run the crop audit immediately after the current curriculum controller exits."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


PYTHON = os.environ.get("PYTHON_BIN", "/workspace/venv/bin/python")
WAIT_PID = int(os.environ.get("WAIT_PID", "0"))
LOGS = Path("logs/crop_audit")
LOGS.mkdir(parents=True, exist_ok=True)


def alive(pid: int) -> bool:
    if not pid:
        return False
    # kill(pid, 0) also succeeds for an unreaped zombie.  Treat zombies as
    # completed so a chained GPU workload cannot wait forever on a dead parent.
    status = Path(f"/proc/{pid}/stat")
    if status.exists():
        try:
            if status.read_text().split()[2] == "Z":
                return False
        except (OSError, IndexError):
            pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def base_env(**values: object) -> dict[str, str]:
    env = os.environ.copy()
    env.update({key: str(value) for key, value in values.items()})
    env.setdefault("OMP_NUM_THREADS", "4")
    return env


while alive(WAIT_PID):
    print(f"[{time.strftime('%FT%T%z')}] waiting for curriculum pid={WAIT_PID}", flush=True)
    time.sleep(30)

print(f"[{time.strftime('%FT%T%z')}] starting crop audit", flush=True)
jobs: list[tuple[list[str], dict[str, str], Path]] = []
for fold in range(5):
    run = f"experiments/oof_h200/baseline_f{fold}_seed42_c256k1_mean_multitask_slidesmooth"
    jobs.append((
        [PYTHON, "code/extract_oof_crop_predictions.py"],
        base_env(RUN_DIR=run, D8=1, EVAL_IMAGE_BS=4),
        LOGS / f"extract_baseline_f{fold}.log",
    ))

encoder = "experiments/domain_jepa_small/encoder_latest.pt"
variants = [
    ("cover4_mean", 4, "cover4", "mean"),
    ("cover4_meanstd", 4, "cover4", "meanstd"),
    ("cover4_attn", 4, "cover4", "attn"),
    ("cover5_mean", 5, "cover5", "mean"),
    ("cover5_meanstd", 5, "cover5", "meanstd"),
    ("cover5_attn", 5, "cover5", "attn"),
    ("random4_mean", 4, "random", "mean"),
    ("random4_attn", 4, "random", "attn"),
]
for tag, count, crop_mode, bag_pool in variants:
    jobs.append((
        [PYTHON, "code/train_oof_h200.py"],
        base_env(
            ROOT="datasets/clear_ec_official",
            OUT_ROOT="experiments/crop_h200",
            CPSAM_MODEL="cpsam_models/cpsam_v2",
            ENCODER_INIT=encoder,
            TAG=tag,
            FOLD=0,
            NFOLDS=5,
            FOLD_SEED=2026,
            SEED=42,
            YSRC="slidesmooth",
            EVAL_D8=0,
            FINAL_D8=1,
            EPOCHS=15,
            CROP=256,
            K_CROPS=count,
            CROP_MODE=crop_mode,
            BAG_POOL=bag_pool,
            FEATURE_AGG=1,
            EVAL_IMAGE_BS=4,
            POOL="mean",
            LOSS_MODE="multitask",
            BS=24 if count == 4 else 20,
            MAX_TRAIN_STEPS=120,
            ENC_LR="1e-5",
            HEAD_LR="1e-3",
        ),
        LOGS / f"train_{tag}.log",
    ))

pending = list(jobs)
active: dict[int, tuple[subprocess.Popen, object, Path]] = {}
failed: list[str] = []
while pending or active:
    for gpu in range(8):
        if gpu in active or not pending:
            continue
        command, env, log = pending.pop(0)
        stream = log.open("w")
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        active[gpu] = (process, stream, log)
        print(f"[{time.strftime('%FT%T%z')}] gpu{gpu} launched {log}", flush=True)
    time.sleep(10)
    for gpu, (process, stream, log) in list(active.items()):
        returncode = process.poll()
        if returncode is None:
            continue
        stream.close()
        if returncode:
            failed.append(f"{log}: exit {returncode}")
        print(f"[{time.strftime('%FT%T%z')}] gpu{gpu} finished {log} rc={returncode}", flush=True)
        del active[gpu]

if all((Path(f"experiments/oof_h200/baseline_f{fold}_seed42_c256k1_mean_multitask_slidesmooth") / "crop_tiles.npz").exists() for fold in range(5)):
    with (LOGS / "summary.log").open("w") as stream:
        result = subprocess.run([PYTHON, "code/summarize_crop_oof.py"], stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        failed.append(f"{LOGS / 'summary.log'}: exit {result.returncode}")

if failed:
    print("FAILED\n" + "\n".join(failed), flush=True)
    raise SystemExit(1)
print(f"[{time.strftime('%FT%T%z')}] crop audit complete", flush=True)
