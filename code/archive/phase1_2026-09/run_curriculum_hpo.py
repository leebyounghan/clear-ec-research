#!/usr/bin/env python3
"""Saturate 8 H200s with external->CLEAR JEPA curriculum and supervised HPO.

Stages:
1. Eight independent representation curricula (one per GPU).
2. Common fold-0 supervised screen for an apples-to-apples encoder comparison.
3. Eight dense encoder/head LR combinations over the two best representations.
4. Full 5-fold confirmation of best-overall, best-CD, and runner-up candidates.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path("datasets/clear_ec_official")
OUT = Path("experiments/curriculum_hpo")
OOF = Path("experiments/oof_h200")
LOGS = Path("logs/curriculum_hpo")
PYTHON = os.environ.get("PYTHON_BIN", "/workspace/venv/bin/python")
SMALL = "srep_topcon_sp3000,rodrep_confocal_endothelium"
ALL_SMALL = f"{SMALL},github_cornea_cells"


VARIANTS: list[dict[str, Any]] = [
    dict(key="srep_only", sources="srep_topcon_sp3000", ext_mask=.55, ext_lr=2e-5,
         clear_mask=.55, clear_lr=1e-5, clear_epochs=18),
    dict(key="srep_rodrep", sources=SMALL, ext_mask=.55, ext_lr=2e-5,
         clear_mask=.55, clear_lr=1e-5, clear_epochs=18),
    dict(key="all_small", sources=ALL_SMALL, ext_mask=.55, ext_lr=2e-5,
         clear_mask=.55, clear_lr=1e-5, clear_epochs=18),
    dict(key="mask45", sources=SMALL, ext_mask=.45, ext_lr=2e-5,
         clear_mask=.45, clear_lr=1e-5, clear_epochs=18),
    dict(key="mask65", sources=SMALL, ext_mask=.65, ext_lr=2e-5,
         clear_mask=.65, clear_lr=1e-5, clear_epochs=18),
    dict(key="low_lr", sources=SMALL, ext_mask=.55, ext_lr=1e-5,
         clear_mask=.55, clear_lr=5e-6, clear_epochs=18),
    dict(key="high_lr", sources=SMALL, ext_mask=.55, ext_lr=3e-5,
         clear_mask=.55, clear_lr=2e-5, clear_epochs=18),
    dict(key="clear_only", sources="", ext_mask=.55, ext_lr=0.0,
         clear_mask=.55, clear_lr=1e-5, clear_epochs=18),
]


@dataclass
class Candidate:
    tag: str
    encoder: str
    enc_lr: float = 1e-5
    head_lr: float = 1e-3
    seed: int = 42
    source: str = ""
    metrics: list[float] | None = None

    @property
    def overall(self) -> float:
        assert self.metrics is not None
        return sum(self.metrics) / 3

    @property
    def cd(self) -> float:
        assert self.metrics is not None
        return self.metrics[0]


def env_with(**values: Any) -> dict[str, str]:
    env = os.environ.copy()
    env.update({key: str(value) for key, value in values.items() if value != ""})
    env.setdefault("OMP_NUM_THREADS", "4")
    return env


def run_checked(command: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as stream:
        process = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"command failed ({process.returncode}), see {log_path}: {' '.join(command)}")


def curriculum_worker(index: int) -> None:
    variant = VARIANTS[index]
    key = variant["key"]
    base = OUT / key
    base.mkdir(parents=True, exist_ok=True)
    initial_encoder = ""
    if variant["sources"]:
        external_out = base / "external"
        run_checked(
            [PYTHON, "code/train_domain_jepa.py"],
            env_with(
                ROOT=ROOT,
                OUT=external_out,
                EXTERNAL_MANIFEST="datasets/external/ssl_manifest.jsonl",
                EXTERNAL_SOURCES=variant["sources"],
                EXTERNAL_FRACTION=1.0,
                EPOCHS=24,
                STEPS_PER_EPOCH=25,
                BS=48,
                LR=variant["ext_lr"],
                MASK_RATIO=variant["ext_mask"],
                WARMUP_STEPS=50,
                SAVE_EVERY=9999,
                SAVE_FULL_STATE=0,
                SEED=2026 + index,
            ),
            LOGS / f"{key}_external.log",
        )
        initial_encoder = str(external_out / "encoder_latest.pt")

    clear_out = base / "clear"
    settings = dict(
        ROOT=ROOT,
        OUT=clear_out,
        EXTERNAL_MANIFEST="datasets/external/ssl_manifest.jsonl",
        EXTERNAL_FRACTION=0.0,
        EPOCHS=variant["clear_epochs"],
        BS=48,
        LR=variant["clear_lr"],
        MASK_RATIO=variant["clear_mask"],
        WARMUP_STEPS=100,
        SAVE_EVERY=9999,
        SAVE_FULL_STATE=0,
        SEED=3026 + index,
    )
    if initial_encoder:
        settings["INITIAL_ENCODER"] = initial_encoder
    run_checked(
        [PYTHON, "code/train_domain_jepa.py"],
        env_with(**settings),
        LOGS / f"{key}_clear.log",
    )
    (base / "complete.json").write_text(
        json.dumps({"variant": variant, "encoder": str(clear_out / "encoder_latest.pt")}, indent=2)
    )


def launch_parallel(jobs: list[tuple[int, list[str], dict[str, str], Path]]) -> None:
    processes: list[tuple[subprocess.Popen, Path]] = []
    for gpu, command, env, log_path in jobs:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("w")
        child_env = env.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        process = subprocess.Popen(command, env=child_env, stdout=stream, stderr=subprocess.STDOUT)
        process._clear_ec_stream = stream  # type: ignore[attr-defined]
        processes.append((process, log_path))
    failed: list[str] = []
    for process, log_path in processes:
        returncode = process.wait()
        process._clear_ec_stream.close()  # type: ignore[attr-defined]
        if returncode:
            failed.append(f"{log_path} (exit {returncode})")
    if failed:
        raise RuntimeError("parallel jobs failed: " + ", ".join(failed))


def launch_queue(jobs: list[tuple[list[str], dict[str, str], Path]], gpu_count: int = 8) -> None:
    pending = list(jobs)
    active: dict[int, tuple[subprocess.Popen, Path, Any]] = {}
    failed: list[str] = []
    while pending or active:
        for gpu in range(gpu_count):
            if gpu in active or not pending:
                continue
            command, env, log_path = pending.pop(0)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open("w")
            child_env = env.copy()
            child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(command, env=child_env, stdout=stream, stderr=subprocess.STDOUT)
            active[gpu] = (process, log_path, stream)
        time.sleep(10)
        for gpu, (process, log_path, stream) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            stream.close()
            if returncode:
                failed.append(f"{log_path} (exit {returncode})")
            del active[gpu]
        if failed:
            for process, _, stream in active.values():
                process.terminate()
                stream.close()
            raise RuntimeError("queued jobs failed: " + ", ".join(failed))


def oof_env(candidate: Candidate, fold: int) -> dict[str, str]:
    values: dict[str, Any] = dict(
        ROOT=ROOT,
        OUT_ROOT=OOF,
        CPSAM_MODEL="cpsam_models/cpsam_v2",
        TAG=candidate.tag,
        FOLD=fold,
        NFOLDS=5,
        FOLD_SEED=2026,
        SEED=candidate.seed,
        YSRC="slidesmooth",
        EVAL_D8=0,
        FINAL_D8=1,
        EPOCHS=25,
        CROP=256,
        K_CROPS=1,
        FEATURE_AGG=0,
        EVAL_IMAGE_BS=8,
        POOL="mean",
        LOSS_MODE="multitask",
        BS=48,
        ENC_LR=candidate.enc_lr,
        HEAD_LR=candidate.head_lr,
    )
    if candidate.encoder:
        values["ENCODER_INIT"] = candidate.encoder
    return env_with(**values)


def result_path(candidate: Candidate, fold: int) -> Path:
    run_name = (
        f"{candidate.tag}_f{fold}_seed{candidate.seed}_c256k1_"
        "mean_multitask_slidesmooth"
    )
    return OOF / run_name / "result.json"


def read_metrics(candidate: Candidate, fold: int = 0) -> Candidate:
    path = result_path(candidate, fold)
    payload = json.loads(path.read_text())
    metrics = payload.get("final_metrics")
    if not metrics:
        raise RuntimeError(f"missing final metrics in {path}")
    candidate.metrics = [float(value) for value in metrics]
    return candidate


def launch_oof(candidates: list[Candidate], folds: list[int], prefix: str) -> None:
    jobs = []
    for candidate in candidates:
        for fold in folds:
            jobs.append(
                (
                    [PYTHON, "code/train_oof_h200.py"],
                    oof_env(candidate, fold),
                    LOGS / f"{prefix}_{candidate.tag}_f{fold}.log",
                )
            )
    launch_queue(jobs)


def existing_reference_candidates() -> list[Candidate]:
    return [
        Candidate(
            tag="jepa_small_mean",
            encoder="experiments/domain_jepa_small/encoder_latest.pt",
            source="simultaneous_small_standard",
        ),
        Candidate(
            tag="jepa_small_headlr2e3",
            encoder="experiments/domain_jepa_small/encoder_latest.pt",
            head_lr=2e-3,
            source="simultaneous_small_cd_candidate",
        ),
    ]


def master() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%FT%T%z')}] stage 1: eight external->CLEAR representation curricula", flush=True)
    jobs = []
    for index, variant in enumerate(VARIANTS):
        jobs.append(
            (
                index,
                [PYTHON, "code/run_curriculum_hpo.py", "--worker", str(index)],
                env_with(),
                LOGS / f"worker_{variant['key']}.log",
            )
        )
    launch_parallel(jobs)

    print(f"[{time.strftime('%FT%T%z')}] stage 2: common fold-0 encoder screen", flush=True)
    stage2 = [
        Candidate(
            tag=f"cur_{variant['key']}_std",
            encoder=str(OUT / variant["key"] / "clear" / "encoder_latest.pt"),
            source=variant["key"],
        )
        for variant in VARIANTS
    ]
    launch_oof(stage2, [0], "screen")
    for candidate in stage2:
        read_metrics(candidate)

    references = existing_reference_candidates()
    for candidate in references:
        read_metrics(candidate)
    bases = sorted(stage2 + references[:1], key=lambda value: value.overall)
    first_base = bases[0]
    second_base = next(value for value in bases[1:] if value.encoder != first_base.encoder)
    print(
        f"selected representations: {first_base.tag} overall={first_base.overall:.4f}; "
        f"{second_base.tag} overall={second_base.overall:.4f}",
        flush=True,
    )

    print(f"[{time.strftime('%FT%T%z')}] stage 3: dense supervised LR tuning", flush=True)
    hp = [
        ("e5h5", 5e-6, 5e-4),
        ("e5h20", 5e-6, 2e-3),
        ("e10h20", 1e-5, 2e-3),
        ("e20h10", 2e-5, 1e-3),
    ]
    stage3: list[Candidate] = []
    for base in (first_base, second_base):
        short = base.tag.removeprefix("cur_").removesuffix("_std")
        for suffix, enc_lr, head_lr in hp:
            stage3.append(
                Candidate(
                    tag=f"curhpo_{short}_{suffix}",
                    encoder=base.encoder,
                    enc_lr=enc_lr,
                    head_lr=head_lr,
                    source=base.tag,
                )
            )
    launch_oof(stage3, [0], "hpo")
    for candidate in stage3:
        read_metrics(candidate)

    pool = stage2 + stage3 + references
    pool.sort(key=lambda value: value.overall)
    best_overall = pool[0]
    best_cd = min(pool, key=lambda value: value.cd)
    # Confirm four candidates that still need folds 1-4.  This creates exactly
    # 16 jobs (two full 8-GPU waves) and avoids spending the second wave with
    # half the node idle.  Already-complete references remain in the ranking
    # pool but do not consume a confirmation slot.
    priority = [best_overall, best_cd]
    priority += sorted(pool, key=lambda value: value.overall)
    priority += sorted(pool, key=lambda value: value.cd)
    selected: list[Candidate] = []
    for candidate in priority:
        if candidate.tag in {value.tag for value in selected}:
            continue
        if all(result_path(candidate, fold).exists() for fold in range(1, 5)):
            continue
        selected.append(candidate)
        if len(selected) == 4:
            break
    if len(selected) < 4:
        raise RuntimeError(f"could only select {len(selected)} candidates for full OOF")
    print(
        "selected for full OOF: "
        + ", ".join(
            f"{value.tag}(overall={value.overall:.4f},CD={value.cd:.4f})" for value in selected
        ),
        flush=True,
    )

    snapshot = {
        "variants": VARIANTS,
        "fold0": [candidate.__dict__ | {"overall": candidate.overall} for candidate in pool],
        "selected": [candidate.__dict__ | {"overall": candidate.overall} for candidate in selected],
    }
    (OUT / "selection.json").write_text(json.dumps(snapshot, indent=2))

    print(f"[{time.strftime('%FT%T%z')}] stage 4: full 5-fold confirmation", flush=True)
    full_jobs = []
    for candidate in selected:
        for fold in range(1, 5):
            if result_path(candidate, fold).exists():
                continue
            full_jobs.append(
                (
                    [PYTHON, "code/train_oof_h200.py"],
                    oof_env(candidate, fold),
                    LOGS / f"full_{candidate.tag}_f{fold}.log",
                )
            )
    launch_queue(full_jobs)
    run_checked(
        [PYTHON, "code/summarize_oof_h200.py"],
        env_with(),
        LOGS / "final_oof_summary.log",
    )
    snapshot["completed_at"] = time.strftime("%FT%T%z")
    (OUT / "complete.json").write_text(json.dumps(snapshot, indent=2))
    print(f"[{time.strftime('%FT%T%z')}] curriculum HPO complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.worker is not None:
        curriculum_worker(args.worker)
    else:
        master()


if __name__ == "__main__":
    main()
