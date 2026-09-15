"""Stage 4/5-member candidates and run their real native entrypoint."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from final_candidate_core import save_json
from plan_final_candidate import scale_fit


ROOT = Path("experiments/deploy_methods_20260905")
OLD = Path("experiments/final_candidate_20260904/staged/model")
METHOD = Path("experiments/methods_20260904")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def copy_verified(source, destination):
    expected = digest(source)
    if destination.exists():
        assert digest(destination) == expected
    else:
        shutil.copy2(source, destination)
    return {"sha256": expected, "bytes": destination.stat().st_size}


def candidate_predictions():
    with np.load(METHOD / "metric35_five_fold_predictions.npz", allow_pickle=True) as metric, \
         np.load(METHOD / "predbag2_35_five_fold_predictions.npz", allow_pickle=True) as bag:
        for key in ("indices", "ids", "target", "folds", "ensemble"):
            assert np.array_equal(metric[key], bag[key])
        old = metric["ensemble"].astype(float)
        metric_pred = metric["standalone"].astype(float)
        bag_pred = bag["standalone"].astype(float)
        target = metric["target"].astype(float)
    return target, {
        "candidate4": 0.75 * old + 0.25 * metric_pred,
        "candidate5": 0.60 * old + 0.20 * metric_pred + 0.20 * bag_pred,
    }


def main():
    for recipe in ("metric35", "predbag2_35"):
        if not (ROOT / "runs" / recipe / "result.json").exists():
            raise RuntimeError(f"incomplete all-data fit: {recipe}")
    staging = ROOT / "staged"
    model = staging / "model"
    model.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in ("cpsam_v2", "seed42.pt", "seed77.pt", "seed2026.pt"):
        manifest[name] = copy_verified(OLD / name, model / name)
    for recipe in ("metric35", "predbag2_35"):
        name = recipe + ".pt"
        manifest[name] = copy_verified(ROOT / "runs" / recipe / "selected.pt", model / name)
    for source, name in ((Path("submission_candidate_20260905/inference.py"), "inference.py"),
                         (Path("code/final_candidate_core.py"), "final_candidate_core.py")):
        manifest[name] = copy_verified(source, staging / name)

    target, predictions = candidate_predictions()
    common = ["seed42.pt", "seed77.pt", "seed2026.pt"]
    configurations = {
        "candidate4.json": dict(members=common + ["metric35.pt"],
            calibration=scale_fit(predictions["candidate4"], target).tolist()),
        "candidate5.json": dict(members=common + ["metric35.pt", "predbag2_35.pt"],
            calibration=scale_fit(predictions["candidate5"], target).tolist()),
    }
    for name, value in configurations.items():
        save_json(model / name, dict(**value, view_batch=8,
            source="Global scale fitted to historically reused five-fold OOF; not independent validation.",
            no_all_data_validation_score=True))

    metadata = np.load("datasets/clear_ec_final9000_20260904/meta.npz", allow_pickle=True)
    source = Path("datasets/clear_ec_fullframe_20260904/train_mha") / (str(metadata["ids"][0]) + ".mha")
    input_root = ROOT / "interface_smoke" / "input"
    image_dir = input_root / "images" / "corneal-specular-microscopy-image"
    image_dir.mkdir(parents=True, exist_ok=True)
    image = image_dir / "image.mha"
    if not image.exists():
        shutil.copy2(source, image)
    save_json(input_root / "inputs.json", [{"socket": {"slug": "corneal-specular-microscopy-image"}}])
    smokes = {}
    for config in configurations:
        for precision, extra in (("bf16", []), ("fp16", ["--fp16"])):
            output = ROOT / "interface_smoke" / f"{config.removesuffix('.json')}_{precision}"
            if output.exists():
                raise FileExistsError(f"inspect existing smoke before retry: {output}")
            started = time.time()
            subprocess.run([sys.executable, str(staging / "inference.py"), "--input", str(input_root),
                "--output", str(output), "--model", str(model), "--config", config] + extra,
                check=True, timeout=900, env=dict(os.environ, OMP_NUM_THREADS="4"))
            values = [json.loads((output / f"{name}.json").read_text()) for name in
                      ("cell-density", "coefficient-of-variation", "hexagonality")]
            assert np.isfinite(values).all() and (np.asarray(values) > 0).all()
            smokes[f"{config}:{precision}"] = {"values": values,
                "cold_process_seconds": time.time() - started}
    save_json(ROOT / "package_manifest.json", dict(files=manifest, candidates=configurations,
        interface_smoke=smokes, staged=str(staging), completed_at=time.strftime("%FT%T%z"),
        status="Staged/native-tested only; not container-built, target-device-tested, uploaded or submitted."))


if __name__ == "__main__":
    main()
