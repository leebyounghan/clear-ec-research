"""Four/five-member candidate entrypoint; strict checkpoints and bounded view batches."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from final_candidate_core import LM, LS, build_model, central512, predict_one


MEMBERS = {
    "candidate4.json": ["seed42.pt", "seed77.pt", "seed2026.pt", "metric35.pt"],
    "candidate5.json": ["seed42.pt", "seed77.pt", "seed2026.pt", "metric35.pt", "predbag2_35.pt"],
}


def validate_config(config, name):
    if name not in MEMBERS or config.get("members") != MEMBERS[name] or config.get("view_batch") != 8:
        raise ValueError("Unexpected candidate membership or view batching")
    scale = np.asarray(config.get("calibration"), dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Invalid calibration scales")
    return scale


def load_strict(net, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config.get("pool") != "mean" or config.get("k_crops") not in (1, 2):
        raise ValueError("Unsupported pooling or number of crops")
    if config.get("feature_agg", 0) != 0 or config.get("bag_pool", "mean") != "mean":
        raise ValueError("Feature bags are not prediction bags")
    np.testing.assert_array_equal(config["lm"], LM.tolist())
    np.testing.assert_array_equal(config["ls"], LS.tolist())
    net.load_state_dict(checkpoint["model"], strict=True)
    net.eval()


def predict(image, model_path, config_name="candidate5.json", dtype=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = dtype or (torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16)
    config = json.loads((model_path / config_name).read_text())
    scale = validate_config(config, config_name)
    net = build_model(model_path / "cpsam_v2", device)
    values = []
    for member in config["members"]:
        load_strict(net, model_path / member)
        values.append(predict_one(net, image, dtype=dtype, view_batch=8, device=device))
    result = np.mean(values, axis=0) * scale
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("invalid candidate output")
    del net
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/input"))
    parser.add_argument("--output", type=Path, default=Path("/output"))
    parser.add_argument("--model", type=Path, default=Path("/opt/ml/model"))
    parser.add_argument("--config", default="candidate5.json")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    socket = json.loads((args.input / "inputs.json").read_text())
    assert sorted(item["socket"]["slug"] for item in socket) == ["corneal-specular-microscopy-image"]
    images = list((args.input / "images/corneal-specular-microscopy-image").glob("*.mha"))
    if len(images) != 1:
        raise ValueError("expected exactly one MHA image")
    image = central512(sitk.GetArrayFromImage(sitk.ReadImage(str(images[0]))))
    started = time.time()
    values = predict(image, args.model, args.config, torch.float16 if args.fp16 else None)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in zip(("cell-density", "coefficient-of-variation", "hexagonality"), values):
        destination = args.output / f"{name}.json"
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_text(json.dumps(float(value), allow_nan=False))
    print(json.dumps({"seconds": time.time() - started, "members": len(json.loads(
        (args.model / args.config).read_text())["members"]), "output_count": 3}), flush=True)


if __name__ == "__main__":
    main()
