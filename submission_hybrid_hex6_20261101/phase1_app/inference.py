"""Experimental six-member entrypoint; explicit configuration, no promotion."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import SimpleITK as sitk
import torch
from final_candidate_core import build_model, central512, predict_one
from base_inference import load_strict
from region_inference_core import native_views, load_region, predict_views

BASE = ["seed42.pt", "seed77.pt", "seed2026.pt", "metric35.pt", "predbag2_35.pt"]
POLICIES = ("anchor_quality", "full_quality")


def validate(config, name):
    policy = config.get("region_policy")
    if policy not in POLICIES or name != f"candidate6_{policy}.json":
        raise ValueError("Unexpected candidate policy/configuration")
    if config.get("members") != BASE + [f"{policy}_seed77.pt"] or config.get("view_batch") != 8:
        raise ValueError("Unexpected member order or view batching")
    scale = np.asarray(config.get("calibration"), dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Invalid calibration")
    return policy, scale


def predict(image, model, config_name, dtype=torch.bfloat16, return_members=False):
    if not torch.cuda.is_available():
        raise RuntimeError("This candidate requires a validated CUDA runtime")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    config = json.loads((model / config_name).read_text())
    policy, scale = validate(config, config_name)
    central = central512(image)
    region = native_views(image, policy)
    net = build_model(model / "cpsam_v2")
    values = []
    for member in BASE:
        load_strict(net, model / member)
        values.append(predict_one(net, central, dtype=dtype))
    load_region(net, model / config["members"][-1], policy)
    values.append(predict_views(net, region, dtype=dtype))
    values = np.asarray(values, dtype=float)
    # Average decoded member predictions; never blend calibrated base5 with raw6.
    result = values.mean(0) * scale
    if not np.isfinite(result).all() or not (result > 0).all():
        raise ValueError("Invalid candidate output")
    del net
    torch.cuda.empty_cache()
    return (result, values) if return_members else result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/input"))
    parser.add_argument("--output", type=Path, default=Path("/output"))
    parser.add_argument("--model", type=Path, default=Path("/opt/ml/model"))
    parser.add_argument("--config", required=True, help="No automatically promoted default branch")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    sockets = json.loads((args.input / "inputs.json").read_text())
    if [x["socket"]["slug"] for x in sockets] != ["corneal-specular-microscopy-image"]:
        raise ValueError("Expected single microscopy input socket")
    paths = list((args.input / "images/corneal-specular-microscopy-image").glob("*.mha"))
    if len(paths) != 1:
        raise ValueError("Expected exactly one MHA")
    outputs = [args.output / f"{key}.json" for key in ("cell-density", "coefficient-of-variation", "hexagonality")]
    if any(p.exists() for p in outputs):
        raise FileExistsError("Refusing to overwrite existing prediction")
    start = time.time()
    image = sitk.GetArrayFromImage(sitk.ReadImage(str(paths[0])))
    value = predict(image, args.model, args.config, torch.float16 if args.fp16 else torch.bfloat16)
    args.output.mkdir(parents=True, exist_ok=True)
    for path, v in zip(outputs, value):
        with path.open("x") as handle:
            json.dump(float(v), handle, allow_nan=False)
    print(json.dumps(dict(seconds=time.time()-start, members=6, output_count=3)), flush=True)


if __name__ == "__main__":
    main()
