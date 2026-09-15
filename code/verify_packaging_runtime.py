"""Eight fixed interface/numerical checks; no new training or model selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=["cu126", "cu130"], required=True)
    parser.add_argument("--precision", choices=["fp16", "bf16"], default="fp16")
    args = parser.parse_args()
    root = Path("/workspace/clear-ec-research")
    stage = root / "experiments/region_candidate_20260907/experimental_stage"
    sys.path.insert(0, str(stage))
    import numpy as np
    import SimpleITK as sitk
    import torch
    import inference

    assert torch.cuda.is_available()
    assert torch.__version__.split("+")[0] == "2.11.0"
    assert torch.version.cuda == {"cu126": "12.6", "cu130": "13.0"}[args.label]
    output = Path("/tmp/clearec-package-check") / (args.label + "_" + args.precision)
    output.mkdir(parents=True, exist_ok=False)
    meta = np.load(root / "datasets/clear_ec_final9000_20260904/meta.npz")
    historic = json.loads((root / "experiments/region_candidate_20260907/integration/full_quality/result.json").read_text())
    reference = {r["index"]: np.asarray(r["values"]) for r in historic["records"] if r["precision"] == args.precision}
    dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    records = []
    for index in sorted(reference):
        path = root / "datasets/clear_ec_fullframe_20260904/train_mha" / (str(meta["ids"][index]) + ".mha")
        raw = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        value = inference.predict(raw, stage / "model", "candidate6_full_quality.json", dtype)
        torch.cuda.synchronize()
        relative = np.abs(value-reference[index])/reference[index]
        # Predeclared numerical tolerance; not a new efficacy estimate.
        assert relative.max() <= .005, (index, value.tolist(), reference[index].tolist())
        record = dict(index=index, values=value.tolist(), relative_to_historic=relative.tolist(),
                      seconds=time.monotonic()-start, peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
        records.append(record)
        print(json.dumps(record), flush=True)
    input_root = root / "experiments/region_candidate_20260907/integration/full_quality/input"
    started = time.monotonic()
    run = subprocess.run([sys.executable, str(stage / "inference.py"), "--input", str(input_root),
        "--output", str(output / "entrypoint"), "--model", str(stage / "model"),
        "--config", "candidate6_full_quality.json"] + (["--fp16"] if args.precision == "fp16" else []),
        capture_output=True, text=True, timeout=300, check=True)
    names = ["cell-density", "coefficient-of-variation", "hexagonality"]
    files = list((output / "entrypoint").glob("*"))
    assert {p.name for p in files} == {k + ".json" for k in names}
    values = [json.loads((output / "entrypoint" / (k + ".json")).read_text()) for k in names]
    assert all(type(v) is float and v > 0 for v in values) and 0 <= values[2] <= 1
    np.testing.assert_allclose(values, records[0]["values"], rtol=1e-5, atol=1e-6)
    report = dict(label=args.label, precision=args.precision, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), architecture=torch.cuda.get_device_capability(),
        compiled_arches=torch.cuda.get_arch_list(), records=records,
        entrypoint=dict(values=values, seconds=time.monotonic()-started, stdout=run.stdout),
        passed=True, actual_submission_image=False,
        scope="Identical pinned base runtime and frozen application on H200; not a Grand Challenge target-device test",
        completed_at=time.strftime("%FT%T%z"))
    (output / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(passed=True, label=args.label)), flush=True)


if __name__ == "__main__":
    main()
