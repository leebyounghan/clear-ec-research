"""Verify denied egress, then fresh-process inference on both assigned GPUs."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np

root = Path("/workspace/clear-ec-research/experiments/region_candidate_20260907")
stage = root / "experimental_stage"
output = Path("/tmp/clearec-package-check/offline")
output.mkdir(exist_ok=False)
try:
    connection = socket.create_connection(("1.1.1.1", 443), timeout=3)
except OSError as error:
    denied = str(error)
else:
    connection.close()
    raise AssertionError("Egress still available; not an offline test")


def run(item):
    gpu, precision = item
    started = time.monotonic()
    destination = output / precision
    command = ["python", str(stage / "inference.py"), "--input", str(root / "integration/full_quality/input"),
               "--output", str(destination), "--model", str(stage / "model"),
               "--config", "candidate6_full_quality.json"] + (["--fp16"] if precision == "fp16" else [])
    result = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
        text=True, capture_output=True, check=True, timeout=300)
    names = ["cell-density", "coefficient-of-variation", "hexagonality"]
    assert {p.name for p in destination.iterdir()} == {n + ".json" for n in names}
    values = [json.loads((destination / (name + ".json")).read_text()) for name in names]
    reference = json.loads((output.parent / ("cu126_" + precision) / "result.json").read_text())["entrypoint"]["values"]
    np.testing.assert_allclose(values, reference, rtol=0, atol=0)
    return dict(precision=precision, values=values, seconds=time.monotonic()-started, stdout=result.stdout)


with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(run, [(0, "fp16"), (1, "bf16")]))
report = dict(passed=True, denied_egress_error=denied, records=results,
              scope="Pinned base container with matched dependencies and frozen code, H200; not the exported image on the Grand Challenge device",
              completed_at=time.strftime("%FT%T%z"))
(output / "result.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
