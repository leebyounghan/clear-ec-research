"""Fetch only the selected immutable stage from the read-only PVC pod."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

DEST = Path(__file__).resolve().parents[1] / "submission_candidate_20260909"
SOURCE = "/workspace/clear-ec-research/experiments/region_candidate_20260907/experimental_stage"
KUBE = ["kubectl", "-n", "p-ajou-mlxp-project", "exec", "clear-ec-package-reader", "--"]


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def fetch(item):
    name, expected = item
    destination = DEST / name if name.startswith("model/") else DEST / "app" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert destination.stat().st_size == expected["bytes"] and sha(destination) == expected["sha256"], name
        print(f"Already verified: {name}", flush=True)
        return name
    partial = destination.with_name(destination.name + ".partial")
    # Resume only this script's partial file; trust it only after full SHA256.
    started = time.monotonic()
    attempts = 0
    last_report = 0
    with partial.open("ab") as output:
        if name == "model/cpsam_v2":
            # Public original is byte-identical: upstream x-linked-etag matches
            # the frozen stage SHA256. Final full-file hashing still gates use.
            offset = partial.stat().st_size
            request = urllib.request.Request(
                "https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam_v2",
                headers={"Range": f"bytes={offset}-"})
            with urllib.request.urlopen(request, timeout=45) as response:
                assert response.status == 206
                assert response.headers["Content-Range"].startswith(f"bytes {offset}-")
                shutil.copyfileobj(response, output, 8 * 1024**2)
                output.flush()
        while partial.stat().st_size < expected["bytes"]:
            offset = partial.stat().st_size
            if time.monotonic() - started > 5000:
                raise TimeoutError(f"Transfer deadline reached: {name}")
            count = min(8 * 1024**2, expected["bytes"] - offset)
            command = KUBE + ["dd", "if=" + SOURCE + "/" + name, "bs=1048576",
                              "iflag=skip_bytes,count_bytes", f"skip={offset}",
                              f"count={count}", "status=none"]
            process = subprocess.Popen(command, stdout=output,
                env=dict(os.environ, KUBECTL_REMOTE_COMMAND_WEBSOCKETS="false"))
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=15)
            except BaseException:
                process.terminate()
                process.wait(timeout=15)
                raise
            size = partial.stat().st_size
            assert size <= offset + count, name
            if size == offset:
                attempts += 1
                if attempts >= 5:
                    raise RuntimeError(f"Repeated transfer failures: {name}")
            else:
                attempts = 0
            if time.monotonic() - last_report >= 30:
                elapsed = time.monotonic() - started
                print(f"Transfer {name}: {size}/{expected['bytes']} bytes ({elapsed:.0f}s)", flush=True)
                last_report = time.monotonic()
    assert partial.stat().st_size == expected["bytes"], name
    assert sha(partial) == expected["sha256"], name
    partial.chmod(0o644)
    partial.rename(destination)
    print(f"Verified: {name} ({expected['bytes']} bytes, {time.monotonic()-started:.1f}s)", flush=True)
    return name


def main():
    files = json.loads((DEST / "provenance/stage_manifest.json").read_text())["files"]
    selected = {k: v for k, v in files.items() if "anchor_quality" not in k}
    missing = sum(v["bytes"] for k, v in selected.items()
                  if not (DEST / k if k.startswith("model/") else DEST / "app" / k).exists())
    assert shutil.disk_usage(DEST).free > missing + 20 * 1024**3, "Insufficient disk headroom"
    small = [(k, v) for k, v in selected.items() if v["bytes"] < 1024**2]
    large = [(k, v) for k, v in selected.items() if v["bytes"] >= 1024**2]
    for item in small:
        fetch(item)
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
        list(pool.map(fetch, large))
    report = dict(files=selected, verified_at=time.strftime("%FT%T%z"),
                  policy="full_quality", deployment_validated=False)
    with (DEST / "provenance/fetch_verified.json").open("x") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
