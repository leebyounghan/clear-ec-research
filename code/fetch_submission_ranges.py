"""Resumable small parallel ranges; assemble only after all ranges arrive."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1] / "submission_candidate_20260909"
SOURCE = "/workspace/clear-ec-research/experiments/region_candidate_20260907/experimental_stage/model/"
KUBE = ["kubectl", "-n", "p-ajou-mlxp-project", "exec", "clear-ec-package-reader", "--"]
PARTS = ROOT / "provenance/transfer_ranges"
CHUNK = 2 * 1024**2


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(data)
    return h.hexdigest()


def main():
    PARTS.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / "provenance/stage_manifest.json").read_text())["files"]
    selected = {k[6:]: v for k, v in manifest.items() if k.startswith("model/") and "anchor_quality" not in k}
    plan_path = PARTS / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        plan = {}
        for name, item in selected.items():
            final = ROOT / "model" / name
            if final.exists():
                assert sha(final) == item["sha256"], name
                continue
            partial = final.with_name(name + ".partial")
            prefix = partial.stat().st_size if partial.exists() else 0
            plan[name] = dict(prefix=prefix, **item)
        with plan_path.open("x") as handle:
            json.dump(plan, handle, indent=2)
    jobs = []
    for name, item in plan.items():
        (PARTS / name).mkdir(exist_ok=True)
        final = ROOT / "model" / name
        if final.exists():
            assert sha(final) == item["sha256"]
            continue
        for offset in range(item["prefix"], item["bytes"], CHUNK):
            jobs.append((name, offset, min(CHUNK, item["bytes"]-offset)))
    # Keep all connections useful until the final range, regardless of file size.
    started = time.monotonic()

    def fetch(job):
        name, offset, count = job
        path = PARTS / name / f"{offset}.part"
        current = path.stat().st_size if path.exists() else 0
        assert current <= count
        failures = 0
        while current < count:
            remaining = count - current
            cmd = KUBE + ["dd", "if=" + SOURCE + name, "bs=1048576",
                "iflag=skip_bytes,count_bytes", f"skip={offset+current}", f"count={remaining}", "status=none"]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=dict(os.environ, KUBECTL_REMOTE_COMMAND_WEBSOCKETS="false"))
            try:
                content, errors = process.communicate(timeout=25)
            except subprocess.TimeoutExpired:
                process.terminate()
                content, errors = process.communicate(timeout=10)
            assert len(content) <= remaining
            if not content:
                failures += 1
                if failures >= 8:
                    raise RuntimeError(f"No progress for {name}:{offset}: {errors[-200:]!r}")
                continue
            with path.open("ab") as handle:
                handle.write(content)
            current += len(content)
        return count

    total = sum(job[2] for job in jobs)
    cached = sum(count for name, offset, count in jobs
                 if (PARTS / name / f"{offset}.part").exists()
                 and (PARTS / name / f"{offset}.part").stat().st_size == count)
    received = 0
    last = 0
    with ThreadPoolExecutor(max_workers=32) as pool:
        for future in as_completed([pool.submit(fetch, job) for job in jobs]):
            received += future.result()
            elapsed = time.monotonic()-started
            if elapsed-last >= 20:
                print(f"Ranges complete: {received}/{total} bytes; {max(0,received-cached)/max(elapsed,1)/1024**2:.2f} new MiB/s; {elapsed:.0f}s", flush=True)
                last = elapsed
    for name, item in plan.items():
        final = ROOT / "model" / name
        if final.exists():
            continue
        partial = final.with_name(name + ".partial")
        current = partial.stat().st_size if partial.exists() else 0
        assert item["prefix"] <= current <= item["bytes"]
        with partial.open("ab") as output:
            for offset in range(item["prefix"], item["bytes"], CHUNK):
                count = min(CHUNK, item["bytes"]-offset)
                path = PARTS / name / f"{offset}.part"
                assert path.stat().st_size == count
                if offset+count <= current:
                    continue
                with path.open("rb") as source:
                    source.seek(max(0, current-offset))
                    shutil.copyfileobj(source, output)
                current = offset+count
        assert partial.stat().st_size == item["bytes"] and sha(partial) == item["sha256"], name
        partial.chmod(0o644)
        partial.rename(final)
        print(f"Assembled and SHA256 verified: {name}", flush=True)
    subprocess.run(["python3", str(ROOT.parent / "code/fetch_submission_package.py")], check=True)


if __name__ == "__main__":
    main()
